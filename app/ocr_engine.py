from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from app.config import settings

logger = logging.getLogger("paddleocr.engine")

PipelineName = Literal["vl", "ocr", "structure"]

_vl: Any | None = None
_ocr: Any | None = None
_structure: Any | None = None


def _ensure_dynamic_paddle_mode() -> None:
    """
    Paddle can run in static graph mode, but PaddleOCR-VL may hit code paths
    that are incompatible with static mode (e.g. logging `int(Tensor)`).

    Force dynamic/eager mode before we instantiate any PaddleOCR pipelines.
    """
    try:
        import paddle

        if not paddle.in_dynamic_mode():
            paddle.disable_static()
            logger.info("Paddle forced into dynamic mode (disable_static).")
    except Exception:
        # If Paddle import fails, let the existing error surface later.
        logger.exception("Failed to ensure Paddle dynamic mode.")


# Important: do this at import time so it affects any background warm threads too.
_ensure_dynamic_paddle_mode()

OCR_MODEL_LABEL = "pp-ocrv6:medium"
STRUCTURE_MODEL_LABEL = "pp-structurev3"


def _build_vl():
    from paddleocr import PaddleOCRVL

    kwargs: dict[str, Any] = {
        "pipeline_version": settings.pipeline_version,
        "device": settings.device,
        "use_doc_orientation_classify": True,
        "use_doc_unwarping": settings.use_doc_unwarping,
        "use_layout_detection": True,
    }
    # Local vLLM accelerates VL recognition; images stay on localhost (not sent to cloud).
    if settings.uses_vlm_server:
        kwargs["vl_rec_backend"] = settings.vl_rec_backend
        kwargs["vl_rec_server_url"] = settings.vl_rec_server_url

    return PaddleOCRVL(**kwargs)


def _build_ocr():
    from paddleocr import PaddleOCR

    return PaddleOCR(
        device=settings.device,
        ocr_version="PP-OCRv6",
        text_detection_model_name="PP-OCRv6_medium_det",
        text_recognition_model_name="PP-OCRv6_medium_rec",
        use_doc_orientation_classify=True,
        use_doc_unwarping=settings.use_doc_unwarping,
        use_textline_orientation=True,
    )


def _build_structure():
    from paddleocr import PPStructureV3

    return PPStructureV3(
        device=settings.device,
        lang=settings.structure_lang,
        use_doc_orientation_classify=True,
        use_doc_unwarping=settings.use_doc_unwarping,
        use_textline_orientation=True,
    )


def warm_engines() -> None:
    global _vl, _ocr, _structure
    # Surface PaddleOCR / PaddleX progress in the uvicorn console.
    for name in ("ppocr", "paddlex", "paddleocr"):
        logging.getLogger(name).setLevel(logging.INFO)

    # Defensive: warm might run after other code initialized Paddle.
    _ensure_dynamic_paddle_mode()

    if settings.warm_vl:
        logger.info(
            "Loading PaddleOCR-VL pipeline_version=%s device=%s unwarping=%s "
            "backend=%s server=%s",
            settings.pipeline_version,
            settings.device,
            settings.use_doc_unwarping,
            settings.vl_rec_backend or "in-process",
            settings.vl_rec_server_url or "-",
        )
        _vl = _build_vl()
        logger.info("VL engine ready model=%s", settings.model_label)
    else:
        logger.info("Skipping VL warm (PADDLEOCR_WARM_VL=false)")

    if settings.warm_ocr:
        logger.info(
            "Loading PP-OCRv6 medium det+rec device=%s unwarping=%s",
            settings.device,
            settings.use_doc_unwarping,
        )
        _ocr = _build_ocr()
        logger.info("OCR engine ready model=%s", OCR_MODEL_LABEL)
    else:
        logger.info("Skipping PP-OCRv6 warm (PADDLEOCR_WARM_OCR=false)")

    if settings.warm_structure:
        logger.info(
            "Loading PP-StructureV3 lang=%s device=%s unwarping=%s",
            settings.structure_lang,
            settings.device,
            settings.use_doc_unwarping,
        )
        _structure = _build_structure()
        logger.info("Structure engine ready model=%s", STRUCTURE_MODEL_LABEL)
    else:
        logger.info("Skipping StructureV3 warm (PADDLEOCR_WARM_STRUCTURE=false)")


def get_vl():
    global _vl
    if _vl is None:
        _vl = _build_vl()
    return _vl


def get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = _build_ocr()
    return _ocr


def get_structure():
    global _structure
    if _structure is None:
        _structure = _build_structure()
    return _structure


def _result_to_dict(res: Any) -> dict[str, Any]:
    if hasattr(res, "json") and isinstance(res.json, dict):
        payload = res.json
        if "res" in payload and isinstance(payload["res"], dict):
            return payload["res"]
        return payload
    if isinstance(res, dict):
        if "res" in res and isinstance(res["res"], dict):
            return res["res"]
        return res
    return {}


def _join_rec_texts(payload: dict[str, Any]) -> tuple[str, int]:
    texts = payload.get("rec_texts") or []
    if not isinstance(texts, list):
        texts = []
    lines = [str(t).strip() for t in texts if str(t).strip()]
    return "\n".join(lines), len(lines)


def _markdown_from_attr(res: Any) -> str | None:
    value = getattr(res, "markdown", None)
    if callable(value):
        try:
            md = value()
            if isinstance(md, str) and md.strip():
                return md.strip()
            if isinstance(md, dict):
                texts = md.get("markdown_texts")
                if isinstance(texts, str) and texts.strip():
                    return texts.strip()
                if isinstance(texts, list):
                    joined = "\n\n".join(str(t).strip() for t in texts if str(t).strip())
                    if joined:
                        return joined
        except Exception:
            pass
    elif isinstance(value, str) and value.strip():
        return value.strip()
    elif isinstance(value, dict):
        texts = value.get("markdown_texts")
        if isinstance(texts, str) and texts.strip():
            return texts.strip()
        if isinstance(texts, list):
            joined = "\n\n".join(str(t).strip() for t in texts if str(t).strip())
            if joined:
                return joined
        for key in ("markdown", "text", "md"):
            if isinstance(value.get(key), str) and value[key].strip():
                return value[key].strip()
    return None


def _extract_structured_text(res: Any) -> str:
    md = _markdown_from_attr(res)
    if md:
        return md

    to_md = getattr(res, "to_markdown", None)
    if callable(to_md):
        try:
            value = to_md()
            if isinstance(value, str) and value.strip():
                return value.strip()
        except Exception:
            pass

    payload = _result_to_dict(res)
    for key in ("markdown", "table_html", "tableHtml", "html"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, dict):
            texts = val.get("markdown_texts")
            if isinstance(texts, str) and texts.strip():
                return texts.strip()
            if isinstance(texts, list):
                joined = "\n\n".join(str(t).strip() for t in texts if str(t).strip())
                if joined:
                    return joined

    parsing = payload.get("parsing_res_list") or payload.get("layout_parsing_result")
    if isinstance(parsing, list):
        chunks: list[str] = []
        for item in parsing:
            if not isinstance(item, dict):
                continue
            for key in ("content", "text", "markdown", "html"):
                if isinstance(item.get(key), str) and item[key].strip():
                    chunks.append(item[key].strip())
                    break
        if chunks:
            return "\n\n".join(chunks)

    text, _ = _join_rec_texts(payload.get("overall_ocr_res") or payload)
    return text


def _extract_classic_ocr_text(res: Any) -> str:
    """Extract text from PP-OCRv6 / general OCR pipeline Result objects."""
    md = _markdown_from_attr(res)
    if md:
        return md

    payload = _result_to_dict(res)
    text, _ = _join_rec_texts(payload)
    if text:
        return text

    # Nested overall_ocr_res style
    nested = payload.get("overall_ocr_res")
    if isinstance(nested, dict):
        text, _ = _join_rec_texts(nested)
        if text:
            return text

    # Fallback: rec_texts at top level via json attribute variants
    if hasattr(res, "json"):
        data = res.json
        if isinstance(data, dict):
            text, _ = _join_rec_texts(data.get("res") if isinstance(data.get("res"), dict) else data)
            if text:
                return text

    return _extract_structured_text(res)


def _count_lines(text: str) -> int:
    return len([line for line in text.splitlines() if line.strip()])


_MIME_TO_SUFFIX = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
}


def _write_temp_upload(data: bytes, mime: str) -> Path:
    suffix = _MIME_TO_SUFFIX.get(mime, ".png")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        return Path(tmp.name)


def _text_preview(text: str, limit: int = 240) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def _run_pipeline(
    image_bytes: bytes,
    mime: str,
    *,
    pipeline: PipelineName,
    model: str,
    get_engine,
    extract_text,
    stage_hint: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    tmp_path: Path | None = None
    logger.info(
        "ocr_start pipeline=%s mime=%s bytes=%s model=%s device=%s",
        pipeline,
        mime,
        len(image_bytes),
        model,
        settings.device,
    )
    try:
        t0 = time.perf_counter()
        tmp_path = _write_temp_upload(image_bytes, mime)
        logger.info(
            "ocr_stage=write_temp path=%s elapsed_ms=%s",
            tmp_path.name,
            int((time.perf_counter() - t0) * 1000),
        )

        t1 = time.perf_counter()
        engine = get_engine()
        logger.info(
            "ocr_stage=engine_ready elapsed_ms=%s (%s)",
            int((time.perf_counter() - t1) * 1000),
            stage_hint,
        )

        t2 = time.perf_counter()
        logger.info("ocr_stage=predict_begin pipeline=%s path=%s", pipeline, tmp_path.name)
        results = engine.predict(str(tmp_path))
        predict_ms = int((time.perf_counter() - t2) * 1000)
        result_count = len(results) if results is not None else 0
        logger.info(
            "ocr_stage=predict_done pipeline=%s results=%s elapsed_ms=%s",
            pipeline,
            result_count,
            predict_ms,
        )

        t3 = time.perf_counter()
        chunks: list[str] = []
        for idx, res in enumerate(results or []):
            text = extract_text(res)
            if text:
                chunks.append(text)
                logger.info(
                    "ocr_stage=extract_page pipeline=%s page=%s chars=%s preview=%r",
                    pipeline,
                    idx,
                    len(text),
                    _text_preview(text),
                )
            else:
                logger.info(
                    "ocr_stage=extract_page pipeline=%s page=%s chars=0 (empty)",
                    pipeline,
                    idx,
                )

        text = "\n\n".join(chunks).strip()
        duration_ms = int((time.perf_counter() - started) * 1000)
        line_count = _count_lines(text)
        logger.info(
            "ocr_done pipeline=%s model=%s duration_ms=%s predict_ms=%s extract_ms=%s "
            "line_count=%s chars=%s",
            pipeline,
            model,
            duration_ms,
            predict_ms,
            int((time.perf_counter() - t3) * 1000),
            line_count,
            len(text),
        )
        return {
            "text": text,
            "model": model,
            "pipeline": pipeline,
            "durationMs": duration_ms,
            "lineCount": line_count,
        }
    except Exception:
        logger.exception(
            "ocr_failed pipeline=%s mime=%s bytes=%s elapsed_ms=%s",
            pipeline,
            mime,
            len(image_bytes),
            int((time.perf_counter() - started) * 1000),
        )
        raise
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
            logger.info("ocr_stage=cleanup temp removed")


def recognize_with_pipeline(
    image_bytes: bytes,
    mime: str = "image/png",
    pipeline: PipelineName = "vl",
) -> dict[str, Any]:
    if pipeline == "vl":
        return _run_pipeline(
            image_bytes,
            mime,
            pipeline="vl",
            model=settings.model_label,
            get_engine=get_vl,
            extract_text=_extract_structured_text,
            stage_hint="orientation → layout → VL recognition → markdown",
        )
    if pipeline == "ocr":
        return _run_pipeline(
            image_bytes,
            mime,
            pipeline="ocr",
            model=OCR_MODEL_LABEL,
            get_engine=get_ocr,
            extract_text=_extract_classic_ocr_text,
            stage_hint="orientation → PP-OCRv6 medium det+rec",
        )
    if pipeline == "structure":
        return _run_pipeline(
            image_bytes,
            mime,
            pipeline="structure",
            model=f"{STRUCTURE_MODEL_LABEL}:{settings.structure_lang}",
            get_engine=get_structure,
            extract_text=_extract_structured_text,
            stage_hint="orientation → PP-StructureV3 layout/tables",
        )
    raise ValueError(f"Unknown pipeline: {pipeline}")


def recognize_text(
    image_bytes: bytes,
    mime: str = "image/png",
    pipeline: PipelineName = "vl",
) -> dict[str, Any]:
    logger.info("recognize_text requested pipeline=%s", pipeline)
    return recognize_with_pipeline(image_bytes, mime, pipeline=pipeline)


def recognize_table(
    image_bytes: bytes,
    mime: str = "image/png",
    pipeline: PipelineName = "structure",
) -> dict[str, Any]:
    logger.info("recognize_table requested pipeline=%s", pipeline)
    return recognize_with_pipeline(image_bytes, mime, pipeline=pipeline)
