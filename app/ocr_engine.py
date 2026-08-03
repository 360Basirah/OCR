from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger("paddleocr.engine")

_vl: Any | None = None


def _build_vl():
    from paddleocr import PaddleOCRVL

    kwargs: dict[str, Any] = {
        "pipeline_version": settings.pipeline_version,
        "device": settings.device,
        "use_doc_orientation_classify": True,
        "use_doc_unwarping": False,
    }
    # Local vLLM accelerates VL recognition; images stay on localhost (not sent to cloud).
    if settings.uses_vlm_server:
        kwargs["vl_rec_backend"] = settings.vl_rec_backend
        kwargs["vl_rec_server_url"] = settings.vl_rec_server_url

    return PaddleOCRVL(**kwargs)


def warm_engines() -> None:
    global _vl
    # Surface PaddleOCR / PaddleX progress in the uvicorn console.
    for name in ("ppocr", "paddlex", "paddleocr"):
        logging.getLogger(name).setLevel(logging.INFO)

    logger.info(
        "Loading PaddleOCR-VL pipeline_version=%s device=%s backend=%s server=%s",
        settings.pipeline_version,
        settings.device,
        settings.vl_rec_backend or "in-process",
        settings.vl_rec_server_url or "-",
    )
    _vl = _build_vl()
    logger.info("OCR engine ready model=%s", settings.model_label)


def get_vl():
    global _vl
    if _vl is None:
        _vl = _build_vl()
    return _vl


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


def _extract_vl_text(res: Any) -> str:
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


def _recognize(image_bytes: bytes, mime: str, model_suffix: str = "") -> dict[str, Any]:
    started = time.perf_counter()
    tmp_path: Path | None = None
    logger.info(
        "ocr_start mime=%s bytes=%s model=%s backend=%s device=%s",
        mime,
        len(image_bytes),
        settings.model_label + model_suffix,
        settings.vl_rec_backend or "in-process",
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
        vl = get_vl()
        logger.info(
            "ocr_stage=engine_ready elapsed_ms=%s "
            "(pipeline: orientation → layout → VL recognition → markdown)",
            int((time.perf_counter() - t1) * 1000),
        )

        t2 = time.perf_counter()
        logger.info("ocr_stage=predict_begin path=%s", tmp_path.name)
        results = vl.predict(str(tmp_path))
        predict_ms = int((time.perf_counter() - t2) * 1000)
        result_count = len(results) if results is not None else 0
        logger.info(
            "ocr_stage=predict_done results=%s elapsed_ms=%s",
            result_count,
            predict_ms,
        )

        t3 = time.perf_counter()
        chunks: list[str] = []
        for idx, res in enumerate(results or []):
            text = _extract_vl_text(res)
            if text:
                chunks.append(text)
                logger.info(
                    "ocr_stage=extract_page page=%s chars=%s preview=%r",
                    idx,
                    len(text),
                    _text_preview(text),
                )
            else:
                logger.info("ocr_stage=extract_page page=%s chars=0 (empty)", idx)

        text = "\n\n".join(chunks).strip()
        model = settings.model_label if not model_suffix else f"{settings.model_label}{model_suffix}"
        duration_ms = int((time.perf_counter() - started) * 1000)
        line_count = _count_lines(text)
        logger.info(
            "ocr_done model=%s duration_ms=%s predict_ms=%s extract_ms=%s "
            "line_count=%s chars=%s",
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
            "durationMs": duration_ms,
            "lineCount": line_count,
        }
    except Exception:
        logger.exception(
            "ocr_failed mime=%s bytes=%s elapsed_ms=%s",
            mime,
            len(image_bytes),
            int((time.perf_counter() - started) * 1000),
        )
        raise
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
            logger.info("ocr_stage=cleanup temp removed")


def recognize_text(image_bytes: bytes, mime: str = "image/png") -> dict[str, Any]:
    logger.info("recognize_text requested")
    return _recognize(image_bytes, mime)


def recognize_table(image_bytes: bytes, mime: str = "image/png") -> dict[str, Any]:
    # VL-1.6 already handles tables/layout; same pipeline as plain OCR.
    logger.info("recognize_table requested (same VL pipeline)")
    return _recognize(image_bytes, mime, model_suffix="+vl")
