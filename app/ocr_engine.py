from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger("paddleocr.engine")

_ocr: Any | None = None
_structure: Any | None = None


def _build_ocr():
    from paddleocr import PaddleOCR

    kwargs: dict[str, Any] = {
        "use_doc_orientation_classify": True,
        "use_textline_orientation": True,
        "use_doc_unwarping": False,
        "lang": settings.lang,
        "device": settings.device,
    }
    if settings.ocr_version:
        kwargs["ocr_version"] = settings.ocr_version
    return PaddleOCR(**kwargs)


def _build_structure():
    from paddleocr import PPStructureV3

    return PPStructureV3(
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_formula_recognition=False,
        use_seal_recognition=False,
        use_chart_recognition=False,
        device=settings.device,
    )


def warm_engines() -> None:
    global _ocr, _structure
    logger.info("Loading PaddleOCR engine device=%s lang=%s", settings.device, settings.lang)
    _ocr = _build_ocr()
    logger.info("Loading PPStructureV3 engine")
    _structure = _build_structure()
    logger.info("OCR engines ready model=%s", settings.model_label)


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
        # Some versions nest under "res"
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


def recognize_text(image_bytes: bytes, mime: str = "image/png") -> dict[str, Any]:
    started = time.perf_counter()
    tmp_path: Path | None = None
    try:
        tmp_path = _write_temp_upload(image_bytes, mime)

        ocr = get_ocr()
        results = ocr.predict(str(tmp_path))
        all_lines: list[str] = []
        line_count = 0
        for res in results or []:
            payload = _result_to_dict(res)
            text, count = _join_rec_texts(payload)
            if text:
                all_lines.append(text)
            line_count += count

        return {
            "text": "\n".join(all_lines).strip(),
            "model": settings.model_label,
            "durationMs": int((time.perf_counter() - started) * 1000),
            "lineCount": line_count,
        }
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _extract_structure_text(res: Any) -> str:
    # Prefer markdown if available on the result object
    for attr in ("markdown", "to_markdown"):
        value = getattr(res, attr, None)
        if callable(value):
            try:
                md = value()
                if isinstance(md, str) and md.strip():
                    return md.strip()
            except Exception:
                pass
        elif isinstance(value, str) and value.strip():
            return value.strip()
        elif isinstance(value, dict):
            for key in ("markdown", "text", "md"):
                if isinstance(value.get(key), str) and value[key].strip():
                    return value[key].strip()

    payload = _result_to_dict(res)
    for key in ("markdown", "table_html", "tableHtml", "html"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    # Nested parsing results
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


def recognize_table(image_bytes: bytes, mime: str = "image/png") -> dict[str, Any]:
    started = time.perf_counter()
    tmp_path: Path | None = None
    try:
        tmp_path = _write_temp_upload(image_bytes, mime)

        structure = get_structure()
        results = structure.predict(str(tmp_path))
        chunks: list[str] = []
        for res in results or []:
            text = _extract_structure_text(res)
            if text:
                chunks.append(text)

        if not chunks:
            # Fallback to plain OCR if structure produced nothing
            fallback = recognize_text(image_bytes, mime=mime)
            return {
                "text": fallback["text"],
                "model": f"{settings.model_label}+structure-fallback",
                "durationMs": int((time.perf_counter() - started) * 1000),
            }

        return {
            "text": "\n\n".join(chunks).strip(),
            "model": f"{settings.model_label}+structure",
            "durationMs": int((time.perf_counter() - started) * 1000),
        }
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
