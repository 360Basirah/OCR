from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.config import settings
from app.gpu_gate import run_on_gpu
from app.middleware import MaxBodySizeMiddleware, RequestContextMiddleware, SecurityHeadersMiddleware
from app.ocr_engine import recognize_table, recognize_text, warm_engines
from app.security import rate_limit_key, require_api_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("paddleocr")

limiter = Limiter(key_func=rate_limit_key, default_limits=[settings.rate_limit_global])

OcrPipeline = Literal["vl", "ocr"]
TablePipeline = Literal["structure", "vl"]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    warm_engines()
    yield


app = FastAPI(
    title="Basirah PaddleOCR Service",
    version="1.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.docs_enabled else None,
    redoc_url="/redoc" if settings.docs_enabled else None,
    openapi_url="/openapi.json" if settings.docs_enabled else None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(MaxBodySizeMiddleware)
app.add_middleware(RequestContextMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)


def _sniff_upload(data: bytes, content_type: str | None) -> str:
    if not data:
        raise HTTPException(status_code=400, detail="Empty file upload")

    detected: str | None = None
    if data.startswith(b"%PDF"):
        detected = "application/pdf"
    elif data.startswith(b"\xff\xd8\xff"):
        detected = "image/jpeg"
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        detected = "image/png"
    elif len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        detected = "image/webp"

    if detected is None:
        raise HTTPException(
            status_code=400,
            detail="Unsupported or spoofed file type. Allowed: JPEG, PNG, WebP, PDF.",
        )

    if content_type and content_type.split(";")[0].strip().lower() not in {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "application/pdf",
        "application/octet-stream",
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Content-Type not allowed: {content_type}. "
                "Allowed: image/jpeg, image/png, image/webp, application/pdf."
            ),
        )

    return detected


async def _read_upload(file: UploadFile) -> tuple[bytes, str]:
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Payload too large. Max upload is {settings.max_upload_bytes} bytes.",
        )
    mime = _sniff_upload(data, file.content_type)
    return data, mime


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


@app.get("/health")
@limiter.limit("120/minute")
async def health(request: Request):
    return {
        "status": "ok",
        "paddleocr": True,
        "device": settings.device,
        "pipeline_version": settings.pipeline_version,
        "model": settings.model_label,
        "vl_rec_backend": settings.vl_rec_backend or "in-process",
        "vl_rec_server_url": settings.vl_rec_server_url,
        "max_concurrent": settings.max_concurrent,
        "use_doc_unwarping": settings.use_doc_unwarping,
        "structure_lang": settings.structure_lang,
        "warm": {
            "vl": settings.warm_vl,
            "ocr": settings.warm_ocr,
            "structure": settings.warm_structure,
        },
    }


@app.post("/ocr")
@limiter.limit(settings.rate_limit_ocr)
async def ocr(
    request: Request,
    file: UploadFile = File(...),
    pipeline: OcrPipeline = Query(
        "vl",
        description="vl = PaddleOCR-VL-1.6 (default, max accuracy); ocr = PP-OCRv6 medium",
    ),
    _api_key: str = Depends(require_api_key),
):
    data, mime = await _read_upload(file)
    req_id = _request_id(request)
    logger.info(
        "POST /ocr request_id=%s pipeline=%s filename=%s mime=%s bytes=%s",
        req_id,
        pipeline,
        file.filename,
        mime,
        len(data),
    )
    try:
        result, queue_wait_ms = await run_on_gpu(
            recognize_text, data, mime=mime, pipeline=pipeline
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ocr_failed request_id=%s", req_id)
        raise HTTPException(status_code=502, detail=f"OCR failed: {exc}") from exc

    result = {**result, "queueWaitMs": queue_wait_ms, "requestId": req_id}
    logger.info(
        "POST /ocr ok request_id=%s pipeline=%s duration_ms=%s queue_wait_ms=%s line_count=%s",
        req_id,
        result.get("pipeline"),
        result.get("durationMs"),
        queue_wait_ms,
        result.get("lineCount"),
    )
    return result


@app.post("/ocr-table")
@limiter.limit(settings.rate_limit_ocr_table)
async def ocr_table(
    request: Request,
    file: UploadFile = File(...),
    pipeline: TablePipeline = Query(
        "structure",
        description="structure = PP-StructureV3 (default); vl = PaddleOCR-VL-1.6",
    ),
    _api_key: str = Depends(require_api_key),
):
    data, mime = await _read_upload(file)
    req_id = _request_id(request)
    logger.info(
        "POST /ocr-table request_id=%s pipeline=%s filename=%s mime=%s bytes=%s",
        req_id,
        pipeline,
        file.filename,
        mime,
        len(data),
    )
    try:
        result, queue_wait_ms = await run_on_gpu(
            recognize_table, data, mime=mime, pipeline=pipeline
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ocr_table_failed request_id=%s", req_id)
        raise HTTPException(status_code=502, detail=f"Table OCR failed: {exc}") from exc

    result = {**result, "queueWaitMs": queue_wait_ms, "requestId": req_id}
    logger.info(
        "POST /ocr-table ok request_id=%s pipeline=%s duration_ms=%s queue_wait_ms=%s line_count=%s",
        req_id,
        result.get("pipeline"),
        result.get("durationMs"),
        queue_wait_ms,
        result.get("lineCount"),
    )
    return result


@app.exception_handler(413)
async def payload_too_large_handler(_request: Request, _exc):
    return JSONResponse(
        status_code=413,
        content={"detail": f"Payload too large. Max upload is {settings.max_upload_bytes} bytes."},
    )
