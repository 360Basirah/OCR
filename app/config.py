from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load paddleocr-service/.env when present (does not override existing env vars).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
        return parsed if parsed > 0 else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    env: str
    host: str
    port: int
    device: str
    pipeline_version: str
    vl_rec_backend: str | None
    vl_rec_server_url: str | None
    ocr_api_key: str
    rate_limit_ocr: str
    rate_limit_ocr_table: str
    rate_limit_global: str
    max_upload_bytes: int
    allowed_hosts: list[str]
    model_label: str
    max_concurrent: int
    use_doc_unwarping: bool
    structure_lang: str
    warm_vl: bool
    warm_ocr: bool
    warm_structure: bool

    @property
    def is_production(self) -> bool:
        return self.env.lower() == "production"

    @property
    def docs_enabled(self) -> bool:
        return not self.is_production

    @property
    def uses_vlm_server(self) -> bool:
        return bool(self.vl_rec_backend and self.vl_rec_server_url)


def load_settings() -> Settings:
    api_key = _env("OCR_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OCR_API_KEY is required. Set it in the environment or .env before starting the service."
        )

    allowed_hosts_raw = _env("ALLOWED_HOSTS", "localhost,127.0.0.1") or "localhost,127.0.0.1"
    allowed_hosts = [h.strip() for h in allowed_hosts_raw.split(",") if h.strip()]
    # RunPod / proxies may send unexpected Host headers; "*" disables host filtering.
    if "*" in allowed_hosts:
        allowed_hosts = ["*"]

    device = _env("PADDLEOCR_DEVICE", "gpu:0") or "gpu:0"
    pipeline_version = _env("PADDLEOCR_VL_PIPELINE_VERSION", "v1.6") or "v1.6"
    # In-process by default. Set PADDLEOCR_VL_REC_BACKEND=vllm-server only when a local
    # vLLM container is running. Empty / unset = in-process (no Docker required).
    # Use os.getenv so an explicit empty value is not replaced by a non-empty default.
    raw_backend = os.getenv("PADDLEOCR_VL_REC_BACKEND")
    if raw_backend is None or raw_backend.strip() == "":
        vl_rec_backend = None
        vl_rec_server_url = None
    else:
        vl_rec_backend = raw_backend.strip()
        vl_rec_server_url = _env(
            "PADDLEOCR_VL_REC_SERVER_URL", "http://localhost:8118/v1"
        )
    model_label = f"paddleocr-vl:{pipeline_version}"
    if vl_rec_backend:
        model_label = f"{model_label}+{vl_rec_backend}"

    structure_lang = (_env("PADDLEOCR_STRUCTURE_LANG", "en") or "en").lower()

    return Settings(
        env=_env("ENV", "development") or "development",
        host=_env("HOST", "127.0.0.1") or "127.0.0.1",
        port=_env_int("PORT", 8090),
        device=device,
        pipeline_version=pipeline_version,
        vl_rec_backend=vl_rec_backend,
        vl_rec_server_url=vl_rec_server_url,
        ocr_api_key=api_key,
        rate_limit_ocr=_env("RATE_LIMIT_OCR", "30/minute") or "30/minute",
        rate_limit_ocr_table=_env("RATE_LIMIT_OCR_TABLE", "15/minute") or "15/minute",
        rate_limit_global=_env("RATE_LIMIT_GLOBAL", "60/minute") or "60/minute",
        max_upload_bytes=_env_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024),
        allowed_hosts=allowed_hosts,
        model_label=model_label,
        max_concurrent=_env_int("PADDLEOCR_MAX_CONCURRENT", 1),
        use_doc_unwarping=_env_bool("PADDLEOCR_USE_DOC_UNWARPING", True),
        structure_lang=structure_lang,
        warm_vl=_env_bool("PADDLEOCR_WARM_VL", True),
        warm_ocr=_env_bool("PADDLEOCR_WARM_OCR", True),
        warm_structure=_env_bool("PADDLEOCR_WARM_STRUCTURE", True),
    )


settings = load_settings()
