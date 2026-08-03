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


@dataclass(frozen=True)
class Settings:
    env: str
    host: str
    port: int
    device: str
    pipeline_version: str
    ocr_api_key: str
    rate_limit_ocr: str
    rate_limit_ocr_table: str
    rate_limit_global: str
    max_upload_bytes: int
    allowed_hosts: list[str]
    model_label: str

    @property
    def is_production(self) -> bool:
        return self.env.lower() == "production"

    @property
    def docs_enabled(self) -> bool:
        return not self.is_production


def load_settings() -> Settings:
    api_key = _env("OCR_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OCR_API_KEY is required. Set it in the environment or .env before starting the service."
        )

    allowed_hosts_raw = _env("ALLOWED_HOSTS", "localhost,127.0.0.1") or "localhost,127.0.0.1"
    allowed_hosts = [h.strip() for h in allowed_hosts_raw.split(",") if h.strip()]

    device = _env("PADDLEOCR_DEVICE", "gpu:0") or "gpu:0"
    pipeline_version = _env("PADDLEOCR_VL_PIPELINE_VERSION", "v1.6") or "v1.6"
    model_label = f"paddleocr-vl:{pipeline_version}"

    return Settings(
        env=_env("ENV", "development") or "development",
        host=_env("HOST", "127.0.0.1") or "127.0.0.1",
        port=_env_int("PORT", 8090),
        device=device,
        pipeline_version=pipeline_version,
        ocr_api_key=api_key,
        rate_limit_ocr=_env("RATE_LIMIT_OCR", "30/minute") or "30/minute",
        rate_limit_ocr_table=_env("RATE_LIMIT_OCR_TABLE", "15/minute") or "15/minute",
        rate_limit_global=_env("RATE_LIMIT_GLOBAL", "60/minute") or "60/minute",
        max_upload_bytes=_env_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024),
        allowed_hosts=allowed_hosts,
        model_label=model_label,
    )


settings = load_settings()
