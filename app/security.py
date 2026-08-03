from __future__ import annotations

import hashlib
import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from app.config import settings


def extract_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> str | None:
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        return token or None
    return None


def require_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    provided = extract_api_key(x_api_key, authorization)
    expected = settings.ocr_api_key
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide X-API-Key or Authorization: Bearer <key>.",
        )
    return provided


def rate_limit_key(request: Request) -> str:
    """Key rate limits by API key hash when present, else client IP."""
    provided = extract_api_key(
        request.headers.get("x-api-key"),
        request.headers.get("authorization"),
    )
    if provided:
        digest = hashlib.sha256(provided.encode("utf-8")).hexdigest()[:16]
        return f"key:{digest}"
    client = request.client.host if request.client else "unknown"
    return f"ip:{client}"
