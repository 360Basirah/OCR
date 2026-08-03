# Basirah PaddleOCR Service

Private FastAPI microservice that runs [PaddleOCR](https://www.paddleocr.ai/) (PP-OCRv6) and PP-StructureV3 for table-aware OCR. Basirah Node calls this over HTTP; OpenAI still does structured JSON extraction.

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | none | Liveness (no model inference) |
| `POST` | `/ocr` | `X-API-Key` | Image → plain text |
| `POST` | `/ocr-table` | `X-API-Key` | Image → table/layout text |

Multipart field name: `file` (JPEG / PNG / WebP / PDF, max 10MB by default).

## Security

- **API key required** for `/ocr` and `/ocr-table` (`X-API-Key` or `Authorization: Bearer`)
- **Rate limits** (env-overridable): OCR `30/minute`, table `15/minute`, global `60/minute`
- **Upload limits** + magic-byte MIME sniff
- **TrustedHost** allowlist; **no CORS** (server-to-server only)
- Security headers: `nosniff`, `DENY` frame, `no-store`
- `/docs` disabled when `ENV=production`
- Bind to `127.0.0.1` locally; in production keep the service on a private network/VPC only

## Local setup (Windows / CPU)

```powershell
cd E:\MainWorkSpace\paddleocr-service
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 1) PaddlePaddle CPU wheel (required before paddleocr)
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# 2) App deps
python -m pip install -r requirements.txt

# 3) Env
copy .env.example .env
# Edit OCR_API_KEY to a long random secret

# 4) Run (binds 127.0.0.1:8090)
uvicorn app.main:app --host 127.0.0.1 --port 8090
```

GPU: install the matching CUDA paddlepaddle wheel from [PaddlePaddle install docs](https://www.paddleocr.ai/main/en/quick_start.html), then set `PADDLEOCR_DEVICE=gpu:0`.

First request may download models; startup warms engines so cold start happens at boot.

### Smoke checks

```powershell
curl http://127.0.0.1:8090/health

# Expect 401 without key
curl -Method POST http://127.0.0.1:8090/ocr -Form "file=@passport.jpg"

# With key
curl -Method POST http://127.0.0.1:8090/ocr `
  -Headers @{ "X-API-Key" = "your-secret" } `
  -Form "file=@passport.jpg"
```

## Basirah wiring

In `Backend/Basirah_backend/.env`:

```env
PADDLEOCR_URL=http://127.0.0.1:8090
PADDLEOCR_API_KEY=your-secret
```

Docker Compose API container:

```env
PADDLEOCR_URL=http://host.docker.internal:8090
PADDLEOCR_API_KEY=your-secret
```

`PADDLEOCR_API_KEY` must equal this service’s `OCR_API_KEY`.

## Environment variables

| Variable | Default | Notes |
|----------|---------|--------|
| `OCR_API_KEY` | *(required)* | Shared secret |
| `HOST` | `127.0.0.1` | Use `0.0.0.0` only behind a private network |
| `PORT` | `8090` | |
| `ENV` | `development` | `production` disables `/docs` |
| `PADDLEOCR_DEVICE` | `cpu` | e.g. `gpu:0` |
| `PADDLEOCR_LANG` | `en` | Override for Arabic-heavy docs |
| `PADDLEOCR_OCR_VERSION` | *(unset)* | e.g. `PP-OCRv6` |
| `RATE_LIMIT_OCR` | `30/minute` | |
| `RATE_LIMIT_OCR_TABLE` | `15/minute` | |
| `RATE_LIMIT_GLOBAL` | `60/minute` | |
| `MAX_UPLOAD_BYTES` | `10485760` | 10MB |
| `ALLOWED_HOSTS` | `localhost,127.0.0.1` | Comma-separated |

## Production notes

- Do not expose this port publicly; put it behind VPC / internal load balancer.
- Prefer reverse-proxy timeouts ≥ 120s for large multi-page jobs (Basirah sends one page image per request).
- Never log API keys or OCR document text (access logs only request id / path / status / duration).
- Pin `paddlepaddle` and `paddleocr` versions in deployments.
