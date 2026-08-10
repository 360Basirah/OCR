# Basirah PaddleOCR Service

Private FastAPI microservice for high-accuracy document OCR. Basirah Node calls this over HTTP; structured JSON extraction stays upstream.

**Default engine:** [PaddleOCR-VL-1.6](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html) (max document accuracy).  
**Optional:** [PP-OCRv6 medium](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html) for classic line OCR; [PP-StructureV3](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PP-StructureV3.html) for tables/layout.

Runs on GPU (RTX 4060 locally, RunPod in production). Inference runs **off the event loop** behind a GPU semaphore so concurrent requests do not stall each other or `/health`.

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | none | Liveness (no model inference) |
| `POST` | `/ocr` | `X-API-Key` | Image/PDF → text / markdown |
| `POST` | `/ocr-table` | `X-API-Key` | Tables / layout → markdown |

Multipart field name: `file` (JPEG / PNG / WebP / PDF, max 10MB by default).

### Pipeline query params

| Endpoint | Query | Engine |
|----------|-------|--------|
| `/ocr` | `pipeline=vl` (default) | PaddleOCR-VL-1.6 |
| `/ocr` | `pipeline=ocr` | PP-OCRv6 medium det+rec |
| `/ocr-table` | `pipeline=structure` (default) | PP-StructureV3 |
| `/ocr-table` | `pipeline=vl` | PaddleOCR-VL-1.6 |

Response fields include `text`, `model`, `pipeline`, `durationMs`, `lineCount`, `queueWaitMs`, `requestId`.

## Concurrency model

- Sync `predict()` runs in a thread pool (`asyncio.to_thread`).
- `PADDLEOCR_MAX_CONCURRENT` (default `1`) gates GPU access.
- Request A returns when A finishes; request B waits for the slot but does **not** delay A’s response.
- `/health` stays responsive while OCR is running.

## Security

- **API key required** for `/ocr` and `/ocr-table` (`X-API-Key` or `Authorization: Bearer`)
- **Rate limits** (env-overridable): OCR `30/minute`, table `15/minute`, global `60/minute`
- **Upload limits** + magic-byte MIME sniff
- **TrustedHost** allowlist; **no CORS** (server-to-server only)
- Security headers: `nosniff`, `DENY` frame, `no-store`
- `/docs` disabled when `ENV=production`
- Bind to `127.0.0.1` locally; in production keep the service on a private network/VPC only

## Local setup (Windows / RTX 4060)

Requires a working NVIDIA driver (CUDA 12.x capable). Pick the Paddle wheel that matches your CUDA toolkit — `cu126` is a common match for 40-series cards.

```powershell
cd E:\MainWorkSpace\paddleocr-service
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 1) PaddlePaddle GPU wheel (required before paddleocr)
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# 2) App deps (paddleocr[doc-parser] >= 3.7 for VL-1.6 + PP-OCRv6 medium)
python -m pip install -U -r requirements.txt

# 3) Env
copy .env.example .env
# Edit OCR_API_KEY to a long random secret
# Confirm PADDLEOCR_DEVICE=gpu:0

# 4) Run (binds 127.0.0.1:8090)
uvicorn app.main:app --host 127.0.0.1 --port 8090
```

Other CUDA indexes: see [PaddlePaddle install docs](https://www.paddlepaddle.org.cn/en/install/quick). CPU (`PADDLEOCR_DEVICE=cpu`) works for debugging only — VL is slow on CPU.

First startup downloads models and warms enabled engines (`PADDLEOCR_WARM_*`).

### Smoke checks

```powershell
curl http://127.0.0.1:8090/health
# Expect model: paddleocr-vl:v1.6, max_concurrent: 1

# Expect 401 without key
curl -Method POST http://127.0.0.1:8090/ocr -Form "file=@passport.jpg"

# VL (default)
curl -Method POST "http://127.0.0.1:8090/ocr?pipeline=vl" `
  -Headers @{ "X-API-Key" = "your-secret" } `
  -Form "file=@passport.jpg"

# PP-OCRv6 medium
curl -Method POST "http://127.0.0.1:8090/ocr?pipeline=ocr" `
  -Headers @{ "X-API-Key" = "your-secret" } `
  -Form "file=@passport.jpg"

# Tables via StructureV3
curl -Method POST "http://127.0.0.1:8090/ocr-table?pipeline=structure" `
  -Headers @{ "X-API-Key" = "your-secret" } `
  -Form "file=@statement.pdf"
```

## Postman: parallel concurrency test

1. Start uvicorn as above.
2. Create two requests: `POST http://127.0.0.1:8090/ocr` with header `X-API-Key` and body form-data `file`.
3. Run them **in parallel** (Collection Runner / two tabs / Newman). Optionally raise `RATE_LIMIT_OCR` if you hit 429.
4. **Pass criteria**
   - First response `durationMs` ≈ a solo OCR run (not ~2×).
   - Second response has higher `queueWaitMs` when `PADDLEOCR_MAX_CONCURRENT=1`.
   - First request’s wall time is **not** stretched to match the second.
   - `GET /health` returns 200 while OCR is in flight.
5. Accuracy spot-check: same image with `pipeline=vl` vs `pipeline=ocr`; use `/ocr-table` for multi-column tables.

Server logs include `request_id`, `pipeline`, `queue_wait_ms`, and `duration_ms`.

## RunPod (production)

1. Use a CUDA GPU pod (8GB+ VRAM for VL; more headroom if warming VL + OCR + Structure together).
2. Build/run the GPU `Dockerfile` (CUDA 12.6 + `paddlepaddle-gpu`).
3. Set runtime env:
   - `OCR_API_KEY` (must match Basirah `PADDLEOCR_API_KEY`)
   - `PADDLEOCR_DEVICE=gpu:0`
   - `PADDLEOCR_VL_PIPELINE_VERSION=v1.6`
   - `PADDLEOCR_MAX_CONCURRENT=1` (raise only after measuring GPU headroom)
   - `ALLOWED_HOSTS` for your private hostname / IP
   - `ENV=production`
4. Keep port `8090` private (VPN / internal network / RunPod private IP). Do not expose publicly.
5. Prefer reverse-proxy / client timeouts **≥ 120–180s**.

### Optional local vLLM (faster VL-1.6, same accuracy)

By default the service runs **in-process** on GPU (no Docker). For faster VL decoding, optionally run a local vLLM server (Docker + NVIDIA GPU; not supported as a native Windows pip install):

```bash
docker run --rm --gpus all --network host \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu \
  paddleocr genai_server --model_name PaddleOCR-VL-1.6-0.9B --host 0.0.0.0 --port 8118 --backend vllm
```

On Docker Desktop for Windows, `--network host` may not work; publish the port instead (`-p 8118:8118`).

Then set in `.env`:

```env
PADDLEOCR_VL_REC_BACKEND=vllm-server
PADDLEOCR_VL_REC_SERVER_URL=http://localhost:8118/v1
```

Leave both empty (default) for in-process Paddle.

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

`PADDLEOCR_API_KEY` must equal this service’s `OCR_API_KEY`. Existing clients keep working: `/ocr` defaults to `pipeline=vl`.

## Environment variables

| Variable | Default | Notes |
|----------|---------|--------|
| `OCR_API_KEY` | *(required)* | Shared secret |
| `HOST` | `127.0.0.1` | Use `0.0.0.0` only behind a private network |
| `PORT` | `8090` | |
| `ENV` | `development` | `production` disables `/docs` |
| `PADDLEOCR_DEVICE` | `gpu:0` | e.g. `cpu` for debug only |
| `PADDLEOCR_VL_PIPELINE_VERSION` | `v1.6` | Passed to `PaddleOCRVL(pipeline_version=...)` |
| `PADDLEOCR_MAX_CONCURRENT` | `1` | GPU job slots |
| `PADDLEOCR_USE_DOC_UNWARPING` | `true` | Skew/warp correction for scans |
| `PADDLEOCR_STRUCTURE_LANG` | `en` | PP-StructureV3 `lang` |
| `PADDLEOCR_WARM_VL` | `true` | Load VL at startup |
| `PADDLEOCR_WARM_OCR` | `true` | Load PP-OCRv6 medium at startup |
| `PADDLEOCR_WARM_STRUCTURE` | `true` | Load StructureV3 at startup |
| `PADDLEOCR_VL_REC_BACKEND` | *(empty)* | Empty = in-process; `vllm-server` when genai container is running |
| `PADDLEOCR_VL_REC_SERVER_URL` | *(empty)* | e.g. `http://localhost:8118/v1` when using vLLM |
| `RATE_LIMIT_OCR` | `30/minute` | Raise for aggressive Postman parallel tests |
| `RATE_LIMIT_OCR_TABLE` | `15/minute` | |
| `RATE_LIMIT_GLOBAL` | `60/minute` | |
| `MAX_UPLOAD_BYTES` | `10485760` | 10MB |
| `ALLOWED_HOSTS` | `localhost,127.0.0.1` | Comma-separated |

## Production notes

- Do not expose this port publicly; put it behind VPC / internal load balancer / RunPod private networking.
- Prefer reverse-proxy timeouts ≥ 120–180s for large multi-page jobs (Basirah sends one page image per request).
- Never log API keys or OCR document text (access logs only request id / path / status / duration).
- Pin `paddlepaddle-gpu` and `paddleocr` versions in deployments.
- Ensure the CUDA wheel matches the RunPod image CUDA major version.
- Warming all three pipelines uses more VRAM; set unused `PADDLEOCR_WARM_*=false` if needed.
