# Basirah PaddleOCR Service

Private FastAPI microservice that runs [PaddleOCR-VL-1.6](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html) for document OCR (text + tables/layout). Basirah Node calls this over HTTP; OpenAI still does structured JSON extraction.

Runs **in-process on GPU** (RTX 4060 locally, RunPod in production). HTTP contract is unchanged from the previous PP-OCRv6 service.

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | none | Liveness (no model inference) |
| `POST` | `/ocr` | `X-API-Key` | Image → text / markdown |
| `POST` | `/ocr-table` | `X-API-Key` | Image → layout/table markdown (same VL pipeline) |

Multipart field name: `file` (JPEG / PNG / WebP / PDF, max 10MB by default).

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

# 2) App deps (needs paddleocr[doc-parser] >= 3.6.0 for VL-1.6)
python -m pip install -U -r requirements.txt

# 3) Env
copy .env.example .env
# Edit OCR_API_KEY to a long random secret
# Confirm PADDLEOCR_DEVICE=gpu:0

# 4) Run (binds 127.0.0.1:8090)
uvicorn app.main:app --host 127.0.0.1 --port 8090
```

Other CUDA indexes: see [PaddlePaddle install docs](https://www.paddlepaddle.org.cn/en/install/quick). CPU (`PADDLEOCR_DEVICE=cpu`) works for debugging only — VL is much slower than PP-OCRv6 on CPU.

First startup downloads VL + layout models and warms the engine at boot.

### Smoke checks

```powershell
curl http://127.0.0.1:8090/health
# Expect model: paddleocr-vl:v1.6, device: gpu:0

# Expect 401 without key
curl -Method POST http://127.0.0.1:8090/ocr -Form "file=@passport.jpg"

# With key
curl -Method POST http://127.0.0.1:8090/ocr `
  -Headers @{ "X-API-Key" = "your-secret" } `
  -Form "file=@passport.jpg"
```

## RunPod (production)

1. Use a CUDA GPU pod (8GB+ VRAM is enough for the ~0.9B VL model; match or exceed the 4060 for comfortable headroom).
2. Build/run the GPU `Dockerfile` (CUDA 12.6 + `paddlepaddle-gpu`).
3. Set runtime env:
   - `OCR_API_KEY` (must match Basirah `PADDLEOCR_API_KEY`)
   - `PADDLEOCR_DEVICE=gpu:0`
   - `PADDLEOCR_VL_PIPELINE_VERSION=v1.6`
   - `ALLOWED_HOSTS` for your private hostname / IP
   - `ENV=production`
4. Keep port `8090` private (VPN / internal network / RunPod private IP). Do not expose publicly.
5. Prefer reverse-proxy / client timeouts **≥ 120–180s** — VL is heavier than PP-OCRv6 (Basirah sends one page image per request).

### Optional local vLLM (faster VL-1.6)

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

`PADDLEOCR_API_KEY` must equal this service’s `OCR_API_KEY`.

## Environment variables

| Variable | Default | Notes |
|----------|---------|--------|
| `OCR_API_KEY` | *(required)* | Shared secret |
| `HOST` | `127.0.0.1` | Use `0.0.0.0` only behind a private network |
| `PORT` | `8090` | |
| `ENV` | `development` | `production` disables `/docs` |
| `PADDLEOCR_DEVICE` | `gpu:0` | e.g. `cpu` for debug only |
| `PADDLEOCR_VL_PIPELINE_VERSION` | `v1.6` | Passed to `PaddleOCRVL(pipeline_version=...)` |
| `PADDLEOCR_VL_REC_BACKEND` | *(empty)* | Empty = in-process; `vllm-server` when genai container is running |
| `PADDLEOCR_VL_REC_SERVER_URL` | *(empty)* | e.g. `http://localhost:8118/v1` when using vLLM |
| `RATE_LIMIT_OCR` | `30/minute` | |
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
