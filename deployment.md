# RunPod OCR Deployment

Path: **Serverless → Deploy a new endpoint → Deploy from a Docker image**

Prereqs: image pushed to GHCR · RunPod credits loaded · replace `OCR_API_KEY` before go-live

## CI (GitHub Actions → GHCR)

On every push to **`main`** / **`master`** (or manual **Run workflow**):

- Workflow: `.github/workflows/docker-publish.yml`
- Image: `ghcr.io/360basirah/360basirah-ocr:latest`
- Also tags: `sha-<commit>`, `build-<run_number>`

Uses `GITHUB_TOKEN` (`packages: write`). No extra secrets.

After a green run, restart a RunPod worker so it pulls the new `:latest`.

---

## Screen 1 — Configure image

| Field | Value |
|-------|--------|
| Container image | `ghcr.io/360basirah/360basirah-ocr:latest` |
| Registry auth | Select GHCR credential if package is private; else skip |
| Start command | leave empty |
| Container disk | `40` GB |
| Expose HTTP ports | `8090` |
| Expose TCP ports | leave empty |
| Endpoint type | **Load balancer** (not Queue) |
| Health check endpoint | `/ping` (or `/health`) |

### Environment variables (Raw editor)

```env
ENV=production
HOST=0.0.0.0
PORT=8090
OCR_API_KEY=basirah-ocr-secret-360
PADDLEOCR_DEVICE=gpu:0
PADDLEOCR_VL_PIPELINE_VERSION=v1.6
PADDLEOCR_MAX_CONCURRENT=1
PADDLEOCR_USE_DOC_UNWARPING=true
PADDLEOCR_STRUCTURE_LANG=en
PADDLEOCR_WARM_VL=true
PADDLEOCR_WARM_OCR=false
PADDLEOCR_WARM_STRUCTURE=false
RATE_LIMIT_OCR=30/minute
RATE_LIMIT_OCR_TABLE=15/minute
RATE_LIMIT_GLOBAL=60/minute
MAX_UPLOAD_BYTES=10485760
ALLOWED_HOSTS=*
```

`OCR_API_KEY` must match Basirah `PADDLEOCR_API_KEY`. Do not set `PADDLEOCR_VL_REC_BACKEND`.

Health check path: leave default `/ping` (app supports it) **or** set `/health`.

Click **Next**.

---

## Screen 2 — Configure endpoint

| Field | Value |
|-------|--------|
| Endpoint name | `basirah-ocr` (any name you like) |
| Compute type | **GPU** |
| GPU | **24 GB** ($0.69/hr — High Supply) |
| Max workers | `1` (use `2` later if traffic grows) |
| Active workers | `0` (scale to zero; set `1` for zero cold start) |
| GPU count | `1` |
| Idle timeout | `600` sec |
| Enable FlashBoot | **ON** |
| Enable execution timeout | **ON** |
| Execution timeout | `600` sec |
| Cached model | leave empty |
| Environment variables | skip (already set on Screen 1) |
| Security & compliance | `Any` |
| Data centers | leave default (all selected) |
| Network volumes | No volumes |
| Allowed CUDA versions | `All versions` |
| Auto scaling type | **Request count** (required for Load Balancer) |
| Request count | `1` req |
| Enabled GPU types | Check **all** GPUs listed (better availability) |

**Important:** first select GPU class **24 GB** above. Then under **Enabled GPU types**, tick every GPU shown for that class (e.g. RTX 4090, L4, etc.).

If you see only `RTX 2000 Ada / RTX 4000 Ada / RTX A4000 / RTX A4500`, you are on the **16 GB** class — go back and pick **24 GB** instead (safer for PaddleOCR-VL).

Click **Deploy**.

### After deploy

1. Copy endpoint URL: `https://<ENDPOINT_ID>.api.runpod.ai`
2. Test:

```bash
curl -s "https://<ENDPOINT_ID>.api.runpod.ai/health" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>"

curl -s -X POST "https://<ENDPOINT_ID>.api.runpod.ai/ocr" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "X-API-Key: <OCR_API_KEY>" \
  -F "file=@test.png"
```

3. Basirah backend:

```env
PADDLEOCR_URL=https://<ENDPOINT_ID>.api.runpod.ai
PADDLEOCR_API_KEY=<same OCR_API_KEY>
RUNPOD_API_KEY=<your RunPod API key>
```
