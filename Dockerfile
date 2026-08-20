# CUDA 12.6 runtime base for PaddleOCR-VL-1.6 on RunPod / local NVIDIA GPUs.
# Match the CUDA tag to your host driver (cu126 is a common RunPod / 40-series choice).
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        paddlepaddle-gpu==3.2.1 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
    && python -m pip install -r requirements.txt

COPY app ./app

ENV HOST=0.0.0.0 \
    PORT=8090 \
    ENV=production \
    PADDLEOCR_DEVICE=gpu:0 \
    PADDLEOCR_VL_PIPELINE_VERSION=v1.6 \
    PADDLEOCR_WARM_VL=true \
    PADDLEOCR_WARM_OCR=false \
    PADDLEOCR_WARM_STRUCTURE=false \
    ALLOWED_HOSTS=*

EXPOSE 8090

# OCR_API_KEY must be provided at runtime (RunPod endpoint env vars).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8090}"]
