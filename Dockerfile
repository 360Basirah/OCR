FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --no-cache-dir \
    paddlepaddle==3.2.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV HOST=0.0.0.0 \
    PORT=8090 \
    ENV=production \
    PADDLEOCR_DEVICE=cpu \
    ALLOWED_HOSTS=localhost,127.0.0.1

EXPOSE 8090

# OCR_API_KEY must be provided at runtime
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]
