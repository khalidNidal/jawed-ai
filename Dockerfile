FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ننسخ الملفات الأساسية
COPY main.py /app/main.py
COPY models /app/models

# 🟢 ننسخ ملف التجويد داخل الإمج
COPY data /app/data

# 🟢 نحدد مسار ملف التجويد كـ ENV
ENV PORT=8080 \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf \
    HF_HUB_CACHE=/tmp/hf \
    PYTHONUNBUFFERED=1 \
    TAJWEED_PATH=/app/data/tajweed.hafs.uthmani-pause-sajdah.json

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
