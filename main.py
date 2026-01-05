import io
import os
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from scipy.signal import resample_poly

MODEL_ID = "ahmedAlawneh/wav2vec2-tajweed-juzamma"
TARGET_SR = 16000

app = FastAPI(title="Tajweed ASR (Wav2Vec2)")

# تحميل الموديل مرة واحدة عند تشغيل السيرفر (أفضل أداء)
processor: Optional[Wav2Vec2Processor] = None
model: Optional[Wav2Vec2ForCTC] = None
device: str = "cpu"


def _to_mono(x: np.ndarray) -> np.ndarray:
    # لو الصوت ستيريو (n_samples, n_channels) أو (n_channels, n_samples)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        # soundfile يرجع غالبًا (n_samples, channels)
        return x.mean(axis=1).astype(np.float32)
    raise ValueError("Unsupported audio shape")


def _resample_if_needed(x: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    if sr == target_sr:
        return x
    # resample_poly سريع وجودته ممتازة
    g = np.gcd(sr, target_sr)
    up = target_sr // g
    down = sr // g
    y = resample_poly(x, up=up, down=down).astype(np.float32)
    return y


@app.on_event("startup")
def startup():
    global processor, model, device

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": device}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    يقبل WAV (مفضل) ويرجع النص.
    """
    if audio.content_type not in ("audio/wav", "audio/x-wav", "application/octet-stream"):
        # أحيانًا المتصفح يرسل octet-stream حتى لو wav
        # فبنسمح به، لكن لو كان واضح أنه مش wav نرفض
        name = (audio.filename or "").lower()
        if not name.endswith(".wav"):
            raise HTTPException(status_code=415, detail="Please upload a WAV file (16kHz mono preferred).")

    try:
        raw = await audio.read()
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read audio. Ensure it's a valid WAV. Error: {str(e)}")

    try:
        x = _to_mono(np.asarray(data, dtype=np.float32))
        x = _resample_if_needed(x, sr, TARGET_SR)
        # تطبيع بسيط (اختياري لكنه مفيد)
        if np.max(np.abs(x)) > 0:
            x = x / (np.max(np.abs(x)) + 1e-9)

        inputs = processor(x, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)

        text = processor.batch_decode(pred_ids)[0]

        return JSONResponse(
            {
                "text": text,
                "sample_rate_used": TARGET_SR,
                "filename": audio.filename,
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
