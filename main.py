import os
import io
import threading
import time
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

MODEL_ID = "ahmedAlawneh/wav2vec2-tajweed-juzamma"

app = FastAPI()

processor = None
model = None
device = "cpu"
load_error = None
loading = False

def _load_model_bg():
    global processor, model, device, load_error, loading
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device)
        model.eval()
    except Exception as e:
        load_error = str(e)
    finally:
        loading = False

@app.on_event("startup")
def startup():
    global loading
    # شغّل التحميل بالخلفية عشان ما يمنع فتح البورت
    loading = True
    t = threading.Thread(target=_load_model_bg, daemon=True)
    t.start()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "loading": loading,
        "model_loaded": model is not None,
        "model_id": MODEL_ID,
        "device": device,
        "error": load_error
    }

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if load_error:
        raise HTTPException(status_code=500, detail=f"Model load failed: {load_error}")
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly")

    data = await audio.read()

    try:
        wav, sr = sf.read(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid WAV file: {e}")

    if isinstance(wav, np.ndarray) and wav.ndim > 1:
        wav = wav.mean(axis=1)

    if sr != 16000:
        raise HTTPException(status_code=400, detail=f"Expected 16kHz WAV, got {sr}Hz")

    inputs = processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(pred_ids)[0]

    return {"text": text}
