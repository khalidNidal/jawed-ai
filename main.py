import os
import io
import soundfile as sf
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

MODEL_ID = "ahmedAlawneh/wav2vec2-tajweed-juzamma"

app = FastAPI()

processor = None
model = None
device = "cpu"

@app.on_event("startup")
def load_model():
    global processor, model, device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device)
    model.eval()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_id": MODEL_ID,
        "device": device
    }

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    data = await audio.read()

    try:
        wav, sr = sf.read(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid WAV file: {e}")

    # لو stereo حوله mono
    if isinstance(wav, np.ndarray) and wav.ndim > 1:
        wav = wav.mean(axis=1)

    # لازم 16kHz ل wav2vec2
    if sr != 16000:
        raise HTTPException(status_code=400, detail=f"Expected 16kHz WAV, got {sr}Hz")

    inputs = processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(pred_ids)[0]

    return {"text": text}
