import os
import io
import threading
import numpy as np
import soundfile as sf
import torch
import joblib

from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

MODEL_ID = os.getenv("MODEL_ID", "ahmedAlawneh/wav2vec2-tajweed-juzamma")
HF_TOKEN = os.getenv("HF_TOKEN")

# SVM path داخل الصورة (Docker image)
SVM_PATH = os.getenv("SVM_PATH", "models/svm_model.joblib")

# Layer index للـ embedding
EMB_LAYER = int(os.getenv("EMB_LAYER", "11"))  # layer 11
# mapping افتراضي (عدّلها من env لو احتجت)
SVM_WRONG_VALUE = os.getenv("SVM_WRONG_VALUE", "1")
SVM_CORRECT_VALUE = os.getenv("SVM_CORRECT_VALUE", "0")

app = FastAPI()

processor = None
model = None
svm_model = None

device = "cpu"
load_error = None
loading = False

svm_error = None
svm_loading = False


def _read_wav_bytes(data: bytes):
    try:
        wav, sr = sf.read(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

    # stereo -> mono
    if isinstance(wav, np.ndarray) and wav.ndim > 1:
        wav = wav.mean(axis=1)

    if sr != 16000:
        raise HTTPException(status_code=400, detail=f"Expected 16kHz WAV, got {sr}Hz")

    # float32
    wav = wav.astype(np.float32, copy=False)
    return wav, sr


def _map_label(raw):
    # لو كان الكلاسيفاير أصلاً يرجّع wrong/correct
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("wrong", "incorrect", "bad", "false"):
            return "wrong"
        if s in ("correct", "right", "good", "true"):
            return "correct"
        # numeric as string
        if s == str(SVM_WRONG_VALUE).strip().lower():
            return "wrong"
        if s == str(SVM_CORRECT_VALUE).strip().lower():
            return "correct"
        return raw  # fallback

    # numeric
    s = str(raw).strip().lower()
    if s == str(SVM_WRONG_VALUE).strip().lower():
        return "wrong"
    if s == str(SVM_CORRECT_VALUE).strip().lower():
        return "correct"
    return str(raw)


def _softmax(x: np.ndarray):
    x = x.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def _svm_predict_with_confidence(vec_2d: np.ndarray):
    """
    vec_2d shape: (1, D)
    Returns: raw_label, confidence(float)
    """
    if svm_model is None:
        raise HTTPException(status_code=503, detail="SVM is still loading, try again shortly")

    # Predict label
    raw = svm_model.predict(vec_2d)[0]

    # Confidence
    conf = None
    if hasattr(svm_model, "predict_proba"):
        probs = svm_model.predict_proba(vec_2d)[0]  # shape (C,)
        idx = int(np.argmax(probs))
        raw = svm_model.classes_[idx]
        conf = float(probs[idx])
    elif hasattr(svm_model, "decision_function"):
        df = svm_model.decision_function(vec_2d)
        df = np.asarray(df)
        # binary: shape (1,) or (1,1)
        if df.ndim == 1 or (df.ndim == 2 and df.shape[1] == 1):
            score = float(df.reshape(-1)[0])
            # sigmoid تقريب
            conf = float(1.0 / (1.0 + np.exp(-score)))
        else:
            # multi-class: softmax على الـ scores
            probs = _softmax(df.reshape(-1))
            idx = int(np.argmax(probs))
            raw = svm_model.classes_[idx]
            conf = float(probs[idx])
    else:
        # ما في طريقة ثقة واضحة
        conf = 0.0

    return raw, conf


def _extract_embedding_from_hidden_states(hidden_states: tuple, layer_index: int):
    """
    hidden_states: tuple of tensors [B, T, H]
    mean+std => (B, 2H)
    """
    if not hidden_states:
        raise RuntimeError("No hidden_states returned from model")

    # clamp layer index
    li = max(0, min(layer_index, len(hidden_states) - 1))
    hs = hidden_states[li]  # [B, T, H]

    mean = hs.mean(dim=1)  # [B, H]
    std = hs.std(dim=1, unbiased=False)  # [B, H]
    emb = torch.cat([mean, std], dim=-1)  # [B, 2H]
    return emb


def _load_all_bg():
    global processor, model, device, load_error, loading
    global svm_model, svm_error, svm_loading

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load HF model
        try:
            processor = Wav2Vec2Processor.from_pretrained(MODEL_ID, token=HF_TOKEN)
            model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, token=HF_TOKEN).to(device)
        except TypeError:
            processor = Wav2Vec2Processor.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
            model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN).to(device)

        model.eval()

    except Exception as e:
        load_error = str(e)

    # Load SVM
    try:
        svm_loading = True
        svm_model = joblib.load(SVM_PATH)
        svm_error = None
    except Exception as e:
        svm_error = str(e)
        svm_model = None
    finally:
        svm_loading = False
        loading = False


@app.on_event("startup")
def startup():
    global loading, svm_loading
    loading = True
    svm_loading = True
    t = threading.Thread(target=_load_all_bg, daemon=True)
    t.start()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "loading": loading,
        "model_loaded": model is not None,
        "model_id": MODEL_ID,
        "device": device,
        "error": load_error,
        "has_hf_token": bool(HF_TOKEN),

        "svm_loaded": svm_model is not None,
        "svm_loading": svm_loading,
        "svm_error": svm_error,

        "emb_layer": EMB_LAYER,
        "svm_wrong_value": SVM_WRONG_VALUE,
        "svm_correct_value": SVM_CORRECT_VALUE,
    }


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if load_error:
        raise HTTPException(status_code=500, detail=f"Model load failed: {load_error}")
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly")

    data = await audio.read()
    wav, _ = _read_wav_bytes(data)

    inputs = processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        out = model(inputs.input_values.to(device))
        logits = out.logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(pred_ids)[0]

    return {"text": text}


@app.post("/classify")
async def classify(audio: UploadFile = File(...)):
    if load_error:
        raise HTTPException(status_code=500, detail=f"Model load failed: {load_error}")
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly")
    if svm_error:
        raise HTTPException(status_code=500, detail=f"SVM load failed: {svm_error}")
    if svm_model is None:
        raise HTTPException(status_code=503, detail="SVM is still loading, try again shortly")

    data = await audio.read()
    wav, _ = _read_wav_bytes(data)

    inputs = processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        out = model(
            inputs.input_values.to(device),
            output_hidden_states=True,
            return_dict=True
        )
        emb_t = _extract_embedding_from_hidden_states(out.hidden_states, EMB_LAYER)  # [1, 2H]
        emb = emb_t.detach().cpu().numpy().astype(np.float32)

    raw_label, conf = _svm_predict_with_confidence(emb)
    mapped = _map_label(raw_label)

    return {
        "label": mapped,
        "raw_label": str(raw_label),
        "confidence": float(conf),
    }


@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    """
    Endpoint واحد يجمع:
    - transcription
    - wrong/correct classification
    - confidence
    ويشغّل الموديل مرة واحدة (logits + hidden_states).
    """
    if load_error:
        raise HTTPException(status_code=500, detail=f"Model load failed: {load_error}")
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly")
    if svm_error:
        raise HTTPException(status_code=500, detail=f"SVM load failed: {svm_error}")
    if svm_model is None:
        raise HTTPException(status_code=503, detail="SVM is still loading, try again shortly")

    data = await audio.read()
    wav, _ = _read_wav_bytes(data)

    inputs = processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        out = model(
            inputs.input_values.to(device),
            output_hidden_states=True,
            return_dict=True
        )

        # (1) transcription
        logits = out.logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(pred_ids)[0]

        # (2) embedding + svm
        emb_t = _extract_embedding_from_hidden_states(out.hidden_states, EMB_LAYER)
        emb = emb_t.detach().cpu().numpy().astype(np.float32)

    raw_label, conf = _svm_predict_with_confidence(emb)
    mapped = _map_label(raw_label)

    return {
        "text": text,
        "label": mapped,          # wrong/correct
        "raw_label": str(raw_label),
        "confidence": float(conf),
    }
