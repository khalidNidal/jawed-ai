import os
import io
import threading
import numpy as np
import soundfile as sf
import torch
import joblib

from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

MODEL_ID = "ahmedAlawneh/wav2vec2-tajweed-juzamma"
HF_TOKEN = os.getenv("HF_TOKEN")  # يتم تمريرها من Cloud Run

# مسار الكلاسيفاير
SVM_PATH = "models/svm_model.joblib"

app = FastAPI()

# ====== ASR Model Globals ======
processor = None
model = None
device = "cpu"
load_error = None
loading = False
_model_lock = threading.Lock()

# ====== Classifier Globals ======
svm_model = None
svm_error = None
svm_loading = False
_svm_lock = threading.Lock()


def _load_asr_model():
    """تحميل موديل wav2vec2 (يمكن استدعاؤه من Thread أو من داخل Request)."""
    global processor, model, device, load_error, loading

    with _model_lock:
        # لو محمّل خلاص
        if model is not None and processor is not None:
            return
        # لو صار فيه خطأ سابق
        if load_error:
            return

        loading = True

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # تحميل باستخدام HF_TOKEN لتجنب 429
            try:
                processor_local = Wav2Vec2Processor.from_pretrained(MODEL_ID, token=HF_TOKEN)
                model_local = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, token=HF_TOKEN).to(device)
            except TypeError:
                processor_local = Wav2Vec2Processor.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
                model_local = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN).to(device)

            model_local.eval()

            processor = processor_local
            model = model_local

        except Exception as e:
            load_error = str(e)

        finally:
            loading = False


def _load_svm():
    """تحميل SVM joblib (يمكن استدعاؤه من Thread أو من داخل Request)."""
    global svm_model, svm_error, svm_loading

    with _svm_lock:
        if svm_model is not None:
            return
        if svm_error:
            return

        svm_loading = True
        try:
            svm_model = joblib.load(SVM_PATH)
        except Exception as e:
            svm_error = str(e)
        finally:
            svm_loading = False


def ensure_asr_loaded():
    """تأكد إن موديل ASR محمّل. لو مش محمّل، حمّله داخل نفس الطلب (يحميك مع cpu-throttling)."""
    if load_error:
        raise HTTPException(status_code=500, detail=f"Model load failed: {load_error}")

    if model is None or processor is None:
        _load_asr_model()

    if load_error:
        raise HTTPException(status_code=500, detail=f"Model load failed: {load_error}")

    if model is None or processor is None:
        # لو لسه ما تحمّل (نادرًا)
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly")


def ensure_svm_loaded():
    """تأكد إن SVM محمّل."""
    if svm_error:
        raise HTTPException(status_code=500, detail=f"SVM load failed: {svm_error}")

    if svm_model is None:
        _load_svm()

    if svm_error:
        raise HTTPException(status_code=500, detail=f"SVM load failed: {svm_error}")

    if svm_model is None:
        raise HTTPException(status_code=503, detail="SVM is still loading, try again shortly")


def read_wav_16k_mono(file_bytes: bytes):
    """قراءة wav والتحقق إنه 16kHz + mono."""
    try:
        wav, sr = sf.read(io.BytesIO(file_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid WAV file: {e}")

    if isinstance(wav, np.ndarray) and wav.ndim > 1:
        wav = wav.mean(axis=1)  # mono

    if sr != 16000:
        raise HTTPException(status_code=400, detail=f"Expected 16kHz WAV, got {sr}Hz")

    return wav, sr


def embedding_layer11_meanstd(wav: np.ndarray):
    """
    استخراج embedding من layer 11 باستخدام mean+std
    الناتج: vector طول 2048 (إذا hidden=1024)
    """
    inputs = processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)

    input_values = inputs.input_values.to(device)
    attention_mask = getattr(inputs, "attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        out = model(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    hs = out.hidden_states
    if hs is None or len(hs) <= 11:
        raise HTTPException(status_code=500, detail="Hidden states not available or layer index out of range.")

    # hs[11]: (B, T, H) -> خذ أول batch
    layer11 = hs[11][0]  # (T, H)

    mean = layer11.mean(dim=0)
    std = layer11.std(dim=0, unbiased=False)
    emb = torch.cat([mean, std], dim=0)  # (2H,)

    emb = emb.detach().cpu().numpy().astype(np.float32)

    if emb.shape[0] != 2048:
        raise HTTPException(status_code=500, detail=f"Embedding size mismatch: {emb.shape[0]} (expected 2048)")

    return emb


@app.on_event("startup")
def startup():
    # تحميل بالخلفية (لتسريع أول استخدام لو CPU Always Allocated أو min-instances=1)
    t1 = threading.Thread(target=_load_asr_model, daemon=True)
    t1.start()

    # تحميل SVM بالخلفية (اختياري)
    t2 = threading.Thread(target=_load_svm, daemon=True)
    t2.start()


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
    }


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    ensure_asr_loaded()

    data = await audio.read()
    wav, _ = read_wav_16k_mono(data)

    inputs = processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(pred_ids)[0]

    return {"text": text}


@app.post("/classify")
async def classify(audio: UploadFile = File(...)):
    ensure_asr_loaded()
    ensure_svm_loaded()

    data = await audio.read()
    wav, _ = read_wav_16k_mono(data)

    emb = embedding_layer11_meanstd(wav)  # (2048,)
    X = emb.reshape(1, -1)

    pred = svm_model.predict(X)[0]

    # confidence
    if hasattr(svm_model, "predict_proba"):
        proba = svm_model.predict_proba(X)[0]
        idx = int(np.argmax(proba))
        conf = float(proba[idx])
    else:
        # SVM بدون probability=True: نستخدم decision_function ثم sigmoid كتقدير
        score = float(svm_model.decision_function(X)[0])
        conf = float(1.0 / (1.0 + np.exp(-score)))

    return {
        "label": str(pred),          # "wrong" or "correct"
        "confidence": conf
    }
