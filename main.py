import os
import io
import threading
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import joblib
import json

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


# =========================
# Config
# =========================
MODEL_ID = os.getenv("MODEL_ID", "ahmedAlawneh/wav2vec2-tajweed-juzamma")
HF_TOKEN = os.getenv("HF_TOKEN")

SVM_PATH = os.getenv("SVM_PATH", "models/svm_model.joblib")

# Embedding layer (hidden_states index)
EMB_LAYER = int(os.getenv("EMB_LAYER", "11"))  # layer 11

# ✅ حسب طلبك (معكوسة):
# 0 -> wrong
# 1 -> correct
SVM_WRONG_VALUE = os.getenv("SVM_WRONG_VALUE", "0")
SVM_CORRECT_VALUE = os.getenv("SVM_CORRECT_VALUE", "1")

# Minimum audio duration after decoding (seconds)
MIN_AUDIO_SEC = float(os.getenv("MIN_AUDIO_SEC", "0.5"))

# Tajweed annotations JSON (optional)
TAJWEED_PATH = os.getenv("TAJWEED_PATH", "")
IQLAB_MARGIN_BEFORE = float(os.getenv("IQLAB_MARGIN_BEFORE", "1.0"))
IQLAB_MARGIN_AFTER = float(os.getenv("IQLAB_MARGIN_AFTER", "1.0"))


# =========================
# App + Globals
# =========================
app = FastAPI()

processor = None
model = None
svm_model = None

device = "cpu"
load_error = None
loading = False

svm_error = None
svm_loading = False

# Tajweed data globals
tajweed_data = None
# (surah, ayah) -> list[annotations]
tajweed_index = {}


# =========================
# Audio helpers
# =========================
def _decode_any_audio_to_wav16k_mono(data: bytes, filename: str | None = None) -> np.ndarray:
    """
    يقبل أي صيغة (mp3/m4a/wav/...) ويحوّل داخليًا لـ WAV mono 16kHz باستخدام ffmpeg
    ويرجع waveform float32 جاهز للـ processor.

    IMPORTANT:
    بعض ملفات m4a/mp4 تحتاج input seekable (ملف على الديسك)،
    لذلك نكتب data لملف مؤقت في /tmp بدل pipe:0.
    """
    # 1) محاولة مباشرة (لو الملف WAV وsoundfile قادر يقرأه)
    try:
        wav, sr = sf.read(io.BytesIO(data))
        if isinstance(wav, np.ndarray) and wav.ndim > 1:
            wav = wav.mean(axis=1)  # stereo -> mono
        if sr == 16000:
            return wav.astype(np.float32, copy=False)
        # إذا WAV لكن sample rate مختلف -> بنحوّل عبر ffmpeg
    except Exception:
        pass

    suffix = ""
    if filename:
        try:
            suf = Path(filename).suffix
            if 1 <= len(suf) <= 10:
                suffix = suf
        except Exception:
            suffix = ""

    in_path = None
    out_path = None

    try:
        # 2) اكتب الملف على /tmp (seekable)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".bin") as f:
            f.write(data)
            in_path = f.name

        # output wav path
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

        # 3) تحويل عبر ffmpeg (يدعم mp3/m4a وغيره)
        # -vn لتجاهل الفيديو
        # -sn/-dn لتجاهل subs/data streams
        cmd = [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-i", in_path,
            "-vn", "-sn", "-dn",
            "-ac", "1",
            "-ar", "16000",
            out_path,
        ]

        try:
            p = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail="ffmpeg not found in container. Please install ffmpeg in Dockerfile and redeploy.",
            )

        if p.returncode != 0:
            err = p.stderr.decode("utf-8", errors="ignore")
            raise HTTPException(status_code=400, detail=f"ffmpeg decode failed: {err[:400]}")

        wav, sr = sf.read(out_path)
        if isinstance(wav, np.ndarray) and wav.ndim > 1:
            wav = wav.mean(axis=1)

        if sr != 16000:
            raise HTTPException(status_code=500, detail=f"Internal decode produced {sr}Hz (expected 16000Hz)")

        return wav.astype(np.float32, copy=False)

    finally:
        # cleanup temp files
        try:
            if in_path and os.path.exists(in_path):
                os.remove(in_path)
        except Exception:
            pass
        try:
            if out_path and os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass


def _ensure_audio_ok(wav: np.ndarray) -> np.ndarray:
    """
    يمنع Crash (Kernel size > input)
    إذا الصوت طلع فاضي/قصير بعد التحويل.
    """
    if wav is None or not isinstance(wav, np.ndarray) or wav.size == 0:
        raise HTTPException(status_code=400, detail="Audio is empty after decoding. Please record again.")

    # تأكد 1D
    if wav.ndim != 1:
        wav = wav.reshape(-1)

    min_samples = int(16000 * MIN_AUDIO_SEC)
    if wav.size < min_samples:
        dur = wav.size / 16000.0
        raise HTTPException(
            status_code=400,
            detail=f"Audio too short ({dur:.3f}s). Need at least {MIN_AUDIO_SEC:.1f}s.",
        )

    # normalize to [-1, 1] if needed
    max_abs = np.max(np.abs(wav))
    if max_abs > 0:
        wav = wav.astype(np.float32, copy=False) / float(max_abs + 1e-9)

    return wav


# =========================
# SVM helpers
# =========================
def _map_label(raw):
    """
    يحوّل خرج SVM إلى wrong/correct حسب env vars.
    """
    s = str(raw).strip().lower()
    if s == str(SVM_WRONG_VALUE).strip().lower():
        return "wrong"
    if s == str(SVM_CORRECT_VALUE).strip().lower():
        return "correct"

    # fallback لو كان أصلاً نص
    if s in ("wrong", "incorrect", "bad", "false"):
        return "wrong"
    if s in ("correct", "right", "good", "true"):
        return "correct"

    return str(raw)


def _softmax(x: np.ndarray):
    x = x.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def _extract_embedding_mean_std(hidden_states, layer_index: int) -> torch.Tensor:
    """
    يأخذ hidden_states (list of Tensors) ويستخرج embedding من layer معيّن:
    - نأخذ mean و std على طول الزمن
    - ندمجهم -> [batch, 2 * hidden_size]
    """
    if not hidden_states:
        raise HTTPException(status_code=500, detail="Model did not return hidden_states.")

    if layer_index < 0 or layer_index >= len(hidden_states):
        raise HTTPException(status_code=500, detail=f"Invalid EMB_LAYER index {layer_index}")

    hs = hidden_states[layer_index]  # [batch, time, dim]
    if hs.ndim != 3:
        raise HTTPException(status_code=500, detail="Unexpected hidden_states shape.")

    mean = hs.mean(dim=1)
    std = hs.std(dim=1)
    emb = torch.cat([mean, std], dim=-1)
    return emb  # [batch, 2 * dim]


def _svm_predict_with_confidence(emb: np.ndarray):
    """
    emb: [1, dim]
    يرجع (raw_label, confidence)
    """
    if svm_model is None:
        raise HTTPException(status_code=503, detail="SVM is not loaded yet.")

    if emb.ndim == 1:
        vec_2d = emb.reshape(1, -1)
    else:
        vec_2d = emb

    raw = svm_model.predict(vec_2d)[0]

    conf = None
    if hasattr(svm_model, "predict_proba"):
        probs = svm_model.predict_proba(vec_2d)[0]
        idx = int(np.argmax(probs))
        raw = svm_model.classes_[idx]
        conf = float(probs[idx])
    elif hasattr(svm_model, "decision_function"):
        df = svm_model.decision_function(vec_2d)
        df = np.asarray(df)

        # binary
        if df.ndim == 1 or (df.ndim == 2 and df.shape[1] == 1):
            score = float(df.reshape(-1)[0])
            conf = float(1.0 / (1.0 + np.exp(-score)))  # sigmoid approx
        else:
            idx = int(np.argmax(df))
            raw = svm_model.classes_[idx]
            conf = float(_softmax(df.reshape(-1))[idx])

    return raw, conf


# =========================
# Background loading
# =========================
def _load_all_bg():
    global processor, model, device, load_error, loading
    global svm_model, svm_error, svm_loading

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load HF model
    try:
        processor = Wav2Vec2Processor.from_pretrained(MODEL_ID, token=HF_TOKEN)
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, token=HF_TOKEN).to(device)
        load_error = None
    except Exception as e:
        load_error = str(e)
        processor = None
        model = None
        loading = False
        svm_loading = False
        return

    # Load SVM model
    try:
        svm_model = joblib.load(SVM_PATH)
        svm_error = None
    except Exception as e:
        svm_error = str(e)
        svm_model = None
    finally:
        svm_loading = False
        loading = False


def _load_tajweed():
    """Load tajweed annotations JSON into memory (if TAJWEED_PATH is set)."""
    global tajweed_data, tajweed_index
    tajweed_data = None
    tajweed_index = {}
    if not TAJWEED_PATH:
        # optional – if not provided, features depending on it will just return empty spans
        print("TAJWEED_PATH not set; tajweed-based features will be disabled.")
        return
    try:
        with open(TAJWEED_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        tajweed_data = data
        idx = {}
        for entry in data:
            try:
                surah = entry.get("surah")
                ayah = entry.get("ayah")
            except AttributeError:
                continue
            if surah is None or ayah is None:
                continue
            ann = entry.get("annotations") or []
            idx[(int(surah), int(ayah))] = ann
        tajweed_index = idx
        print(f"Loaded tajweed data from {TAJWEED_PATH}, entries={len(tajweed_index)}")
    except FileNotFoundError:
        print(f"TAJWEED_PATH file not found: {TAJWEED_PATH}")
    except Exception as e:
        print(f"Error loading tajweed data from {TAJWEED_PATH}: {e}")


def get_iqlab_spans(surah: int, ayah: int):
    """Return list of iqlab spans for (surah, ayah) from tajweed_index.

    Each span is the original dict from the JSON (includes rule, start, end, ...).
    If tajweed data is not loaded or no entries exist, returns [].
    """
    if not tajweed_index:
        return []
    try:
        key = (int(surah), int(ayah))
    except Exception:
        return []
    annots = tajweed_index.get(key, []) or []
    return [a for a in annots if str(a.get("rule")).lower() == "iqlab"]


@app.on_event("startup")
def startup():
    global loading, svm_loading
    loading = True
    svm_loading = True

    # Load tajweed annotations (if configured) before starting the model loader thread
    _load_tajweed()

    t = threading.Thread(target=_load_all_bg, daemon=True)
    t.start()


# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "loading": loading,
        "model_loaded": model is not None,
        "model_id": MODEL_ID,
        "device": device,
        "error": load_error,
        "svm_loaded": svm_model is not None,
        "svm_loading": svm_loading,
        "svm_error": svm_error,
        "emb_layer": EMB_LAYER,
        "svm_wrong_value": SVM_WRONG_VALUE,
        "svm_correct_value": SVM_CORRECT_VALUE,
        "audio_auto_convert": True,
        "min_audio_sec": MIN_AUDIO_SEC,
    }


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if load_error:
        raise HTTPException(status_code=500, detail=f"Model load failed: {load_error}")
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly")

    data = await audio.read()
    wav = _decode_any_audio_to_wav16k_mono(data, audio.filename)
    wav = _ensure_audio_ok(wav)

    inputs = processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
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
    wav = _decode_any_audio_to_wav16k_mono(data, audio.filename)
    wav = _ensure_audio_ok(wav)

    inputs = processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        out = model(
            inputs.input_values.to(device),
            output_hidden_states=True,
            return_dict=True,
        )
        emb_t = _extract_embedding_mean_std(out.hidden_states, EMB_LAYER)  # [1, 2048]
        emb = emb_t.detach().cpu().numpy().astype(np.float32)

    raw_label, conf = _svm_predict_with_confidence(emb)
    label = _map_label(raw_label)

    return {
        "label": label,
        "raw_label": str(raw_label),
        "confidence": float(conf),
    }


@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    """
    Endpoint واحد:
    - transcribe + classify
    - نفس forward pass (logits + hidden_states)
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
    wav = _decode_any_audio_to_wav16k_mono(data, audio.filename)
    wav = _ensure_audio_ok(wav)

    inputs = processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        out = model(
            inputs.input_values.to(device),
            output_hidden_states=True,
            return_dict=True,
        )

        # (1) transcription
        logits = out.logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(pred_ids)[0]

        # (2) embedding + svm
        emb_t = _extract_embedding_mean_std(out.hidden_states, EMB_LAYER)
        emb = emb_t.detach().cpu().numpy().astype(np.float32)

    raw_label, conf = _svm_predict_with_confidence(emb)
    label = _map_label(raw_label)

    return {
        "text": text,
        "label": label,
        "raw_label": str(raw_label),
        "confidence": float(conf),
    }


@app.post("/analyze_ayah")
async def analyze_ayah(
    audio: UploadFile = File(...),
    surah: int = Form(...),
    ayah: int = Form(...),
):
    """Analyze a full ayah recitation.

    This endpoint:
    - looks up iqlab spans for (surah, ayah) from tajweed annotations
    - decodes the full audio
    - approximates a time window around each span
    - crops each window and runs the existing SVM classifier on it

    NOTE:
    The current time mapping is a simple linear approximation over the ayah text
    based on the span indices. For production, you should replace this with a
    proper ASR + alignment pipeline to get precise timestamps.
    """
    if load_error:
        raise HTTPException(status_code=500, detail=f"Model load failed: {load_error}")
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly")
    if svm_error:
        raise HTTPException(status_code=500, detail=f"SVM load failed: {svm_error}")
    if svm_model is None:
        raise HTTPException(status_code=503, detail="SVM is still loading, try again shortly")

    # 1) decode & sanity-check audio
    data = await audio.read()
    wav = _decode_any_audio_to_wav16k_mono(data, audio.filename)
    wav = _ensure_audio_ok(wav)
    sample_rate = 16000
    duration = float(wav.size) / float(sample_rate)

    # 2) get iqlab spans for this ayah
    spans = get_iqlab_spans(surah, ayah)
    if not spans:
        return {
            "surah": surah,
            "ayah": ayah,
            "duration": duration,
            "iqlab_spans": [],
            "segments": [],
            "margin_before": IQLAB_MARGIN_BEFORE,
            "margin_after": IQLAB_MARGIN_AFTER,
            "note": "No iqlab spans found for this ayah or tajweed data not loaded.",
        }

    # determine a rough max index to normalise time mapping
    max_end = 0
    for s in spans:
        try:
            e = int(s.get("end", 0))
        except Exception:
            e = 0
        if e > max_end:
            max_end = e
    if max_end <= 0:
        max_end = 1  # avoid division by zero

    segments = []
    for span in spans:
        try:
            start_idx = int(span.get("start", 0))
        except Exception:
            start_idx = 0
        try:
            end_idx = int(span.get("end", start_idx))
        except Exception:
            end_idx = start_idx

        center_idx = 0.5 * (start_idx + end_idx)
        # simple linear mapping: character position -> time within the ayah
        core_time = (center_idx / float(max_end)) * duration

        # expand with margins to be safe
        t_start = core_time - IQLAB_MARGIN_BEFORE
        t_end = core_time + IQLAB_MARGIN_AFTER

        # clamp to audio bounds
        if t_start < 0.0:
            t_start = 0.0
        if t_end > duration:
            t_end = duration
        if t_end <= t_start:
            # fallback to a tiny window around the core_time
            t_start = max(0.0, core_time - 0.1)
            t_end = min(duration, core_time + 0.1)

        start_sample = int(t_start * sample_rate)
        end_sample = int(t_end * sample_rate)
        seg_wav = wav[start_sample:end_sample]
        # ensure the cropped segment is still valid
        seg_wav = _ensure_audio_ok(seg_wav)

        # 3) classify this segment with the existing SVM pipeline
        inputs = processor(seg_wav, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model(
                inputs.input_values.to(device),
                output_hidden_states=True,
                return_dict=True,
            )
        emb_t = _extract_embedding_mean_std(out.hidden_states, EMB_LAYER)
        emb = emb_t.detach().cpu().numpy().astype(np.float32)
        raw_label, conf = _svm_predict_with_confidence(emb)
        label = _map_label(raw_label)

        segments.append(
            {
                "span": {
                    "rule": span.get("rule"),
                    "start": start_idx,
                    "end": end_idx,
                },
                "t_start": t_start,
                "t_end": t_end,
                "label": label,
                "raw_label": str(raw_label),
                "confidence": float(conf),
            }
        )

    return {
        "surah": surah,
        "ayah": ayah,
        "duration": duration,
        "iqlab_spans": spans,
        "segments": segments,
        "margin_before": IQLAB_MARGIN_BEFORE,
        "margin_after": IQLAB_MARGIN_AFTER,
        "note": "Segments were derived using a simple linear mapping from span indices to time. Replace with real ASR alignment for higher precision.",
    }
