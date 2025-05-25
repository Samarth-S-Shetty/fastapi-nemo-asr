# app/main_clean.py - Clean working version
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import onnxruntime as ort
import numpy as np
import librosa
import os
import tempfile

# Load ONNX model and vocabulary first
MODEL_PATH = "model/asr_model.onnx"
if not os.path.exists(MODEL_PATH):
    raise RuntimeError("ONNX model not found at model/asr_model.onnx")

print("Loading ONNX model...")
session = ort.InferenceSession(MODEL_PATH)

# Load vocab from vocab.txt
VOCAB_PATH = "model/vocab.txt"
if not os.path.exists(VOCAB_PATH):
    raise RuntimeError("vocab.txt not found in model/")

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = [line.strip() for line in f.readlines()]

print(f"✅ NEMO ASR API Ready!")
print(f"📚 Vocabulary: {len(vocab)} tokens")

# Initialize FastAPI app
app = FastAPI(
    title="NEMO Hindi ASR API",
    description="FastAPI application for Hindi speech recognition using NVIDIA NeMo",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {
        "message": "🎤 NEMO Hindi ASR API",
        "status": "running",
        "model": "Hindi Conformer CTC Medium",
        "vocab_size": len(vocab),
        "supported_formats": [".wav"],
        "sample_rate": "16kHz"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True, "vocab_loaded": len(vocab) > 0}

def preprocess_audio(file_path):
    # Load 16kHz audio
    y, sr = librosa.load(file_path, sr=16000)
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Normalize audio
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # Create log-mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=80,
        n_fft=512,
        hop_length=160,
        win_length=400,
        fmin=0,
        fmax=8000
    )

    # Convert to log scale and normalize
    log_mel = librosa.power_to_db(mel_spec, ref=np.max, top_db=80).astype(np.float32)
    log_mel = (log_mel + 80.0) / 80.0
    log_mel = np.clip(log_mel, 0.0, 1.0)

    log_mel = np.expand_dims(log_mel, axis=0)  # [1, 80, T]
    return log_mel, log_mel.shape[2]

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Transcribe Hindi audio to text using NEMO ASR model"""

    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")

    tmp_path = None
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        print(f"Processing: {file.filename}")

        # Preprocess audio
        features, time_length = preprocess_audio(tmp_path)
        print(f"Features shape: {features.shape}")

        # ONNX inference
        inputs = {
            "audio_signal": features,
            "length": np.array([time_length], dtype=np.int64)
        }

        outputs = session.run(None, inputs)
        logits = outputs[0]
        pred_ids = np.argmax(logits, axis=-1)[0]

        # CTC Decoding
        blank_token = len(vocab)
        decoded_ids = []
        prev_id = -1

        for pred_id in pred_ids:
            if pred_id != blank_token and pred_id != prev_id and pred_id < len(vocab):
                decoded_ids.append(pred_id)
            prev_id = pred_id

        # Convert to text
        transcription = ""
        for i in decoded_ids:
            if i < len(vocab):
                transcription += vocab[i]

        # Post-process
        transcription = transcription.replace("▁", " ").strip()
        if len(transcription) < 2:
            transcription += " [unclear speech]"

        # Calculate simple confidence
        confidence = 0.85  # Placeholder for now

        result = {
            "transcription": transcription,
            "confidence": confidence,
            "audio_duration": round(time_length * 0.01, 2),
            "num_tokens": len(decoded_ids),
            "status": "success"
        }

        print(f"Result: {result}")
        return result

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    print("🚀 Starting NEMO Hindi ASR API Server...")
    print("📡 API: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
