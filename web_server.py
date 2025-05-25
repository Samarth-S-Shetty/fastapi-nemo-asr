#!/usr/bin/env python3
"""
Web Server for NEMO ASR with HTML Upload Interface
Serves the HTML file and handles uploads
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import onnxruntime as ort
import numpy as np
import librosa
import os
import tempfile
from pathlib import Path

# Load ONNX model and vocabulary
MODEL_PATH = "model/asr_model.onnx"
VOCAB_PATH = "model/vocab.txt"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("ONNX model not found at model/asr_model.onnx")

if not os.path.exists(VOCAB_PATH):
    raise RuntimeError("vocab.txt not found in model/")

print("🔄 Loading ONNX model...")
session = ort.InferenceSession(MODEL_PATH)

print("📚 Loading vocabulary...")
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = [line.strip() for line in f.readlines()]

print(f"✅ Model loaded! Vocabulary: {len(vocab)} tokens")

# Initialize FastAPI app
app = FastAPI(
    title="🎤 NEMO Hindi ASR Web Interface",
    description="Upload Hindi audio files and get transcriptions",
    version="1.0.0"
)

def preprocess_audio(file_path):
    """Preprocess audio for NEMO ASR model"""
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

@app.get("/", response_class=HTMLResponse)
async def serve_upload_page():
    """Serve the HTML upload interface"""
    html_path = "model/index.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    else:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>🎤 NEMO Hindi ASR</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }
                .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
                button { background: #4CAF50; color: white; padding: 15px 30px; border: none; border-radius: 5px; cursor: pointer; }
                .result { margin-top: 20px; padding: 15px; background: #f0f0f0; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>🎤 NEMO Hindi ASR</h1>
            <p>Upload a Hindi audio file (.wav) to get transcription</p>
            
            <form id="uploadForm" action="/transcribe" method="post" enctype="multipart/form-data">
                <div class="upload-area">
                    <p>📁 Choose your Hindi audio file</p>
                    <input type="file" name="file" accept=".wav" required>
                </div>
                <button type="submit">🚀 Transcribe Audio</button>
            </form>

            <div id="result" class="result" style="display: none;">
                <h3>📝 Transcription Result:</h3>
                <div id="transcription"></div>
                <div id="details"></div>
            </div>

            <script>
                document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                    e.preventDefault();
                    
                    const formData = new FormData(this);
                    const button = this.querySelector('button');
                    const result = document.getElementById('result');
                    
                    button.textContent = '⏳ Processing...';
                    button.disabled = true;
                    result.style.display = 'none';
                    
                    try {
                        const response = await fetch('/transcribe', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const data = await response.json();
                        
                        if (response.ok) {
                            document.getElementById('transcription').innerHTML = 
                                `<strong style="font-size: 18px; color: #2196F3;">"${data.transcription}"</strong>`;
                            document.getElementById('details').innerHTML = 
                                `<br><small>Confidence: ${data.confidence} | Duration: ${data.audio_duration}s | Tokens: ${data.num_tokens}</small>`;
                            result.style.display = 'block';
                        } else {
                            alert('Error: ' + data.detail);
                        }
                    } catch (error) {
                        alert('Error: ' + error.message);
                    }
                    
                    button.textContent = '🚀 Transcribe Audio';
                    button.disabled = false;
                });
            </script>
        </body>
        </html>
        """)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model_loaded": True, 
        "vocab_size": len(vocab),
        "message": "🎤 NEMO Hindi ASR is ready!"
    }

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Transcribe Hindi audio to text"""
    
    # Validate file
    if not file.filename.endswith(".wav"):
        raise HTTPException(
            status_code=400, 
            detail="Only .wav files are supported. Please convert your audio to WAV format."
        )

    if file.size and file.size > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 10MB."
        )

    tmp_path = None
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        print(f"🎵 Processing: {file.filename} ({len(content)} bytes)")
        
        # Preprocess audio
        features, time_length = preprocess_audio(tmp_path)
        print(f"📊 Features shape: {features.shape}, Duration: {time_length} frames")

        # ONNX inference
        inputs = {
            "audio_signal": features,
            "length": np.array([time_length], dtype=np.int64)
        }

        print("🤖 Running inference...")
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

        # Calculate confidence (simplified)
        try:
            probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = probs / np.sum(probs, axis=-1, keepdims=True)
            max_probs = np.max(probs, axis=-1)
            non_blank_indices = pred_ids != blank_token
            if np.any(non_blank_indices):
                confidence = float(np.mean(max_probs[0][non_blank_indices]))
            else:
                confidence = 0.0
        except:
            confidence = 0.75  # Default confidence
        
        result = {
            "transcription": transcription,
            "confidence": round(confidence, 3),
            "audio_duration": round(time_length * 0.01, 2),
            "num_tokens": len(decoded_ids),
            "status": "success"
        }
        
        print(f"✅ Result: {result}")
        return result

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    print("🚀 Starting NEMO Hindi ASR Web Server...")
    print("🌐 Open your browser and go to: http://localhost:8000")
    print("📁 Upload WAV files and get instant transcriptions!")
    print("=" * 50)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info",
        reload=False
    )
