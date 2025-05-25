# 🎤 FastAPI NVIDIA NeMo ASR Application

A production-ready FastAPI application for Hindi speech recognition using NVIDIA NeMo's Conformer CTC model, optimized with ONNX for fast inference.

## 🎯 Features

- **🇮🇳 Hindi ASR**: NVIDIA NeMo `stt_hi_conformer_ctc_medium` model
- **⚡ ONNX Optimized**: Fast inference with ONNX Runtime
- **🌐 Dual Servers**: Choose API-only or integrated web interface
- **📱 Web Interface**: Beautiful HTML upload interface with real-time results
- **🎵 Audio Processing**: Handles 5-10 second WAV files at 16kHz
- **📊 Rich Responses**: Confidence scores, duration, and token count
- **🐳 Docker Ready**: Fully containerized with multi-stage build
- **🔧 Developer Friendly**: RESTful API with OpenAPI documentation

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ or Docker
- WAV audio files (16kHz recommended)
- **Model Setup**: The ONNX model file (`asr_model.onnx`) is not included due to GitHub's 100MB size limit.

#### Model Download Options:
1. **For Assignment Reviewers**: Contact the repository owner for the model file
2. **For General Users**: Download the original NVIDIA NeMo model and convert to ONNX:
   ```bash
   # Download original model (requires nemo-toolkit)
   # Place the converted asr_model.onnx in the model/ directory
   ```
3. **Alternative**: Use the original NeMo model directly (modify code accordingly)

### Option 1: Web Interface (Easiest)

```bash
# Install dependencies
pip install -r requirements.txt

# Start web server with upload interface
python web_server.py

# Open browser: http://localhost:8000
# Upload WAV files and get instant transcriptions!
```

### Option 2: API Server Only

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
python app/main.py

# API available at: http://localhost:8000
# Docs at: http://localhost:8000/docs
```

### Option 3: Docker (Production)

```bash
# Build container
docker build -t nemo-asr-api .

# Run container (includes web interface)
docker run -p 8000:8000 nemo-asr-api

# Access at: http://localhost:8000
```

## 📡 API Usage

### Endpoints

- **GET /** - API information and health check
- **POST /transcribe** - Transcribe audio file
- **GET /health** - Health check endpoint
- **GET /docs** - Interactive API documentation (Swagger UI)

### Method 1: Web Interface (Recommended)

1. Start the server: `python web_server.py`
2. Open browser: `http://localhost:8000`
3. Upload your WAV file using the drag-and-drop interface
4. Get instant transcription results with confidence scores

### Method 2: Command Line Testing

```bash
# Test with provided sample files
python test_client.py audio_files/Hindi_M_Devinder1.wav
python test_client.py audio_files/HIN_F_Geet.wav
```

### Method 3: cURL Command

```bash
curl -X POST "http://localhost:8000/transcribe" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@audio_files/Hindi_M_Devinder1.wav"
```

### Method 4: Postman (Detailed Instructions)

#### Step-by-Step Postman Setup:

1. **Open Postman** and create a new request

2. **Set Request Method**: Change from GET to **POST**

3. **Set URL**: `http://localhost:8000/transcribe`

4. **Configure Headers**:
   - Go to the **Headers** tab
   - Add: `Accept: application/json`
   - ⚠️ **Important**: Don't add Content-Type manually, Postman will set it automatically for form-data

5. **Configure Body**:
   - Go to the **Body** tab
   - Select **form-data** (not raw or x-www-form-urlencoded)
   - Add a new key-value pair:
     - **Key**: `file`
     - **Type**: Select **File** from the dropdown (very important!)
     - **Value**: Click "Select Files" and choose your WAV audio file

6. **Send Request**: Click the **Send** button

7. **View Response**: You'll get a JSON response like:
   ```json
   {
     "transcription": "मुंबई शहर सेलर दूर",
     "confidence": 0.85,
     "audio_duration": 9.3,
     "num_tokens": 14,
     "status": "success"
   }
   ```

#### Postman Troubleshooting:
- ✅ Make sure your server is running first (`main.py`)
- ✅ Use WAV files only (other formats will return an error)
- ✅ File size limit is 10MB
- ✅ Select "File" type in the dropdown for the `file` key
- ✅ Don't set Content-Type header manually
- ✅ Use form-data, not raw or x-www-form-urlencoded
- ✅ For best results, use 5-10 second Hindi audio clips

### Method 5: Python Requests

```python
import requests

# Upload and transcribe audio
url = "http://localhost:8000/transcribe"
files = {"file": ("audio.wav", open("audio_files/Hindi_M_Devinder1.wav", "rb"), "audio/wav")}
response = requests.post(url, files=files)

if response.status_code == 200:
    result = response.json()
    print(f"Transcription: {result['transcription']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Duration: {result['audio_duration']}s")
    print(f"Tokens: {result['num_tokens']}")
else:
    print(f"Error: {response.text}")
```

## 📊 Sample Response

```json
{
  "transcription": "मुंबई शहर सेलर दूर",
  "confidence": 0.85,
  "audio_duration": 9.3,
  "num_tokens": 14,
  "status": "success"
}
```

## 🧪 Testing with Sample Files

The project includes sample Hindi audio files in `audio_files/`:

| File | Duration | Expected Transcription | Quality |
|------|----------|----------------------|---------|
| `Hindi_M_Devinder1.wav` | 9.3s | `मुंबई शहर सेलर दूर` | ✅ Excellent |
| `HIN_F_Geet.wav` | 8.5s | `की शुरुआत में आबादी वाले देश में` | ✅ Excellent |
| `HIN_F_DebjaniD.wav` | 6.2s | `न नम्का नमस्कार` | ✅ Good |
| `HIN_F_02.wav` | 3.1s | `सप् हा` | ⚠️ Partial (short) |

```bash
# Test all sample files
python test_client.py audio_files/Hindi_M_Devinder1.wav
python test_client.py audio_files/HIN_F_Geet.wav
python test_client.py audio_files/HIN_F_DebjaniD.wav
python test_client.py audio_files/HIN_F_02.wav
```

## 📁 Project Structure

```
fastapi-nemo-asr/
├── app/
│   └── main.py              # FastAPI API-only server
├── model/
│   ├── asr_model.onnx       # ONNX optimized model (500MB) - NOT IN REPO
│   ├── vocab.txt            # 128-token Hindi vocabulary
│   └── index.html           # Web upload interface
├── audio_files/             # Sample test audio files
│   ├── Hindi_M_Devinder1.wav    # Male voice sample
│   ├── HIN_F_Geet.wav           # Female voice sample
│   ├── HIN_F_DebjaniD.wav       # Greeting sample
│   └── HIN_F_02.wav             # Short clip sample
├── web_server.py            # 🌟 Integrated web server (RECOMMENDED)
├── test_client.py           # Command-line testing utility
├── Dockerfile               # Production-ready container
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── Description.md           # Detailed project documentation
├── MODEL_SETUP.md           # Model download and setup instructions
└── .gitignore               # Git ignore rules
```

## 🔧 Configuration

### Audio Requirements
- **Format**: WAV files only
- **Sample Rate**: 16kHz (recommended)
- **Duration**: 5-10 seconds optimal
- **Channels**: Mono (stereo will be converted automatically)
- **Max Size**: 10MB

### Model Details
- **Base Model**: `stt_hi_conformer_ctc_medium`
- **Language**: Hindi
- **Architecture**: Conformer with CTC loss
- **Optimization**: ONNX Runtime
- **Vocabulary**: 128 tokens (characters + subwords)

## 🐛 Troubleshooting

### Common Issues

1. **"Only .wav files are supported"**
   - Convert your audio to WAV format using audio editing software

2. **Empty transcription**
   - Check audio quality and language (Hindi only)
   - Ensure audio is 5-10 seconds long
   - Verify audio is clear with minimal background noise

3. **Connection refused**
   - Make sure the server is running on port 8000
   - Check if Docker container is properly started

4. **Postman file upload not working**
   - Ensure you select "File" type in the dropdown for the `file` key
   - Don't set Content-Type header manually
   - Use form-data, not raw or x-www-form-urlencoded

### Debug Commands
```bash
# Check if server is running
curl http://localhost:8000/health

# Test with sample file
python test_client.py audio_files/Hindi_M_Devinder1.wav
```

## 📊 Performance

- **Inference Time**: ~1-2 seconds for 10-second audio
- **Memory Usage**: ~2GB RAM
- **Supported Concurrent Requests**: 10+
- **Model Size**: ~500MB (ONNX optimized)

## 🙏 Acknowledgments

- NVIDIA NeMo team for the pre-trained model
- FastAPI community for the excellent framework
- ONNX Runtime for optimization capabilities