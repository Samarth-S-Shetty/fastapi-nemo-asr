# 🤖 Model Setup Instructions

## 📋 Overview

The ONNX model file (`asr_model.onnx`) is not included in this repository due to GitHub's 100MB file size limit. This file contains the optimized NVIDIA NeMo Hindi ASR model required for the application to function.

## 📊 Model Details

- **Original Model**: `stt_hi_conformer_ctc_medium` from NVIDIA NeMo
- **Optimized Format**: ONNX Runtime
- **File Size**: ~500MB (126.88 MB compressed)
- **Language**: Hindi
- **Architecture**: Conformer with CTC loss
- **Vocabulary**: 128 tokens (included in `model/vocab.txt`)

## 🔧 Setup Options

### Option 1: For Assignment Reviewers

**Contact Information:**
- **Repository Owner**: Samarth S Shetty
- **GitHub**: [@Samarth-S-Shetty](https://github.com/Samarth-S-Shetty)
- **Request**: Please contact for the `asr_model.onnx` file

**File Placement:**
```bash
# Place the downloaded file in the model directory
fastapi-nemo-asr/
├── model/
│   ├── asr_model.onnx       # ← Place downloaded file here
│   ├── vocab.txt            # ✅ Already included
│   └── index.html           # ✅ Already included
```

### Option 2: Download and Convert Original Model

If you want to recreate the ONNX model yourself:

```bash
# Install NVIDIA NeMo (large dependency)
pip install nemo-toolkit[asr]

# Download and convert the model (Python script)
python -c "
import nemo.collections.asr as nemo_asr

# Load the pre-trained model
model = nemo_asr.models.EncDecCTCModel.from_pretrained('stt_hi_conformer_ctc_medium')

# Export to ONNX
model.export('model/asr_model.onnx')
print('Model exported to model/asr_model.onnx')
"
```

### Option 3: Alternative Implementation

Modify the code to use the original NeMo model directly:

```python
# In app/main.py or web_server.py, replace ONNX loading with:
import nemo.collections.asr as nemo_asr

# Load model directly
model = nemo_asr.models.EncDecCTCModel.from_pretrained('stt_hi_conformer_ctc_medium')

# Use model.transcribe() instead of ONNX inference
```

## ✅ Verification

After placing the model file, verify the setup:

```bash
# Check file exists and size
ls -lh model/asr_model.onnx

# Expected output: ~500MB file

# Test the application
python web_server.py

# Test with sample audio
python test_client.py audio_files/Hindi_M_Devinder1.wav
```

## 🐛 Troubleshooting

### Model File Not Found
```
RuntimeError: ONNX model not found at model/asr_model.onnx
```
**Solution**: Ensure the `asr_model.onnx` file is placed in the `model/` directory.

### Model Loading Error
```
ONNXRuntimeError: Model loading failed
```
**Solution**: Verify the model file is not corrupted and is the correct ONNX format.

### Memory Issues
```
Out of memory error during model loading
```
**Solution**: Ensure you have at least 2GB of available RAM.

## 📞 Support

For assignment reviewers or users needing the model file:
- **GitHub Issues**: [Create an issue](https://github.com/Samarth-S-Shetty/fastapi-nemo-asr/issues)
- **Contact**: Repository owner via GitHub

## 📄 License Note

The NVIDIA NeMo model is subject to NVIDIA's licensing terms. This implementation is for educational/assignment purposes.
