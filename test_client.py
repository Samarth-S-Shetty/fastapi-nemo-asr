#!/usr/bin/env python3
"""
Simple test client for the NEMO ASR API
"""
import requests
import sys
import os

def test_transcribe(audio_file_path):
    """Test the transcribe endpoint with an audio file"""

    if not os.path.exists(audio_file_path):
        print(f"❌ Audio file not found: {audio_file_path}")
        return

    if not audio_file_path.endswith('.wav'):
        print("❌ Only .wav files are supported")
        return

    url = "http://localhost:8000/transcribe"

    try:
        with open(audio_file_path, 'rb') as f:
            files = {'file': (os.path.basename(audio_file_path), f, 'audio/wav')}

            print(f"🎵 Uploading: {audio_file_path}")
            print("⏳ Processing...")

            response = requests.post(url, files=files)

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success!")
                print(f"📝 Transcription: '{result.get('transcription', 'N/A')}'")
                print(f"🎯 Confidence: {result.get('confidence', 'N/A')}")
                print(f"⏱️  Duration: {result.get('audio_duration', 'N/A')}s")
                print(f"🔤 Tokens: {result.get('num_tokens', 'N/A')}")
                print(f"📊 Status: {result.get('status', 'N/A')}")
                print(f"🔍 Full Response: {result}")
            else:
                print(f"❌ Error {response.status_code}: {response.text}")

    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API. Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_health():
    """Test if the API is running"""
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API is running!")
            print(f"📊 Vocab size: {result.get('vocab_size', 'unknown')}")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API. Make sure the server is running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🎤 NEMO ASR API Test Client")
    print("=" * 40)

    # Test if API is running
    if not test_health():
        sys.exit(1)

    print()

    # Test transcription
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        test_transcribe(audio_file)
    else:
        print("Usage: python test_client.py <path_to_wav_file>")
        print("Example: python test_client.py sample.wav")
