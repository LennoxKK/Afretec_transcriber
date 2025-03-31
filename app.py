from flask import Flask, request, jsonify
from flask_cors import CORS
import torchaudio
import ffmpeg
import numpy as np
from transformers import pipeline
import os
import tempfile
import time
import torch
import json
import Levenshtein

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit
CORS(app)  # Enable CORS for all routes

# Audio format support
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a', '3gp', 'mp4', 'aac'}
ALLOWED_MIME_TYPES = {
    'audio/wav', 'audio/x-wav',
    'audio/mpeg', 'audio/mp3',
    'audio/ogg', 'audio/webm',
    'audio/x-m4a', 'audio/m4a',  # Added audio/m4a
    'audio/mp4',
    'audio/3gpp', 'audio/aac',
    'application/octet-stream'
}

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
print("⚡ Loading Whisper model...")
try:
    transcriber = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        device=DEVICE,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )
    print(f"✅ Model loaded on {DEVICE.upper()}")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    exit(1)

def is_valid_file(filename, content_type):
    ext = filename.split('.')[-1].lower()
    return (ext in ALLOWED_EXTENSIONS and 
            content_type in ALLOWED_MIME_TYPES)

def convert_to_wav(input_path):
    """Convert any audio file to WAV format"""
    try:
        output_path = f"{input_path}.wav"
        (
            ffmpeg.input(input_path)
            .output(output_path,
                   format='wav',
                   acodec='pcm_s16le',
                   ac=1,
                   ar='16k',
                   threads=0)
            .run(overwrite_output=True, quiet=True)
        )
        return output_path
    except ffmpeg.Error as e:
        print(f"FFmpeg conversion error: {e.stderr.decode()}")
        return None

@app.route('/transcribe', methods=['POST'])
def transcribe():
    start_time = time.time()
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Get options from form data
    options = request.form.get('options')
    if not options:
        return jsonify({"error": "No options provided"}), 400
    
    try:
        options_list = json.loads(options)
        if not isinstance(options_list, list):
            return jsonify({"error": "Options must be an array"}), 400
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid options format"}), 400

    if not is_valid_file(file.filename, file.content_type):
        return jsonify({
            "error": "Unsupported file type",
            "allowed_extensions": list(ALLOWED_EXTENSIONS),
            "received_type": file.content_type
        }), 400

    temp_path = None
    converted_path = None
    
    try:
        # Save original file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file.save(tmp.name)
            temp_path = tmp.name

        # Convert to WAV if needed
        if not file.filename.lower().endswith('.wav'):
            converted_path = convert_to_wav(temp_path)
            if not converted_path:
                return jsonify({"error": "Audio conversion failed"}), 400
            audio_path = converted_path
        else:
            audio_path = temp_path

        # Load and normalize audio
        waveform = torchaudio.load(audio_path)[0].numpy()[0]
        waveform = waveform / np.max(np.abs(waveform))

        # Transcribe
        transcribe_start = time.time()
        result = transcriber(
            waveform,
            chunk_length_s=30,
            stride_length_s=5,
            batch_size=16,
            return_timestamps=False
        )
        transcribe_time = time.time() - transcribe_start

        # Find best match from options
        transcribed_text = result["text"].strip().lower()
        best_match = None
        min_distance = float('inf')
        
        for option in options_list:
            distance = Levenshtein.distance(transcribed_text, option.lower())
            if distance < min_distance:
                min_distance = distance
                best_match = option

        # Return results
        response = {
            "text": result["text"],
            "options": options_list,
            "match_analysis": {
                "best_match": best_match,
                "distance": min_distance,
                "is_match": min_distance <= 3  # Threshold for acceptable match
            },
            "time": {
                "total": round(time.time() - start_time, 3),
                "transcription": round(transcribe_time, 3)
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "error": str(e),
            "time": round(time.time() - start_time, 3)
        }), 500

    finally:
        # Clean up files
        for path in [temp_path, converted_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
    
    
# git init
# git add README.md
# git commit -m "first commit"
# git branch -M master
# git remote add origin https://github.com/LennoxKK/Afretec_transcriber.git
# git push -u origin master