#!/usr/bin/env python3
"""
Simple RVC API Server for RunPod
Run this on your RunPod GPU pod
"""

from flask import Flask, request, jsonify
import base64
import tempfile
import os
import sys
import time
import traceback

# Setup
APPLIO_PATH = "/workspace/Applio"
MODELS_DIR = "/workspace/models"

sys.path.insert(0, APPLIO_PATH)
os.chdir(APPLIO_PATH)

app = Flask(__name__)

# Global converter
_converter = None

def get_converter():
    global _converter
    if _converter is None:
        print("Loading VoiceConverter...")
        from rvc.infer.infer import VoiceConverter
        _converter = VoiceConverter()
        print(f"Loaded on device: {_converter.config.device}")
    return _converter

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "device": get_converter().config.device})

@app.route('/models', methods=['GET'])
def list_models():
    """List available models."""
    models = []
    for f in os.listdir(MODELS_DIR):
        if f.endswith('.pth'):
            model_id = f.replace('.pth', '')
            has_index = os.path.exists(os.path.join(MODELS_DIR, f"{model_id}.index"))
            models.append({
                "id": model_id,
                "name": model_id.replace('_', ' ').title(),
                "has_index": has_index
            })
    return jsonify({"models": models})

@app.route('/convert', methods=['POST'])
def convert():
    """
    Convert voice using RVC.

    POST JSON:
    {
        "audio_base64": "base64 encoded audio",
        "model_id": "spb",
        "pitch": 0,
        "index_ratio": 0.75,
        "f0_method": "rmvpe",
        "protect": 0.33
    }
    """
    try:
        data = request.json

        audio_b64 = data.get('audio_base64')
        if not audio_b64:
            return jsonify({"error": "audio_base64 required"}), 400

        model_id = data.get('model_id', 'spb')
        pitch = int(data.get('pitch', 0))
        index_ratio = float(data.get('index_ratio', 0.75))
        f0_method = data.get('f0_method', 'rmvpe')
        protect = float(data.get('protect', 0.33))
        rms_mix = float(data.get('rms_mix_rate', 0.25))

        # Validate f0 method
        if f0_method not in ['rmvpe', 'fcpe', 'crepe']:
            f0_method = 'rmvpe'

        # Model paths
        model_path = os.path.join(MODELS_DIR, f"{model_id}.pth")
        index_path = os.path.join(MODELS_DIR, f"{model_id}.index")

        if not os.path.exists(model_path):
            return jsonify({"error": f"Model {model_id} not found"}), 404

        if not os.path.exists(index_path):
            index_path = ""

        print(f"Converting: model={model_id}, pitch={pitch}, f0={f0_method}")

        # Temp files
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            input_path = f.name
            f.write(base64.b64decode(audio_b64))

        output_path = tempfile.mktemp(suffix='.wav')

        try:
            start = time.time()

            conv = get_converter()
            conv.convert_audio(
                audio_input_path=input_path,
                audio_output_path=output_path,
                model_path=model_path,
                index_path=index_path,
                pitch=pitch,
                f0_method=f0_method,
                index_rate=index_ratio,
                volume_envelope=rms_mix,
                protect=protect,
                hop_length=128,
                split_audio=False,
                f0_autotune=False,
                embedder_model='contentvec',
                clean_audio=False,
                export_format='WAV'
            )

            duration_ms = int((time.time() - start) * 1000)
            print(f"Done in {duration_ms}ms")

            with open(output_path, 'rb') as f:
                out_b64 = base64.b64encode(f.read()).decode()

            return jsonify({
                "success": True,
                "audio_base64": out_b64,
                "duration_ms": duration_ms,
                "model_id": model_id
            })

        finally:
            for p in [input_path, output_path]:
                if os.path.exists(p):
                    os.unlink(p)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Preload converter
    print("Preloading converter...")
    get_converter()
    print("Starting server on port 5000...")
    app.run(host='0.0.0.0', port=5000, threaded=True)
