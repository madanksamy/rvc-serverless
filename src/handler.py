"""
RunPod Serverless Handler for RVC Voice Conversion
"""

import runpod
import os
import sys
import base64
import tempfile
import time
import traceback
from pathlib import Path

print("=== RVC Handler Starting ===", flush=True)

WORKSPACE = Path("/workspace")
APPLIO_PATH = WORKSPACE / "Applio"
MODELS_DIR = WORKSPACE / "models"

def setup():
    """Setup environment once."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if str(APPLIO_PATH) not in sys.path:
        sys.path.insert(0, str(APPLIO_PATH))
    os.chdir(str(APPLIO_PATH))

    # Download prerequisites if needed
    rmvpe = APPLIO_PATH / "rvc" / "models" / "predictors" / "rmvpe.pt"
    if not rmvpe.exists():
        print("Downloading prerequisites...", flush=True)
        try:
            from rvc.lib.tools.prerequisites_download import prequisites_download_pipeline
            prequisites_download_pipeline(True, True, False)
        except Exception as e:
            print(f"Prereq download failed: {e}", flush=True)

def download_model(model_id):
    """Download model from S3."""
    import boto3

    model_path = MODELS_DIR / f"{model_id}.pth"
    index_path = MODELS_DIR / f"{model_id}.index"

    if model_path.exists():
        return str(model_path), str(index_path) if index_path.exists() else ""

    print(f"Downloading {model_id} from S3...", flush=True)

    s3 = boto3.client('s3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name='us-east-1'
    )

    for ver in ['v2', 'v1']:
        try:
            s3.download_file('synthica-rvc-models', f"models/{ver}/{model_id}.pth", str(model_path))
            try:
                s3.download_file('synthica-rvc-models', f"models/{ver}/{model_id}.index", str(index_path))
            except:
                pass
            return str(model_path), str(index_path) if index_path.exists() else ""
        except:
            continue

    raise Exception(f"Model {model_id} not found")

# Global converter
_conv = None

def get_converter():
    global _conv
    if _conv is None:
        setup()
        from rvc.infer.infer import VoiceConverter
        _conv = VoiceConverter()
        print(f"Converter ready: {_conv.config.device}", flush=True)
    return _conv

def handler(job):
    """Main handler - MUST return dict with output data."""
    job_input = job.get("input", {})

    try:
        audio_b64 = job_input.get("audio_base64")
        if not audio_b64:
            return {"error": "No audio_base64 provided"}

        model_id = job_input.get("model_id", "spb")
        pitch = int(job_input.get("pitch", 0))
        f0_method = job_input.get("f0_method", "rmvpe")
        index_ratio = float(job_input.get("index_ratio", 0.75))
        protect = float(job_input.get("protect", 0.33))
        rms_mix = float(job_input.get("rms_mix_rate", 0.25))

        print(f"Job: {model_id}, pitch={pitch}, f0={f0_method}", flush=True)

        # Get model
        model_path, index_path = download_model(model_id)

        # Write input to temp file
        input_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        input_file.write(base64.b64decode(audio_b64))
        input_file.close()
        input_path = input_file.name

        output_path = f"/tmp/out_{job.get('id', 'x')}.wav"

        start = time.time()

        # Convert
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

        elapsed = int((time.time() - start) * 1000)
        print(f"Conversion done: {elapsed}ms", flush=True)

        # Check output
        if not os.path.exists(output_path):
            return {"error": "No output file created"}

        if os.path.getsize(output_path) == 0:
            return {"error": "Output file is empty"}

        # Read output
        with open(output_path, "rb") as f:
            out_b64 = base64.b64encode(f.read()).decode()

        # Cleanup
        try:
            os.unlink(input_path)
            os.unlink(output_path)
        except:
            pass

        print(f"Returning audio: {len(out_b64)} chars", flush=True)

        return {
            "audio_base64": out_b64,
            "duration_ms": elapsed,
            "model_id": model_id
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

print("Starting handler...", flush=True)
runpod.serverless.start({"handler": handler})
