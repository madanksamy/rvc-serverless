"""
RunPod Serverless Handler for RVC Voice Conversion
Optimized for Tamil singer models
"""

import runpod
import os
import sys
import base64
import tempfile
import traceback
from pathlib import Path

# Add Applio to path
APPLIO_PATH = "/workspace/Applio"
sys.path.insert(0, APPLIO_PATH)
os.chdir(APPLIO_PATH)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger().setLevel(logging.ERROR)

# Global converter instance (loaded once, reused)
converter = None
loaded_model_id = None

def get_converter():
    """Get or create the VoiceConverter instance."""
    global converter
    if converter is None:
        print("Loading VoiceConverter...")
        from rvc.infer.infer import VoiceConverter
        converter = VoiceConverter()
        print(f"VoiceConverter loaded on device: {converter.config.device}")
    return converter

def download_model_from_s3(model_id: str) -> tuple:
    """Download model from S3 if not already present."""
    import boto3

    models_dir = Path("/workspace/models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / f"{model_id}.pth"
    index_path = models_dir / f"{model_id}.index"

    # Check if already downloaded
    if model_path.exists():
        print(f"Model {model_id} already exists locally")
        return str(model_path), str(index_path) if index_path.exists() else ""

    # Download from S3
    print(f"Downloading model {model_id} from S3...")
    s3 = boto3.client('s3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('AWS_REGION', 'us-east-1')
    )

    bucket = os.environ.get('S3_BUCKET', 'synthica-rvc-models')

    try:
        # Download model file
        s3.download_file(bucket, f"models/{model_id}.pth", str(model_path))
        print(f"Downloaded {model_id}.pth")

        # Try to download index file
        try:
            s3.download_file(bucket, f"models/{model_id}.index", str(index_path))
            print(f"Downloaded {model_id}.index")
        except:
            print(f"No index file for {model_id}")
            index_path = ""
    except Exception as e:
        raise Exception(f"Failed to download model {model_id}: {e}")

    return str(model_path), str(index_path) if index_path else ""

def handler(job):
    """
    RunPod serverless handler for RVC voice conversion.

    Input:
    {
        "input": {
            "audio_base64": "base64 encoded audio",
            "model_id": "spb" | "kj_yesudas" | "s_janaki" | etc,
            "pitch": 0,  # -12 to 12
            "index_ratio": 0.75,  # 0 to 1
            "filter_radius": 3,  # 0 to 7
            "rms_mix_rate": 0.25,  # 0 to 1
            "protect": 0.33,  # 0 to 0.5
            "f0_method": "rmvpe"  # rmvpe, fcpe, crepe
        }
    }

    Output:
    {
        "audio_base64": "base64 encoded converted audio",
        "duration_ms": 1234,
        "model_id": "spb"
    }
    """
    try:
        job_input = job["input"]

        # Extract parameters
        audio_base64 = job_input.get("audio_base64")
        model_id = job_input.get("model_id", "spb")
        pitch = int(job_input.get("pitch", 0))
        index_ratio = float(job_input.get("index_ratio", 0.75))
        filter_radius = int(job_input.get("filter_radius", 3))
        rms_mix_rate = float(job_input.get("rms_mix_rate", 0.25))
        protect = float(job_input.get("protect", 0.33))
        f0_method = job_input.get("f0_method", "rmvpe")

        if not audio_base64:
            return {"error": "audio_base64 is required"}

        # Validate f0_method
        valid_methods = ["rmvpe", "fcpe", "crepe", "crepe-tiny"]
        if f0_method not in valid_methods:
            f0_method = "rmvpe"

        print(f"Converting with model={model_id}, pitch={pitch}, f0={f0_method}")

        # Get model paths
        model_path, index_path = download_model_from_s3(model_id)

        # Create temp files
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as input_file:
            input_path = input_file.name
            audio_bytes = base64.b64decode(audio_base64)
            input_file.write(audio_bytes)

        output_path = tempfile.mktemp(suffix=".wav")

        try:
            import time
            start_time = time.time()

            # Get converter
            conv = get_converter()

            # Run conversion
            conv.convert_audio(
                audio_input_path=input_path,
                audio_output_path=output_path,
                model_path=model_path,
                index_path=index_path,
                pitch=pitch,
                f0_method=f0_method,
                index_rate=index_ratio,
                volume_envelope=rms_mix_rate,
                protect=protect,
                hop_length=128,
                split_audio=False,
                f0_autotune=False,
                embedder_model='contentvec',
                clean_audio=False,
                export_format='WAV',
                resample_sr=0
            )

            duration_ms = int((time.time() - start_time) * 1000)
            print(f"Conversion completed in {duration_ms}ms")

            # Read output and encode
            with open(output_path, "rb") as f:
                output_base64 = base64.b64encode(f.read()).decode('utf-8')

            return {
                "audio_base64": output_base64,
                "duration_ms": duration_ms,
                "model_id": model_id,
                "params": {
                    "pitch": pitch,
                    "index_ratio": index_ratio,
                    "f0_method": f0_method,
                    "protect": protect
                }
            }

        finally:
            # Cleanup temp files
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# For local testing
if __name__ == "__main__":
    # Test with a simple job
    test_job = {
        "input": {
            "audio_base64": "",  # Add test audio
            "model_id": "spb",
            "pitch": 0
        }
    }
    print("Handler loaded successfully")
