"""
RunPod Serverless Handler for RVC Voice Conversion
Tamil Singer Models
"""

import runpod
import os
import sys
import base64
import tempfile
import time
import subprocess
import traceback
from pathlib import Path

print("=== RVC Serverless Handler Starting ===", flush=True)

# Setup paths
WORKSPACE = Path("/workspace")
APPLIO_PATH = WORKSPACE / "Applio"
MODELS_DIR = WORKSPACE / "models"

def setup_environment():
    """One-time setup of Applio and models."""
    print("Setting up environment...", flush=True)

    # Create directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Add Applio to path FIRST
    if str(APPLIO_PATH) not in sys.path:
        sys.path.insert(0, str(APPLIO_PATH))
    os.chdir(str(APPLIO_PATH))

    # Check if prerequisites exist
    hubert_path = APPLIO_PATH / "rvc" / "models" / "pretraineds" / "embedders" / "contentvec"
    rmvpe_path = APPLIO_PATH / "rvc" / "models" / "predictors" / "rmvpe.pt"

    print(f"Checking prerequisites...", flush=True)
    print(f"  Applio exists: {APPLIO_PATH.exists()}", flush=True)
    print(f"  Hubert dir exists: {hubert_path.exists()}", flush=True)
    print(f"  RMVPE exists: {rmvpe_path.exists()}", flush=True)

    # Always try to download prerequisites if missing
    if not rmvpe_path.exists():
        print("Downloading RVC prerequisite models...", flush=True)
        try:
            # Import and run prerequisite download
            from rvc.lib.tools.prerequisites_download import prequisites_download_pipeline
            prequisites_download_pipeline(True, True, False)
            print("Prerequisites downloaded!", flush=True)
        except Exception as e:
            print(f"Warning: prerequisite download failed: {e}", flush=True)
            traceback.print_exc()

    print("Environment ready!", flush=True)

def download_model_from_s3(model_id: str) -> tuple:
    """Download model from S3 if not cached."""
    import boto3

    model_path = MODELS_DIR / f"{model_id}.pth"
    index_path = MODELS_DIR / f"{model_id}.index"

    if model_path.exists():
        print(f"Model {model_id} cached locally", flush=True)
        return str(model_path), str(index_path) if index_path.exists() else ""

    print(f"Downloading {model_id} from S3...", flush=True)

    s3 = boto3.client('s3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name='us-east-1'
    )

    bucket = 'synthica-rvc-models'

    # Try v2 models first, then v1
    for version in ['v2', 'v1']:
        try:
            s3.download_file(bucket, f"models/{version}/{model_id}.pth", str(model_path))
            print(f"Downloaded models/{version}/{model_id}.pth", flush=True)

            try:
                s3.download_file(bucket, f"models/{version}/{model_id}.index", str(index_path))
                print(f"Downloaded index file", flush=True)
            except:
                pass

            return str(model_path), str(index_path) if index_path.exists() else ""
        except Exception as e:
            print(f"  Not found in {version}: {e}", flush=True)
            continue

    raise Exception(f"Model {model_id} not found in S3")

# Global converter (loaded once per worker)
_converter = None

def get_converter():
    """Get or create the VoiceConverter instance."""
    global _converter
    if _converter is None:
        setup_environment()
        print("Loading VoiceConverter...", flush=True)
        from rvc.infer.infer import VoiceConverter
        _converter = VoiceConverter()
        print(f"Converter ready on device: {_converter.config.device}", flush=True)
    return _converter

def handler(job):
    """
    RunPod handler for RVC voice conversion.
    """
    try:
        inp = job.get("input", {})

        # Validate input
        audio_b64 = inp.get("audio_base64")
        if not audio_b64:
            return {"error": "audio_base64 is required"}

        # Parse parameters
        model_id = inp.get("model_id", "spb")
        pitch = int(inp.get("pitch", 0))
        index_ratio = float(inp.get("index_ratio", 0.75))
        f0_method = inp.get("f0_method", "rmvpe")
        protect = float(inp.get("protect", 0.33))
        rms_mix = float(inp.get("rms_mix_rate", 0.25))

        # Validate f0 method
        if f0_method not in ["rmvpe", "fcpe", "crepe"]:
            f0_method = "rmvpe"

        print(f"Job: model={model_id}, pitch={pitch}, f0={f0_method}", flush=True)

        # Download model from S3
        model_path, index_path = download_model_from_s3(model_id)

        # Verify model file exists and is valid
        if not os.path.exists(model_path):
            return {"error": f"Model file not found: {model_path}"}

        model_size = os.path.getsize(model_path)
        print(f"Model file: {model_path} ({model_size} bytes)", flush=True)

        # Create temp files
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            input_path = f.name
            audio_data = base64.b64decode(audio_b64)
            f.write(audio_data)

        # Use a fixed output path in /tmp
        output_path = f"/tmp/rvc_output_{job.get('id', 'unknown')}.wav"

        try:
            start = time.time()

            # Verify input file was written correctly
            input_size = os.path.getsize(input_path)
            print(f"Input file: {input_path} ({input_size} bytes)", flush=True)

            # Get converter and run
            conv = get_converter()
            print(f"Running conversion...", flush=True)
            print(f"  Model: {model_path}", flush=True)
            print(f"  Index: {index_path}", flush=True)
            print(f"  Output: {output_path}", flush=True)

            # Call convert_audio
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
            print(f"Conversion call completed in {duration_ms}ms", flush=True)

            # List /tmp to see what was created
            print(f"Files in /tmp:", flush=True)
            for f in os.listdir("/tmp"):
                if f.endswith('.wav'):
                    print(f"  {f}", flush=True)

            # Check if output file exists
            if not os.path.exists(output_path):
                # Try to find any output file
                possible_outputs = [f for f in os.listdir("/tmp") if f.endswith('.wav') and 'output' in f.lower()]
                print(f"Possible outputs: {possible_outputs}", flush=True)
                return {"error": f"Output file not created at {output_path}"}

            output_size = os.path.getsize(output_path)
            print(f"Output file: {output_path} ({output_size} bytes)", flush=True)

            if output_size == 0:
                return {"error": "Output file is empty"}

            # Read and encode output
            with open(output_path, "rb") as f:
                out_b64 = base64.b64encode(f.read()).decode()

            return {
                "audio_base64": out_b64,
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
            for p in [input_path, output_path]:
                if os.path.exists(p):
                    try:
                        os.unlink(p)
                    except:
                        pass

    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        traceback.print_exc()
        return {"error": str(e)}

# Start the serverless handler
print("Starting RunPod serverless handler...", flush=True)
runpod.serverless.start({"handler": handler})
