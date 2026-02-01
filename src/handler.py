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

print("=== RVC Serverless Handler Starting ===")

# Setup paths
WORKSPACE = Path("/workspace")
APPLIO_PATH = WORKSPACE / "Applio"
MODELS_DIR = WORKSPACE / "models"

def setup_environment():
    """One-time setup of Applio and models."""
    print("Setting up environment...")

    # Create directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Clone Applio if not present
    if not APPLIO_PATH.exists():
        print("Cloning Applio...")
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/IAHispano/Applio.git",
            str(APPLIO_PATH)
        ], check=True, capture_output=True)

        # Install requirements
        print("Installing Applio requirements...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q",
            "-r", str(APPLIO_PATH / "requirements.txt")
        ], check=True, capture_output=True)

        # Download prerequisite models
        print("Downloading RVC models (hubert, rmvpe, fcpe)...")
        os.chdir(str(APPLIO_PATH))
        try:
            subprocess.run([
                sys.executable, "-c",
                "from rvc.lib.tools.prerequisites_download import prequisites_download_pipeline; "
                "prequisites_download_pipeline(True, True, True, True)"
            ], check=True, capture_output=True, timeout=300)
        except:
            print("Warning: Some prerequisite models may not have downloaded")

    # Add Applio to path
    if str(APPLIO_PATH) not in sys.path:
        sys.path.insert(0, str(APPLIO_PATH))
    os.chdir(str(APPLIO_PATH))

    print("Environment ready!")

def download_model_from_s3(model_id: str) -> tuple:
    """Download model from S3 if not cached."""
    import boto3

    model_path = MODELS_DIR / f"{model_id}.pth"
    index_path = MODELS_DIR / f"{model_id}.index"

    if model_path.exists():
        print(f"Model {model_id} cached locally")
        return str(model_path), str(index_path) if index_path.exists() else ""

    print(f"Downloading {model_id} from S3...")

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
            print(f"Downloaded models/{version}/{model_id}.pth")

            try:
                s3.download_file(bucket, f"models/{version}/{model_id}.index", str(index_path))
                print(f"Downloaded index file")
            except:
                pass

            return str(model_path), str(index_path) if index_path.exists() else ""
        except Exception as e:
            continue

    raise Exception(f"Model {model_id} not found in S3")

# Global converter (loaded once per worker)
_converter = None

def get_converter():
    """Get or create the VoiceConverter instance."""
    global _converter
    if _converter is None:
        setup_environment()
        print("Loading VoiceConverter...")
        from rvc.infer.infer import VoiceConverter
        _converter = VoiceConverter()
        print(f"Converter ready on: {_converter.config.device}")
    return _converter

def handler(job):
    """
    RunPod handler for RVC voice conversion.

    Input:
    {
        "audio_base64": "base64 encoded WAV/MP3",
        "model_id": "spb" | "kj_yesudas" | "s_janaki" | etc,
        "pitch": 0,           # -12 to 12
        "index_ratio": 0.75,  # 0 to 1
        "f0_method": "rmvpe", # rmvpe, fcpe, crepe
        "protect": 0.33,      # 0 to 0.5
        "rms_mix_rate": 0.25  # 0 to 1
    }

    Output:
    {
        "audio_base64": "converted audio base64",
        "duration_ms": 1234,
        "model_id": "spb"
    }
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

        print(f"Job: model={model_id}, pitch={pitch}, f0={f0_method}")

        # Download model from S3
        model_path, index_path = download_model_from_s3(model_id)

        # Create temp files
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            input_path = f.name
            f.write(base64.b64decode(audio_b64))

        output_path = tempfile.mktemp(suffix=".wav")

        try:
            start = time.time()

            # Verify input file was written correctly
            input_size = os.path.getsize(input_path)
            print(f"Input file: {input_path} ({input_size} bytes)")

            # Get converter and run
            conv = get_converter()
            print(f"Running conversion with model: {model_path}")
            print(f"Index: {index_path}, pitch: {pitch}, f0: {f0_method}")

            result = conv.convert_audio(
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

            print(f"convert_audio returned: {result}")

            duration_ms = int((time.time() - start) * 1000)
            print(f"Conversion done in {duration_ms}ms")

            # Check if output file exists
            if not os.path.exists(output_path):
                # Sometimes Applio returns the path instead of writing to our path
                if result and isinstance(result, str) and os.path.exists(result):
                    output_path = result
                    print(f"Using returned path: {output_path}")
                else:
                    return {"error": f"Output file not created. convert_audio returned: {result}"}

            output_size = os.path.getsize(output_path)
            print(f"Output file: {output_path} ({output_size} bytes)")

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
        traceback.print_exc()
        return {"error": str(e)}

# Start the serverless handler
print("Starting RunPod serverless handler...")
runpod.serverless.start({"handler": handler})
