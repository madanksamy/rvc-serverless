# RVC Serverless Handler for RunPod
# Tamil Singer Voice Conversion

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /workspace

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone Applio
RUN git clone --depth 1 https://github.com/IAHispano/Applio.git /workspace/Applio

# Install Applio requirements (skip torch as it's already in base image)
WORKDIR /workspace/Applio
RUN pip install --upgrade pip && \
    grep -v "^torch==" requirements.txt | grep -v "^torchaudio==" | grep -v "^torchvision==" > requirements_no_torch.txt && \
    pip install --no-cache-dir -r requirements_no_torch.txt || echo "Some deps failed, continuing..."

# Install torchcrepe and other missing dependencies
RUN pip install --no-cache-dir torchcrepe praat-parselmouth pyworld

# Ensure PyTorch works with CUDA
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Download RVC prerequisite models (hubert, rmvpe, fcpe) - this is critical!
RUN python -c "import sys; sys.path.insert(0, '/workspace/Applio'); from rvc.lib.tools.prerequisites_download import prequisites_download_pipeline; prequisites_download_pipeline(True, True, False)"

# Verify prerequisites were downloaded
RUN ls -la /workspace/Applio/rvc/models/predictors/ && \
    ls -la /workspace/Applio/rvc/models/pretraineds/embedders/

# Install RunPod and AWS SDK
RUN pip install --no-cache-dir runpod boto3

# Create models directory
RUN mkdir -p /workspace/models

# Copy handler
WORKDIR /workspace
COPY src/handler.py /workspace/handler.py

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
