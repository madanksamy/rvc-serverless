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

WORKDIR /workspace/Applio

# Pin torch to version in base image to avoid conflicts
RUN pip install --upgrade pip && \
    sed -i '/^torch==/d' requirements.txt && \
    sed -i '/^torchaudio==/d' requirements.txt && \
    sed -i '/^torchvision==/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt || echo "Some deps failed"

# Install additional deps
RUN pip install --no-cache-dir torchcrepe praat-parselmouth pyworld runpod boto3

# Force reinstall compatible torch
RUN pip install --no-cache-dir torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Verify torch
RUN python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Download RVC prerequisites at runtime (they need network access)
# The handler will download these on first run

# Create models directory
RUN mkdir -p /workspace/models

# Copy handler
WORKDIR /workspace
COPY src/handler.py /workspace/handler.py

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
