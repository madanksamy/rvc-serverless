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

# Install Applio requirements
WORKDIR /workspace/Applio
RUN pip install --no-cache-dir -r requirements.txt

# Download RVC prerequisite models (hubert, rmvpe, fcpe)
RUN python -c "from rvc.lib.tools.prerequisites_download import prequisites_download_pipeline; prequisites_download_pipeline(True, True, True, True)" || true

# Install RunPod and AWS SDK
RUN pip install --no-cache-dir runpod boto3

# Create models directory
RUN mkdir -p /workspace/models

# Copy handler
WORKDIR /workspace
COPY src/handler.py /workspace/handler.py

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
