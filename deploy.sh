#!/bin/bash
# Deploy RVC Serverless to RunPod
# Requires: RUNPOD_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY environment variables

set -e

# Check required env vars
if [ -z "$RUNPOD_API_KEY" ]; then
    echo "Error: RUNPOD_API_KEY not set"
    exit 1
fi

DOCKER_IMAGE="synthica/rvc-serverless:latest"

echo "=== Building Docker Image ==="
docker build -t $DOCKER_IMAGE .

echo "=== Pushing to Docker Hub ==="
docker push $DOCKER_IMAGE

echo "=== Creating RunPod Serverless Endpoint ==="

# Create endpoint via RunPod API
curl -X POST "https://api.runpod.io/v2/endpoints" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"name\": \"rvc-tamil-voices\",
    \"templateId\": null,
    \"gpuIds\": \"NVIDIA RTX A4000\",
    \"networkVolumeId\": null,
    \"workersMin\": 0,
    \"workersMax\": 3,
    \"idleTimeout\": 5,
    \"flashBoot\": true,
    \"dockerImage\": \"$DOCKER_IMAGE\",
    \"env\": {
      \"AWS_ACCESS_KEY_ID\": \"$AWS_ACCESS_KEY_ID\",
      \"AWS_SECRET_ACCESS_KEY\": \"$AWS_SECRET_ACCESS_KEY\",
      \"AWS_REGION\": \"us-east-1\",
      \"S3_BUCKET\": \"synthica-rvc-models\"
    }
  }"

echo ""
echo "=== Deployment Complete ==="
echo "Save the endpoint ID from the response above"
