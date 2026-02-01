#!/usr/bin/env python3
"""
Deploy RVC Serverless to RunPod
"""

import requests
import json
import os

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
DOCKER_IMAGE = "iahispano/applio:latest"  # Use official Applio image

# AWS credentials for S3 access (from environment)
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")

def create_endpoint():
    """Create a serverless endpoint on RunPod."""

    # GraphQL mutation to create endpoint
    query = """
    mutation createEndpoint($input: EndpointInput!) {
        saveEndpoint(input: $input) {
            id
            name
            templateId
            gpuIds
            workersMin
            workersMax
        }
    }
    """

    variables = {
        "input": {
            "name": "rvc-tamil-voices",
            "templateId": None,
            "gpuIds": "AMPERE_16",  # RTX A4000 or similar
            "workersMin": 0,
            "workersMax": 3,
            "idleTimeout": 5,
            "flashBoot": True,
            "volumeInGb": 20,
            "env": [
                {"key": "AWS_ACCESS_KEY_ID", "value": AWS_ACCESS_KEY},
                {"key": "AWS_SECRET_ACCESS_KEY", "value": AWS_SECRET_KEY},
                {"key": "S3_BUCKET", "value": "synthica-rvc-models"}
            ]
        }
    }

    response = requests.post(
        "https://api.runpod.io/graphql",
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json"
        },
        json={"query": query, "variables": variables}
    )

    print("Response:", response.json())
    return response.json()

def list_endpoints():
    """List existing endpoints."""
    query = """
    query {
        myself {
            endpoints {
                id
                name
                gpuIds
                workersMin
                workersMax
            }
        }
    }
    """

    response = requests.post(
        "https://api.runpod.io/graphql",
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json"
        },
        json={"query": query}
    )

    return response.json()

def test_endpoint(endpoint_id: str, audio_path: str, model_id: str = "spb"):
    """Test an endpoint with audio file."""
    import base64

    # Read audio file
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    # Send request
    response = requests.post(
        f"https://api.runpod.io/v2/{endpoint_id}/runsync",
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "input": {
                "audio_base64": audio_b64,
                "model_id": model_id,
                "pitch": 0,
                "f0_method": "rmvpe",
                "index_ratio": 0.75
            }
        },
        timeout=300
    )

    return response.json()

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python deploy.py [list|create|test <endpoint_id> <audio_file>]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "list":
        print(json.dumps(list_endpoints(), indent=2))
    elif cmd == "create":
        print(json.dumps(create_endpoint(), indent=2))
    elif cmd == "test" and len(sys.argv) >= 4:
        endpoint_id = sys.argv[2]
        audio_file = sys.argv[3]
        model_id = sys.argv[4] if len(sys.argv) > 4 else "spb"
        result = test_endpoint(endpoint_id, audio_file, model_id)
        print(json.dumps(result, indent=2))
    else:
        print("Unknown command")
