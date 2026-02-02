#!/bin/bash
# Build latest version of vllm and push vllm-gaudi image to registry.rc.asu.edu

set -e

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

REGISTRY="registry.rc.asu.edu"
IMAGE_NAME="vllm-gaudi"
TAG="latest"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile.vllm.gaudi"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building ${IMAGE_NAME}:${TAG}...${NC}"

# Build the image
podman build \
    -f "${DOCKERFILE}" \
    -t "${REGISTRY}/${IMAGE_NAME}:${TAG}" \
    -t "${REGISTRY}/${IMAGE_NAME}:$(date +%Y%m%d)" \
    .

echo -e "${GREEN}Build complete!${NC}"
echo -e "${YELLOW}Pushing to ${REGISTRY}...${NC}"

# Push the image (registry.rc.asu.edu should work now with nginx)
podman push "${REGISTRY}/${IMAGE_NAME}:${TAG}"
podman push "${REGISTRY}/${IMAGE_NAME}:$(date +%Y%m%d)"

echo -e "${GREEN}Push complete!${NC}"
echo -e "${GREEN}Image available at: ${REGISTRY}/${IMAGE_NAME}:${TAG}${NC}"

# Verify the image was pushed
echo -e "${YELLOW}Verifying image in registry...${NC}"
curl -s "https://${REGISTRY}/v2/${IMAGE_NAME}/tags/list" | jq '.' || echo "Note: jq not installed, but image should be pushed"
