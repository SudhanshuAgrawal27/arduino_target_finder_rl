#!/usr/bin/env bash
set -euo pipefail

IMAGE="tf-512-gpu"
MAX_JOBS="${MAX_JOBS:-4}"

echo "Building $IMAGE (MAX_JOBS=$MAX_JOBS)..."

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

docker build \
    --build-arg MAX_JOBS="$MAX_JOBS" \
    --tag "$IMAGE" \
    --file "$REPO_ROOT/docker/Dockerfile" \
    "$REPO_ROOT"

echo ""
echo "Build complete."
docker images "$IMAGE" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
