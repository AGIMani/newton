#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_NAME="${NEWTON_DIRECT_GPU_IMAGE:-newton-direct-gpu:latest}"
BASE_IMAGE="${NEWTON_DIRECT_GPU_BASE_IMAGE:-harness-camera-streamer-lite:latest}"
UBUNTU_MIRROR="${UBUNTU_MIRROR:-http://mirrors.tuna.tsinghua.edu.cn/ubuntu}"
UBUNTU_SUITE="${UBUNTU_SUITE:-noble}"

cd "${REPO_DIR}"
docker build \
    -f docker/Dockerfile.direct_gpu \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "UBUNTU_MIRROR=${UBUNTU_MIRROR}" \
    --build-arg "UBUNTU_SUITE=${UBUNTU_SUITE}" \
    -t "${IMAGE_NAME}" \
    .

docker run --rm --gpus all "${IMAGE_NAME}"
