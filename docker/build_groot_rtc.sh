#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_NAME="${NEWTON_GROOT_RTC_IMAGE:-newton-direct-gpu-groot:latest}"
BASE_IMAGE="${NEWTON_GROOT_RTC_BASE_IMAGE:-newton-direct-gpu:latest}"

cd "${REPO_DIR}"
docker build \
    -f docker/Dockerfile.groot_rtc \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    -t "${IMAGE_NAME}" \
    .

docker run --rm --gpus all "${IMAGE_NAME}"
