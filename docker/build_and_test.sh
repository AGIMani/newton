#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-newton:latest}"
TEST_CMD="${TEST_CMD:-python -m newton.tests -k test_basic.example_basic_shapes}"
BASE_IMAGE="${BASE_IMAGE:-newton-ubuntu:24.04}"

cd "${REPO_DIR}"

docker build --build-arg "BASE_IMAGE=${BASE_IMAGE}" -f docker/Dockerfile -t "${IMAGE_NAME}" .
docker run --rm --gpus all "${IMAGE_NAME}" bash -lc "${TEST_CMD}"
