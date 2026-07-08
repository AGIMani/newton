#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${REPO_DIR}/.." && pwd)"
HOST_HOME="${HOST_HOME:-${HOME}}"
CONTAINER_HOME="${CONTAINER_HOME:-${HOST_HOME}}"
IMAGE_NAME="${IMAGE_NAME:-newton:latest}"
DISPLAY_ARG="${DISPLAY:-:0}"
MODEL_PATH="${MODEL_PATH:-${CONTAINER_HOME}/.cache/teleop_stack/vosk/vosk-model-small-cn-0.22}"
ISAAC_TELEOP_ROOT="${ISAAC_TELEOP_ROOT:-${PROJECT_DIR}/IsaacTeleop}"
IMPORTED_WEBXR_DIR="${IMPORTED_WEBXR_DIR:-${CONTAINER_HOME}/.cache/teleop_stack/cloudxr_web_client_remote/webxr_client}"
CAMERA_STREAMER_LITE_IMAGE="${NEWTON_CAMERA_STREAMER_LITE_IMAGE:-harness-camera-streamer-lite:latest}"

if [[ ! -e /dev/video44 ]]; then
    sudo modprobe v4l2loopback video_nr=44 card_label=teleop_sim_screen exclusive_caps=1 max_buffers=2 max_width=1920 max_height=1080
fi

if command -v xhost >/dev/null 2>&1; then
    xhost +local:root >/dev/null 2>&1 || true
fi

docker_args=(
    --rm
    --name newton-vr-stack
    --gpus all
    --runtime=nvidia
    --privileged
    --network=host
    --ipc=host
    -e "DISPLAY=${DISPLAY_ARG}"
    -e "HOME=${CONTAINER_HOME}"
    -e "USER=${USER:-user}"
    -e "REPO_DIR=${REPO_DIR}"
    -e "PYTHONPATH=${REPO_DIR}"
    -e "PYTHON_BIN=/workspace/newton/.venv/bin/python"
    -e "SCENE_PYTHON_BIN=/workspace/newton/.venv/bin/python"
    -e "TELEOP_PYTHON_BIN=/workspace/newton/.venv/bin/python"
    -e "ISAAC_TELEOP_ROOT=${ISAAC_TELEOP_ROOT}"
    -e "MODEL_PATH=${MODEL_PATH}"
    -e "IMPORTED_IMAGE=cloudxr-web-app:latest"
    -e "IMPORTED_WEBXR_DIR=${IMPORTED_WEBXR_DIR}"
    -e "NEWTON_CAMERA_STREAMER_LITE_IMAGE=${CAMERA_STREAMER_LITE_IMAGE}"
    -e "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-graphics,video,compute,utility,display}"
    -v "${PROJECT_DIR}:${PROJECT_DIR}:rw"
    -v "${HOST_HOME}/.cache:${CONTAINER_HOME}/.cache:rw"
    -v "${HOST_HOME}/.cloudxr:${CONTAINER_HOME}/.cloudxr:rw"
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw
    -v /dev:/dev
    -v /run/udev:/run/udev:rw
    -v /var/run/docker.sock:/var/run/docker.sock
)

if [[ -x /usr/bin/docker ]]; then
    docker_args+=(-v /usr/bin/docker:/usr/bin/docker:ro)
fi

if [[ -t 0 && -t 1 ]]; then
    docker_args+=(-it)
fi

if [[ -n "${XAUTHORITY:-}" && -f "${XAUTHORITY}" ]]; then
    docker_args+=(-e "XAUTHORITY=${XAUTHORITY}" -v "${XAUTHORITY}:${XAUTHORITY}:ro")
elif [[ -f "${HOST_HOME}/.Xauthority" ]]; then
    docker_args+=(-e "XAUTHORITY=${CONTAINER_HOME}/.Xauthority")
fi

if [[ -d /usr/share/vulkan/icd.d ]]; then
    docker_args+=(-v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro)
fi

exec docker run "${docker_args[@]}" "${IMAGE_NAME}" \
    bash -lc 'cd "${REPO_DIR}" && exec scripts/run_newton_vr_prereqs.sh --display "${DISPLAY}" "$@"' \
    bash "$@"
