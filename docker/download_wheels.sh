#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEELHOUSE="${WHEELHOUSE:-${SCRIPT_DIR}/wheelhouse}"
WARP_WHEEL="warp_lang-1.16.0.dev20260707-py3-none-manylinux_2_28_x86_64.whl"
WARP_URL="${WARP_URL:-https://pypi.nvidia.com/warp-lang/${WARP_WHEEL}}"
WARP_SIZE=162876591
WARP_PATH="${WHEELHOUSE}/${WARP_WHEEL}"

mkdir -p "${WHEELHOUSE}"

wheel_is_complete() {
    [[ -f "${WARP_PATH}" ]] || return 1
    [[ ! -f "${WARP_PATH}.aria2" ]] || return 1
    python3 -m zipfile -t "${WARP_PATH}" >/dev/null 2>&1
}

repair_incomplete_wheel() {
    if [[ -f "${WARP_PATH}" && ! -f "${WARP_PATH}.aria2" ]] && ! python3 -m zipfile -t "${WARP_PATH}" >/dev/null 2>&1; then
        rm -f "${WARP_PATH}"
    fi
}

until wheel_is_complete; do
    repair_incomplete_wheel
    if command -v aria2c >/dev/null 2>&1; then
        aria2c \
            --allow-overwrite=true \
            --connect-timeout=30 \
            --continue=true \
            --dir="${WHEELHOUSE}" \
            --lowest-speed-limit=1K \
            --max-connection-per-server=8 \
            --max-tries=0 \
            --min-split-size=1M \
            --out="${WARP_WHEEL}" \
            --retry-wait=2 \
            --split=8 \
            --timeout=30 \
            "${WARP_URL}" || true
    else
        curl -fL --retry 50 --retry-delay 2 --connect-timeout 30 \
            --speed-limit 1024 --speed-time 30 -C - \
            -o "${WARP_PATH}" \
            "${WARP_URL}" || true
    fi
    sleep 2
done

stat -c "downloaded %n (%s bytes, expected ${WARP_SIZE})" "${WARP_PATH}"
