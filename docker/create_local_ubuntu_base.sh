#!/usr/bin/env bash

set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-newton-ubuntu:24.04}"
ROOTFS_DIR="${ROOTFS_DIR:-/tmp/newton-ubuntu-rootfs}"
TARBALL="${TARBALL:-/tmp/newton-ubuntu-rootfs.tar}"
MIRROR="${MIRROR:-http://mirrors.tuna.tsinghua.edu.cn/ubuntu}"
SUITE="${SUITE:-noble}"
COMPONENTS="${COMPONENTS:-main,universe}"

sudo_cmd() {
    if [[ "${EUID}" -eq 0 ]]; then
        "$@"
    elif [[ -n "${SUDO_PASSWORD:-}" ]]; then
        printf '%s\n' "${SUDO_PASSWORD}" | sudo -S -p '' "$@"
    else
        sudo "$@"
    fi
}

if docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    exit 0
fi

if ! command -v debootstrap >/dev/null 2>&1; then
    sudo_cmd apt-get update
    sudo_cmd apt-get install -y debootstrap
fi

sudo_cmd rm -rf "${ROOTFS_DIR}" "${TARBALL}"
sudo_cmd debootstrap --variant=minbase --components="${COMPONENTS}" --arch=amd64 "${SUITE}" "${ROOTFS_DIR}" "${MIRROR}"
sudo_cmd tar -C "${ROOTFS_DIR}" -cpf "${TARBALL}" .
docker import "${TARBALL}" "${IMAGE_NAME}"
sudo_cmd rm -rf "${ROOTFS_DIR}" "${TARBALL}"
