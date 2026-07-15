#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-autolabsim-act}"
CUDA_FLAVOR="${1:-auto}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/environment-act.yml"

if ! command -v conda >/dev/null 2>&1; then
  echo "错误：未找到 conda。请先安装并初始化 Miniconda。"
  exit 1
fi

# Allow conda activate inside this non-interactive shell.
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda 环境 ${ENV_NAME} 已存在，跳过创建。"
else
  conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
fi

conda activate "${ENV_NAME}"

if [[ "${CUDA_FLAVOR}" == "auto" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_FLAVOR="cpu"
  else
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 || true)"
    DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 || true)"
    DRIVER_MAJOR="${DRIVER_VERSION%%.*}"

    if echo "${GPU_NAME}" | grep -Eqi 'RTX[[:space:]]*50|Blackwell'; then
      CUDA_FLAVOR="cu128"
    elif [[ "${DRIVER_MAJOR:-0}" =~ ^[0-9]+$ ]] && (( DRIVER_MAJOR >= 525 )); then
      CUDA_FLAVOR="cu126"
    else
      CUDA_FLAVOR="cu118"
    fi

    echo "检测到 GPU: ${GPU_NAME}"
    echo "检测到驱动: ${DRIVER_VERSION}"
  fi
fi

case "${CUDA_FLAVOR}" in
  cu118|cu126|cu128|cpu)
    ;;
  *)
    echo "错误：参数必须是 auto、cu118、cu126、cu128 或 cpu。"
    exit 2
    ;;
esac

echo "安装 PyTorch ${TORCH_VERSION} / torchvision ${TORCHVISION_VERSION} (${CUDA_FLAVOR})"
python -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url "https://download.pytorch.org/whl/${CUDA_FLAVOR}"

echo
echo "验证环境："
python - <<'PY'
import sys
import numpy as np
import cv2
import mujoco
import toppra
import torch
import torchvision

print("python      :", sys.version.split()[0])
print("numpy       :", np.__version__)
print("opencv      :", cv2.__version__)
print("mujoco      :", mujoco.__version__)
print("torch       :", torch.__version__)
print("torchvision :", torchvision.__version__)
print("cuda usable :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda runtime:", torch.version.cuda)
    print("gpu         :", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    print("cuda test   :", float(y.mean()))
PY

echo
echo "环境创建完成。以后使用："
echo "  conda activate ${ENV_NAME}"
echo "  cd ~/user_lcy/AutoLabSim"
