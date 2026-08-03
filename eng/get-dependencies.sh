#!/usr/bin/env bash
set -euo pipefail

ARCHITECTURE="${ARCHITECTURE:-x64}"
TENSORRT_ROOT_INPUT="${1:-${JYPPX_TENSORRT_ROOT:-${TENSORRT_ROOT:-}}}"
CUDA_ROOT_INPUT="${2:-${JYPPX_CUDA_ROOT:-${CUDAToolkit_ROOT:-}}}"

resolve_first_existing_path() {
  for candidate in "$@"; do
    if [[ -n "${candidate}" && -e "${candidate}" ]]; then
      realpath "${candidate}"
      return 0
    fi
  done

  return 1
}

test_tensorrt_root() {
  local path="$1"
  [[ -n "${path}" && ( -f "${path}/include/NvInfer.h" || -f "${path}/include/NvInferVersion.h" ) ]]
}

test_cuda_root() {
  local path="$1"
  [[ -n "${path}" && -f "${path}/include/cuda_runtime.h" ]]
}

tensor_rt_candidates=(
  "${TENSORRT_ROOT_INPUT}"
  "/usr/local/TensorRT"
  "/opt/tensorrt"
  "/usr"
)

cuda_candidates=(
  "${CUDA_ROOT_INPUT}"
  "/usr/local/cuda"
  "/opt/cuda"
)

resolved_tensorrt_root=""
if candidate="$(resolve_first_existing_path "${tensor_rt_candidates[@]}" 2>/dev/null)"; then
  if test_tensorrt_root "${candidate}"; then
    resolved_tensorrt_root="${candidate}"
  fi
fi

resolved_cuda_root=""
if candidate="$(resolve_first_existing_path "${cuda_candidates[@]}" 2>/dev/null)"; then
  if test_cuda_root "${candidate}"; then
    resolved_cuda_root="${candidate}"
  fi
fi

echo "Architecture : ${ARCHITECTURE}"
if [[ -n "${resolved_tensorrt_root}" ]]; then
  echo "TensorRT     : FOUND"
  echo "  Root       : ${resolved_tensorrt_root}"
else
  echo "TensorRT     : MISSING"
  echo "  Root       : <not found>"
fi

if [[ -n "${resolved_cuda_root}" ]]; then
  echo "CUDA         : FOUND"
  echo "  Root       : ${resolved_cuda_root}"
else
  echo "CUDA         : MISSING"
  echo "  Root       : <not found>"
fi
