#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUNTIME_KEY="${1:?runtime package key is required}"
TENSORRT_ROOT="${2:?TensorRT root is required}"
CUDA_ROOT="${3:?CUDA root is required}"

if command -v pwsh >/dev/null 2>&1; then
  pwsh -File "${REPO_ROOT}/eng/Validate-LinuxRuntimeInputs.ps1" -RuntimePackageKey "${RUNTIME_KEY}" -TensorRtRoot "${TENSORRT_ROOT}" -CudaRoot "${CUDA_ROOT}" -RepositoryRoot "${REPO_ROOT}"
else
  echo "pwsh is required to run Validate-LinuxRuntimeInputs.ps1 on this platform." >&2
  exit 1
fi

