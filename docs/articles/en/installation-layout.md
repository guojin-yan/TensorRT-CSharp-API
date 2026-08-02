# Installation Layout

## Recommended local Windows layout

CUDA root:

- standard NVIDIA Toolkit install under `%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA`

Example CUDA folders validated in this repository:

- `v11.6`
- `v11.8`
- `v12.1`
- `v12.9`
- `v13.2`

TensorRT root:

- `<repo-root>\third_party\nvidia`

Example TensorRT package folders:

- `TensorRT-10.11.0.33-cuda 11.8`
- `TensorRT-10.11.0.33-cuda 12.9`
- `TensorRT-8.6.1.6-cuda 11.8`
- `TensorRT-8.6.1.6-cuda 12.0`

## Matching rules

- `TensorRT-10.11.0.33-cuda 11.8` matches CUDA `11.0` to `11.8`
- `TensorRT-10.11.0.33-cuda 12.9` matches CUDA `12.0` to `12.9`
- `TensorRT-8.6.1.6-cuda 11.8` matches CUDA `11.0` to `11.8`
- `TensorRT-8.6.1.6-cuda 12.0` matches CUDA `12.0` to `12.1`

## Important limitation

When multiple CUDA and TensorRT versions exist on the same machine, build and packaging steps must explicitly select the intended combination. Do not rely on implicit discovery alone.

Machine-specific Windows roots should be stored in:

- `pack/runtime/runtime-packages.local.json`

This file is ignored by Git. Use `pack/runtime/runtime-packages.local.example.json` as the starting point. The public `runtime-packages.manifest.json` must not contain machine-specific local drive roots.

Managed runtime loading is production-first:

- normal probing checks the application base directory and `runtimes/<rid>/native`
- explicit native bridge loading uses `JYPPX_NATIVE_BRIDGE_PATH`
- explicit vendor roots use `JYPPX_TENSORRT_ROOT` and `JYPPX_CUDA_ROOT`
- local development scanning of `build-out` and `third_party` requires `JYPPX_ENABLE_DEVELOPMENT_PROBING=1`

The resolver normalizes and moves preferred dependency directories ahead of automatically discovered development candidates, even when a selected directory already appeared later in `PATH`. On hosts with multiple CUDA installations, verify the absolute path of the `cudart64_*` module loaded by the process; a correct environment-variable value alone is not loaded-module evidence.

CMake presets now refresh selected CUDA toolkit cache variables when a CUDA root is resolved, so stale `CUDAToolkit_NVCC_EXECUTABLE` values from another CUDA line should not survive reconfiguration.

## Validated combinations

- TensorRT package: `TensorRT-10.11.0.33-cuda 11.8`
- CUDA root: `%CUDA_PATH_V11_8%`
- CMake preset: `win-x64-trt10-cuda11-release`

Additional Windows combinations validated in this repository:

- `TensorRT-10.11.0.33-cuda 12.9` + `CUDA v12.9`
- `TensorRT-8.6.1.6-cuda 11.8` + `CUDA v11.8`
- `TensorRT-8.6.1.6-cuda 12.0` + `CUDA v12.1`
- `TensorRT-11.0.0.114-cuda 12.9` + `CUDA v12.9`

## TensorRT 11 status

TensorRT 11 is now part of the Windows build and runtime matrix. The `trt11.0-cuda12.9-cudnn9.22` path has native adapter, runtime package, package consumer, and smoke validation. The `trt11.0-cuda13.2-cudnn9.22` path compiles, but runtime smoke remains pending until a CUDA 13-capable driver/runtime environment is available.

## Linux expectation

The repository assumes Linux runtime packaging will be driven from explicit roots on self-hosted Linux x64 runners rather than auto-discovery from a Windows development machine.
