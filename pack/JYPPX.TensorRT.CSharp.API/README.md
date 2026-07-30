# JYPPX.TensorRT.CSharp.API

This package contains the managed TensorRtSharp4.0 assemblies:

- `JYPPX.TensorRtSharp`
- `JYPPX.CudaSharp`
- `JYPPX.Shared`

The managed package does not contain NVIDIA runtime binaries. Install CUDA, TensorRT, cuDNN, and optional NVRTC from NVIDIA for the version line used by the application.

Install the matching project-owned bridge package for native ABI access. Bridge package IDs encode their build matrix and end in `.Bridge`, for example:

```text
JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda12.9.cudnn9.22.Bridge
```

The bridge package contains only `jyppxtrtbridge.dll` on Windows or `libjyppxtrtbridge.so` on Linux. It does not contain CUDA, cuDNN, TensorRT, parser, plugin, builder-resource, NVRTC, or NVRTC builtins libraries.

The enforced publication policy is `pack/external-vendor-runtime-policy.json` in the source repository.
