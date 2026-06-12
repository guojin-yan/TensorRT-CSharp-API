# JYPPX.TensorRT.CSharp.API

This package contains the managed TensorRtSharp4.0 assemblies:

- `JYPPX.TensorRtSharp`
- `JYPPX.CudaSharp`
- `JYPPX.Shared`

Install one matching runtime package for native assets. Runtime package names include the target TensorRT, CUDA, and cuDNN major.minor versions, for example:

- `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt8.6.cuda11.8.cudnn8.9`
- `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt8.6.cuda12.1.cudnn8.9`
- `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda11.8.cudnn8.9`
- `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22`
- `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda12.9.cudnn9.22`
- `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda13.2.cudnn9.22`

The package metadata keeps the full vendor versions, such as TensorRT `10.11.0.33` and cuDNN `8.9.7.29`. CUDA `12.9` is installed locally and must be used for `cuda12.9` runtime packages; the earlier CUDA `12.3` fallback is retired and must not be used for locally validated packages.

TensorRT 10 and TensorRT 11 runtime packages are large and are currently treated as private-feed or split-delivery candidates until NVIDIA redistribution and package-size review are complete.

Prototype split-delivery package names are reserved for TensorRT 10:

- `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda11.8.cudnn8.9.Core`
- `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda11.8.cudnn8.9.Extensions`
- `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Core`
- `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Extensions`

The `Core` package is intended to carry the bridge, CUDA runtime, and core TensorRT runtime libraries. The `Extensions` package is intended to carry builder resources, plugin libraries, and parser libraries. These split packages are design-only prototypes until split consumer validation and NVIDIA redistribution review are complete.
