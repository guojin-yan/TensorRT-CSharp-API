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

TensorRT, CUDA, and cuDNN runtime packages are versioned independently from this managed package. Publish the `CudaCudnn` and `TensorRt` component packages only when those dependency sets change. Publish the bridge component package when the local C ABI bridge changes. The original runtime package ID is kept as a lightweight collection package that pins one tested component-version combination.

Split runtime package names follow the component role:

- `<runtime-package-id>.Bridge`
- `<runtime-package-id>.CudaCudnn`
- `<runtime-package-id>.TensorRt`

Large stable dependency component packages may stay on GitHub Packages or GitHub Releases when they exceed nuget.org package-size limits. NVIDIA redistribution terms must be reviewed for the exact binaries being shipped.
