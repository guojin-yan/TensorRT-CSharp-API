# split runtime packages

This directory contains split runtime component packages for Windows runtime combinations that are too large to publish as a single nupkg.

Current goals:

- keep the original runtime package ID available through a lightweight meta package
- move native binaries into component packages that each stay below GitHub package and release size limits
- validate that consumers can install the meta package and still receive the full native runtime layout

Current split strategies:

- TensorRT 10: historical two-part prototype (`Core` + `Extensions`)
- TensorRT 11: publish-oriented component split for `win-x64-trt11.0-cuda12.9-cudnn9.22`
  - `Bridge`
  - `CudaCudnn`
  - `TensorRtRuntime`
  - `TensorRtBuilder.Sm75Sm86`
  - `TensorRtBuilder.Sm89Sm90`
  - `TensorRtBuilder.Sm100Sm120Ptx`

These packages still require:

- NVIDIA redistribution review
- package-size review
- consumer validation proof for the split package set

Supporting scripts:

- `eng/Validate-SplitDeliveryPrototype.ps1`
- `eng/Collect-SplitRuntimeAssets.ps1`
- `eng/Invoke-LocalSplitRuntimePackage.ps1`
