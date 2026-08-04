# API Reference

The generated managed API reference is built by DocFX from the current C# projects under `src/`.

Primary entry points:

- [JYPPX.TensorRtSharp](../../api/JYPPX.TensorRtSharp.yml)
- [JYPPX.CudaSharp](../../api/JYPPX.CudaSharp.yml)
- [JYPPX.TensorRtSharp.Shared](../../api/JYPPX.TensorRtSharp.Shared.yml)

The public C# surface uses only two product namespace roots: TensorRT and shared bridge types live under `JYPPX.TensorRtSharp`, while CUDA types live under `JYPPX.CudaSharp`. `JYPPX.Shared.dll` remains an internal assembly split, but its types use the `JYPPX.TensorRtSharp.Shared` namespace and do not introduce a third public root.

Use this page as the stable documentation entry when you want to browse:

- TensorRT managed wrappers
- CUDA managed wrappers
- shared bridge/runtime helper types
