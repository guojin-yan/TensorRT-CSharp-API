# CUDA Runtime Compilation Sample

This sample exercises the owner-safe NVRTC compile surface with a virtual header, a name expression,
copied PTX metadata, an intentional compiler failure, and a best-effort `CudaKernelLibrary.Load` check.

```powershell
$env:JYPPX_NATIVE_BRIDGE_PATH = '<absolute path to jyppxtrtbridge.dll>'
$env:JYPPX_NVRTC_LIBRARY = '<absolute path to nvrtc64_120_0.dll>'
dotnet run --project .\samples\CudaRuntimeCompilation
```

The bridge-only package does not bundle NVRTC. Install a compatible CUDA Toolkit or set
`JYPPX_NVRTC_LIBRARY` to an exact library path. The NVRTC builtins library must remain next to the
selected compiler library.

This batch is compile-to-load evidence only. Until the public owner-bound named-kernel launch and
typed argument packing API lands, it is not kernel-runtime, GPU-readback, or numerical-correctness proof.
