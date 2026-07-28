# CUDA Runtime Compilation Sample

This sample exercises the owner-safe NVRTC compile surface with a virtual header, a name expression,
copied PTX metadata, an intentional compiler failure, and an owner-bound vector-add launch/readback check.

```powershell
$env:JYPPX_NATIVE_BRIDGE_PATH = '<absolute path to jyppxtrtbridge.dll>'
$env:JYPPX_NVRTC_LIBRARY = '<absolute path to nvrtc64_120_0.dll>'
dotnet run --project .\samples\CudaRuntimeCompilation
```

The bridge-only package does not bundle NVRTC. Install a compatible CUDA Toolkit or set
`JYPPX_NVRTC_LIBRARY` to an exact library path. The NVRTC builtins library must remain next to the
selected compiler library.

When the generated PTX can be loaded by the current CUDA runtime/driver, the sample packs three
owner-bound `CudaMemory` arguments (two inputs and one output) plus an `Int32` scalar, launches `vector_add`,
synchronizes the `CudaKernelLaunch` completion owner, reads back 257 floats, and validates every value. The
sample disposes the library, stream, and input-memory owners before synchronization to prove lease retention.
A Toolkit whose PTX version is newer than the current driver remains compile-only and is reported separately.

This is local-toolkit kernel runtime proof, not package-consumer, public-package, Linux, or post-publish proof.
