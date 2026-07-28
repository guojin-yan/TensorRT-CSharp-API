# CUDA Kernel Wrapper Roadmap

Custom CUDA preprocessing is a useful deployment scenario, but this repository should not publish a sample that asks users to call raw generated launch entry points directly.

The current managed CUDA surface is deployment-safe for device memory, pinned memory, streams, events, memory pools, graphs, copies, fills, and diagnostics. CUDA 12.9+ also has owner-safe `CudaKernelLibrary.Load(byte[])`, copied inventory, name-based queries, and typed `CudaKernelLibrary.Launch(...)`. A unified Driver module owner for CUDA 11.8/12.1 is still pending.

Raw CUDA launch entry points are present only as generated/internal native boundaries. They should not be used directly from a sample because argument packing, module ownership, stream ordering, and version-specific launch configuration need a higher-level C# API.

For the complete CUDA C++ source-to-PTX/CUBIN/LTO IR and owner-safe load/launch design, see the [CUDA Runtime Compilation (NVRTC) Roadmap](cuda-runtime-compilation-roadmap.md).

Use this runnable sample first:

```powershell
dotnet run --project .\samples\MultiStream
```

It validates the CUDA memory and stream primitives that a future GPU preprocessing demo will build on.

## Implemented and Planned API Shape

- Implemented owner-bound launch by name, typed kernel argument packing, and launch configuration validation for the CUDA 12.9+ runtime-library path.
- Add a unified `CudaModule` / `CudaKernel` Driver owner for older toolkits.
- Add version guards for CUDA launch APIs that differ across toolkit lines.
- Add a runnable normalization or NHWC-to-NCHW preprocessing sample only after the public API is safe.
