# CUDA Kernel Wrapper Roadmap

Custom CUDA preprocessing is a useful deployment scenario, but this repository should not publish a sample that asks users to call raw generated launch entry points directly.

The current managed CUDA surface is deployment-safe for device memory, pinned memory, streams, events, memory pools, graphs, copies, fills, and diagnostics. It does not yet expose a safe public abstraction for loading CUDA modules, packing kernel arguments, launching arbitrary kernels, and owning function/module lifetimes.

Raw CUDA launch entry points are present only as generated/internal native boundaries. They should not be used directly from a sample because argument packing, module ownership, stream ordering, and version-specific launch configuration need a higher-level C# API.

Use this runnable sample first:

```powershell
dotnet run --project .\samples\MultiStream
```

It validates the CUDA memory and stream primitives that a future GPU preprocessing demo will build on.

## Planned API Shape

- Add safe `CudaModule` and `CudaKernel` wrappers with explicit ownership and disposal.
- Add typed kernel argument packing instead of exposing raw pointer arrays.
- Add launch configuration validation for block/grid dimensions and shared memory.
- Add version guards for CUDA launch APIs that differ across toolkit lines.
- Add a runnable normalization or NHWC-to-NCHW preprocessing sample only after the public API is safe.
