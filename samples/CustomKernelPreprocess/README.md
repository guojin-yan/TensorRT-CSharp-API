# CustomKernelPreprocess

Status: roadmap sample, blocked on safe public CUDA kernel/module wrappers.

The current managed CUDA surface is deployment-safe for device memory, pinned memory, streams, events, memory pools, graphs, copies, fills, and diagnostics. It does not yet expose a safe public abstraction for loading CUDA modules, packing kernel arguments, launching arbitrary kernels, and owning function/module lifetimes.

Raw CUDA launch entry points are present only as generated/internal native boundaries. They should not be used directly from a sample because argument packing, module ownership, stream ordering, and version-specific launch configuration need a higher-level C# API.

Use these runnable samples first:

```powershell
dotnet .\samples\CudaSmokeRunner\bin\Debug\net8.0\CudaSmokeRunner.dll
dotnet .\samples\MultiStream\bin\Debug\net8.0\MultiStream.dll
```

They validate the CUDA memory and stream primitives that a future GPU preprocessing demo will build on.

## Roadmap

- Add safe `CudaModule` and `CudaKernel` wrappers with explicit ownership and disposal.
- Add typed kernel argument packing instead of exposing raw pointer arrays.
- Add launch configuration validation for block/grid dimensions and shared memory.
- Add version guards for CUDA launch APIs that differ across toolkit lines.
- Add a sample kernel for normalization or NHWC-to-NCHW conversion after the public API is safe.
