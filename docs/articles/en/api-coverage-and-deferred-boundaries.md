# API Coverage and Deferred Boundaries

Status as of 2026-06-12:

- TensorRT coverage: no missing rows in `artifacts/interface-coverage/tensorrt-interface-coverage.csv`.
- CUDA runtime coverage: no missing rows in `artifacts/interface-coverage/cuda-runtime-interface-coverage.csv`.
- Manifest inventory: 3271 API records across 102 manifests.
- Coverage summary: `artifacts/interface-coverage/interface-coverage-summary.md`.

The coverage matrix proves that each scanned official interface has a corresponding manifest/native source entry. It does not mean every official function is promoted to a high-level managed API. TensorRtSharp intentionally separates three levels of support.

## Support Levels

### Safe Managed Wrapper

Use this level for deployment-critical APIs. The C ABI validates inputs, maps native failures into `JYPPX_StatusCode`, and the managed layer exposes typed objects such as `TensorRtEngine`, `TensorRtExecutionContext`, `CudaMemory`, `CudaStream`, `CudaGraph`, or `CudaMemoryPool`.

Examples:

- TensorRT build, serialize, deserialize, engine metadata, tensor binding, and enqueue paths.
- CUDA device, memory, pinned memory, managed memory, pitched memory, streams, events, graphs, and memory pools.
- Parser diagnostics, refitter metadata, and deployment snapshots where the underlying ABI is stable enough.

### Native Boundary

Use this level when the native bridge has a safe C ABI, but the public C# wrapper is intentionally narrow or internal. This is useful for generated interop, diagnostics, and future wrapper lifting.

### Deferred Boundary

Use this level when an API exists in the scanned official headers but cannot be safely exposed yet. Deferred boundaries are still manifest/native entries, and each native stub records a reason before returning `JYPPX_STATUS_NOT_IMPLEMENTED`.

Common deferred reasons:

- Callback lifetime or destructor ownership is not modeled.
- Raw driver entrypoint or export-table pointers would escape the safe API surface.
- CUDA external memory, semaphore, texture, surface, IPC, or library descriptors need dedicated ownership structs.
- CUDA graph/user-object APIs require non-owning handle semantics that are not yet public.
- Batch pointer arrays need typed validation before they can be safely passed from .NET.

## Current Deferred Focus Areas

High-value candidates for future promotion:

- `cudaMemRangeGetAttribute` and `cudaMemRangeGetAttributes` with typed output buffers.
- `cudaSetValidDevices` and `cudaInitDevice` with strict validation.
- `cudaMemAdvise_v2` and `cudaMemPrefetchAsync_v2` with a public `CudaMemLocation` model.
- Texture/surface object descriptors where the resource lifetime can be represented without raw pointer leakage.
- External memory/semaphore wrappers after handle ownership rules are explicit.

Keep deferred for now:

- Driver export table and entrypoint raw pointers.
- CUDA library JIT option arrays.
- CUDA logs callbacks.
- Green context and execution context resource descriptors.
- User-object destructor callbacks.

## Verification Commands

Run these gates after changing manifests, native wrappers, generated interop, or sample code:

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
powershell -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
powershell -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
dotnet build .\TensorRtSharp.sln -c Debug --no-restore
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
```

The expected invariant is simple: TensorRT missing rows remain `0`, CUDA missing rows remain `0`, and new public wrappers do not expose unmanaged ownership details to ordinary users.
