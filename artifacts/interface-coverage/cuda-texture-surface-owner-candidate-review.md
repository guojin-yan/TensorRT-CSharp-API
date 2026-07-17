# CUDA Texture and Surface Owner Candidate Review

Generated: 2026-07-17

## Selected Batch

| Candidate | Decision | Safe boundary |
| --- | --- | --- |
| `cudaCreateSurfaceObject` / `cudaDestroySurfaceObject` | implement | bridge-owned object with a managed `CudaArray` SafeHandle lease |
| `cudaGetSurfaceObjectResourceDesc` | implement | copied resource kind, sizes and pointer-presence booleans only |
| `cudaCreateTextureObject` / `cudaDestroyTextureObject` | implement | bridge-owned object with typed scalar sampling descriptor and array lease |
| `cudaCreateTextureObject_v2` | implement | explicit CUDA 11.8-only route; other versions return diagnosed `NotSupported` |
| `cudaGetTextureObjectResourceDesc` | implement | copied resource kind, sizes and pointer-presence booleans only |
| `cudaGetTextureObjectTextureDesc` | implement | copied enum, bool, float and color scalars |
| `cudaGetTextureObjectTextureDesc_v2` | implement | explicit CUDA 11.8-only copied descriptor route |
| `cudaGetTextureObjectResourceViewDesc` | implement | copied format, extent, mip and layer scalars, or explicit `IsSpecified=false` when creation supplied no view |

## Continue Deferred

| Candidate | Reason |
| --- | --- |
| linear and pitch2D texture creation | requires a device-pointer lifetime lease beyond the existing array owner model |
| mipmapped texture creation | needs a SafeHandle lease route for `CudaMipmappedArray` and validated resource-view creation semantics |
| custom resource-view creation | format reinterpretation constraints need full array format and layered/mipmap validation |
| external memory/semaphore objects | imported handle ownership and asynchronous use are not modeled |
| CUDA graphics interop resources | mapping state and foreign graphics API ownership remain external |
| symbol/library/kernel handles | expose borrowed symbol addresses or driver-owned objects |

Old deferred manifests remain in place. Coverage may prefer the real owner-safe entries only while retaining deferred history for every promoted official function.
