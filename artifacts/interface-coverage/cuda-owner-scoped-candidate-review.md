# CUDA Owner-Scoped Deferred Candidate Review

Generated: 2026-07-17

## Selected Batch

| Candidate | Decision | Safe boundary |
| --- | --- | --- |
| `cudaGraphAddMemsetNode` | implement | graph-owned node plus managed `CudaMemory` owner |
| `cudaGraphExecMemsetNodeSetParams` | implement | managed graph-exec and memory owners; scalar value/count |
| `cudaGraphDestroyNode` | implement | supplied graph membership is checked before removal |
| `cudaGraphKernelNodeGetParams` | implement | copy dimensions, shared memory and pointer-presence booleans only |
| `cudaGraphHostNodeGetParams` | implement | copy callback/user-data presence only |
| `cudaGraphMemAllocNodeGetParams` | implement | copy allocation scalars and pointer-presence booleans only |
| `cudaGraphMemFreeNodeGetParams` | implement | copy device-pointer presence only |
| `cudaGraphExternalSemaphoresSignalNodeGetParams` | implement | copy count and array-presence booleans only |
| `cudaGraphExternalSemaphoresWaitNodeGetParams` | implement | copy count and array-presence booleans only |
| `cudaStreamGetCaptureInfo_v3` | implement | copy status/id/graph-presence/dependency-count/edge-data-presence |
| `cudaStreamUpdateCaptureDependencies` | implement | immediate validated node-token array; no borrowed array escape |
| `cudaStreamBeginCaptureToGraph` | implement | owner-scoped session retains stream/graph wrappers and End validates the same graph handle |

## Continue Deferred

| Candidate | Reason |
| --- | --- |
| `cudaStreamUpdateCaptureDependencies_v2` | edge-data ABI/version path still needs a complete CUDA 12.3/12.9/13 model |
| `cudaGraphAddKernelNode` | requires kernel function and argument pointers |
| `cudaGraphAddHostNode` | requires a managed callback trampoline and user-data lifetime |
| `cudaGraphAddMemAllocNode` | creates a device pointer whose graph/runtime lifetime is not modeled |
| `cudaGraphAddMemFreeNode` | accepts a device pointer without an owner-bound allocation contract |
| external semaphore add/set APIs | require imported external handle ownership and parameter arrays |
| `cudaGraphNodeGetParams` | generic union contains callback, function, device and external-resource pointers |
| CUDA graph user-object APIs | require release callbacks and cross-language reference accounting |
| CUDA library/kernel/resource APIs | expose borrowed kernels, globals, resources or driver-owned handles |

Old deferred manifests remain in place. A selected real entry is allowed to supersede a deferred row only when coverage explicitly matches the real alias first and still records deferred history.
