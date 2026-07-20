# CUDA Graph Memory Allocation Owner Safety

CUDA graph memory-allocation nodes return a device address through native parameters. The managed
API intentionally does not expose that address. `CudaGraphMemoryAllocation` is graph-bound bridge
metadata that can only be consumed through its owning `CudaGraph`.

## Build a Pointer-Free Allocation Flow

```csharp
const int byteCount = 64;

using CudaStream stream = new CudaStream(CudaStreamCreationFlags.NonBlocking);
using CudaPinnedMemory output = new CudaPinnedMemory(byteCount);
using CudaGraph graph = CudaGraph.Create();
using CudaGraphMemoryAllocation allocation =
    graph.AddMemoryAllocationNode(byteCount, CudaDevice.Current);

CudaGraphNode fill = graph.AddMemsetNode(allocation, 0x6B, byteCount);
CudaGraphNode copy = graph.AddDeviceToHostMemcpyNodeAfter(
    fill,
    output,
    allocation,
    byteCount);
graph.AddMemoryFreeNode(allocation, copy);

using CudaGraphExec executable = graph.Instantiate();
executable.Launch(stream);
stream.Synchronize();
byte[] bytes = output.ToArray(byteCount);
```

The bridge keeps the device address native, automatically includes the allocation node as a
dependency of every consumer, and allows one successful matching free node. The free node should
depend on the final allocation consumer.

## Lifetime Rules

- Use an allocation only with the graph that created it.
- Add all memset and copy consumers before adding the free node.
- Add the matching free node exactly once.
- Dispose the allocation metadata before disposing the graph.
- Keep pinned host memory alive while the executable graph can write to it.

Cross-graph use and a second free fail before a native call. Disposing a graph while an allocation
wrapper remains active also fails. Generic node removal cannot remove memory-allocation or
memory-free nodes because that would bypass these invariants.

## Evidence Boundary

`CudaGraphSmokeRunner` verifies a 64-byte `0x6B` allocation/copy/free round trip on a compatible
local CUDA 12.9 host. That ProjectReference smoke is local runtime evidence only. It is not a clean
public package-consumer run, post-publish evidence, or authorization to publish packages.
