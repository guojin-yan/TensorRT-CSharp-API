# CUDA Graph Memory Allocation Owner-Safe 封装

CUDA graph memory-allocation node 会通过原生参数返回 device address。托管 API 不公开这个
地址，而是返回与创建它的 `CudaGraph` 绑定的 `CudaGraphMemoryAllocation` bridge 元数据；
调用者只能通过所属 graph 使用它。

## 构建无指针分配流程

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

bridge 将 device address 始终保留在 native 内部，并自动让每个消费节点依赖 allocation
node。每个 allocation 只能成功添加一个匹配的 free node；free node 应依赖最后一个消费节点。

## 生命周期规则

- allocation 只能交给创建它的 graph。
- 添加 free node 前先完成全部 memset 和 copy consumer。
- 匹配的 free node 只能添加一次。
- 先释放 allocation 元数据 wrapper，再释放 graph。
- executable graph 仍可能写入 pinned host memory 时，必须保持 host owner 存活。

跨 graph 使用和二次 free 会在 native 调用前失败；allocation wrapper 活跃时释放 graph 也会
失败。通用 node removal 明确拒绝 memory-allocation 和 memory-free node，防止绕过这些约束。

## 证据边界

`CudaGraphSmokeRunner` 已在兼容 CUDA 12.9 本机完成 64 字节 `0x6B` 的
allocate/memset/copy/free round trip。这是 ProjectReference 本地 runtime evidence，不是
clean public package consumer、post-publish 或公开发布许可。
