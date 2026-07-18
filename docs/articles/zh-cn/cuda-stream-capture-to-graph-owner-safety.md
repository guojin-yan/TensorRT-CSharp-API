# CUDA Stream Capture To Graph 的 Owner 安全边界

`cudaStreamBeginCaptureToGraph` 的难点不是把一个函数签名翻译成 P/Invoke，
而是解释 capture 期间谁拥有 stream、谁拥有 graph，以及 `cudaStreamEndCapture`
返回的 graph 是否会制造第二个托管 owner。

## 选择范围

该入口从 CUDA 12.3 开始可用。本项目用独立的 `CUDART_VERSION >= 12030`
guard 连接 CUDA 12.3、12.9 和 13.2；CUDA 11.x 与 CUDA 12.1 仍保持
deferred，不通过生成器或 managed build 假装存在。

## Session 设计

```csharp
using CudaStreamCaptureToGraphSession session = stream.BeginCaptureToGraph(
    graph,
    Array.Empty<CudaGraphNodeDependency>(),
    CudaStreamCaptureMode.Relaxed);

// Enqueue work on stream while the session is active.
session.End();
```

开始时 session 增加 stream 和 graph 的 capture-use 计数。只要计数非零，
两个 wrapper 的 `Dispose()` 都会拒绝执行。结束时 native bridge 调用
`cudaStreamEndCapture`，验证 CUDA 返回的句柄仍然是传入的 graph，再释放计数。
因此 managed 层不会因为 End 产生第二个 graph wrapper。

dependency node token 和 edge data 只在同步 native call 内复制或 pin；它们不
会保存调用方数组地址。native helper 负责把 C++ 异常、分配失败和 Windows
SEH 转成 bridge status，异常不会跨 ABI。

## 证据与边界

TRT10/CUDA12.9 的 `CudaGraphSmokeRunner` 输出了 `ToGraph=True Nodes=1`，并
完成了后续 graph round trip。这个结果是当前兼容主机上的 source-tree
compatible-host smoke；它不是公开 NuGet 的 clean package-consumer runtime
proof，也不是 post-publish proof 或 release close approval。

对应候选审计见
`artifacts/interface-coverage/cuda-stream-capture-to-graph-candidate-audit.md`。

## 相邻的 Conditional Graph 边界

后续 conditional graph uplift 使用同一套 owner-first 原则，但不把 CUDA body
graph 句柄公开给托管层。`CudaGraphConditionalHandle` 和
`CudaGraphConditionalNode` 是 bridge-owned metadata wrapper；C# 只读取 body
数量、root/edge topology，并能向指定 body 添加 bridge-owned empty node。父图在
handle 或 node metadata wrapper 存活时拒绝 Dispose，generic node destroy 也不会
误接管 conditional node。

CUDA 12.9 smoke 已完成 IF 条件节点、两个 body、default value、instantiate/launch
和主动 owner 释放拒绝。CUDA 13.2 的本机 runtime smoke 在 CUDA error 35 处停止，
所以当前结果仍属于 source-tree/compatible-host 边界；body capture、kernel/raw
pointer node、callback 和外部资源 ownership 继续 deferred。
