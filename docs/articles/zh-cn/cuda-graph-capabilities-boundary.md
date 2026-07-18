# CUDA Graph 当前能力与边界

本文介绍 TensorRtSharp4.0 当前 CUDA Graph wrapper 覆盖的能力，以及哪些行为仍应通过 smoke 和环境证据来判断。

## 当前可用能力

`CudaGraphSmokeRunner` 覆盖了以下能力：

- stream capture 到 graph。
- `CudaStream.BeginCaptureToGraph` 将 capture 写入已有 `CudaGraph`，并通过
  `CudaStreamCaptureToGraphSession` 保持 stream/graph owner 在 End 前有效。
- graph instantiate、upload、launch。
- graph clone。
- node/root/edge 查询。
- dependency 和 dependent 查询。
- edge data 查询、remove、restore。
- debug DOT 导出。
- event record/wait nodes。
- 1D memcpy nodes。
- memset node 参数读取和更新。
- graph memory info、trim 和 high watermark reset。
- device graph memory summary：`CudaDevice.GetGraphMemorySummary(int)`、`CudaDevice.CurrentGraphMemorySummary` 与 `CudaDeviceGraphMemoryInfo.ToSummary()` 将 `UsedMemoryCurrent`、`UsedMemoryHigh`、`ReservedMemoryCurrent`、`ReservedMemoryHigh` 四个 `cudaDeviceGetGraphMemAttribute` copied scalar 统一为 `CudaDeviceGraphMemorySummary`。

这些 API 的核心目标是让用户能在 C# 中描述、检查和执行常见 graph 拓扑，而不需要直接处理 CUDA driver/runtime 的裸句柄。

## Smoke Runner

运行：

```powershell
dotnet run --project .\smoke\CudaGraphSmokeRunner\CudaGraphSmokeRunner.csproj
```

成功时预期看到：

```text
CudaGraphCaptureRoundTrip=True
CudaGraphCapturedTopology ...
CudaGraphManualTopology ...
CudaGraphMemory ...
```

如果 CUDA runtime 不可用，runner 会输出：

```text
Skipped=True Reason=CudaException:...
```

这类 skip 是环境证据，不是 API proof，也不是接口缺失。

## 能力图

```mermaid
flowchart TD
  A["CudaStream.BeginCapture"] --> B["memory fill/copy async"]
  B --> C["CudaStream.EndCapture"]
  C --> D["CudaGraph"]
  D --> E["Clone / Query Nodes / Query Edges"]
  D --> F["Instantiate"]
  F --> G["Upload + Launch"]
  G --> H["Pinned destination round-trip check"]
```

已有 graph 的 capture 使用另一条生命周期路径：

```mermaid
sequenceDiagram
  participant S as CudaStream
  participant G as CudaGraph
  participant X as CaptureSession
  S->>G: BeginCaptureToGraph(dependencies)
  S->>X: retain stream + graph owners
  Note over S,G: Dispose is rejected while capture is active
  S->>G: enqueue work
  S->>X: End()
  X->>S: cudaStreamEndCapture
  X->>G: verify returned handle is the same graph
  X-->>S: release session owner count
```

## Debug DOT

`CudaGraph.ExportDebugDot` 可把图导出为 DOT 文件。Smoke 会检查文件存在、非空并包含 `digraph` 标记。该文件适合放进文章或 issue 作为拓扑截图来源；如果要对外发布，可以用 Graphviz 渲染为 PNG。

示例：

```powershell
dot -Tpng graph.dot -o graph.png
```

## 边界

- CUDA Graph smoke 证明 wrapper 能在当前 CUDA runtime 上完成 graph round-trip。
- 它不证明 TensorRT engine enqueue 已经使用 graph capture。
- 它不证明所有 kernel node attribute 都适用于所有 CUDA 版本。
- `BeginCaptureToGraph` 只在 `CUDART_VERSION >= 12030` 映射到 vendor API；CUDA
  11.x/12.1 不会伪造该入口。`ToGraph=True` 是当前兼容主机上的 source-tree
  smoke 结果，不是 clean package-consumer runtime proof。
- `CudaDeviceGraphMemorySummary` 是 pointer-free copied readonly summary，`RuntimeEvidenceKind=copied-readonly-summary`，不能晋级 runtime proof，也不能删除 deferred history。
- 它不替代 package consumer 的 native asset copy 证据。

如果当前机器返回 CUDA error 35，应先处理 driver/runtime 兼容性，不要修改 graph API 来绕过环境问题。
