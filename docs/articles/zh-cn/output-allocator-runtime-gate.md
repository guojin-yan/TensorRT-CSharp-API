# OutputAllocator Runtime Gate

> 状态：internal-runtime-gate
> readiness marker：`output-allocator-internal-runtime-gate`
> runtime evidence：`RealCallbackRuntime=False`
> 适用范围：`IOutputAllocator::notifyShape` 与 `IOutputAllocator::reallocateOutput` 的下一步门禁。

## 目标

`output-allocator-internal-runtime-gate` 用于把 OutputAllocator 的最小 runtime 证据形状固定下来，但不把它声明成真实 TensorRT callback。

当前实现位于 `TensorRtOutputAllocatorRuntimeGate`。它只在 dedicated smoke 和 quality tests 中通过反射触发：

- `RunInternalNotifyShapeRuntimeGate`
- `RunInternalReallocateOutputRuntimeGate`
- `GetInternalRuntimeGateSnapshot`

它复制以下信息：

| 字段 | 说明 |
| --- | --- |
| `TensorName` | copied tensor name。 |
| `RequestedSize` / `Alignment` | copied `reallocateOutput` 请求参数。 |
| `ShapeRank` / `ShapeSummary` | copied shape metadata。 |
| `NotifyShapeCount` | synthetic `notifyShape` 门禁调用次数。 |
| `ReallocateOutputCount` | synthetic `reallocateOutput` 门禁调用次数。 |
| `InFlightCallbackCount` | managed gate 内部 in-flight counter。 |
| `ReleaseHookCount` | dispose 后释放 `GCHandle` 与 delegate keep-alive 的诊断计数。 |
| `OutputBufferPointerExposed` | 固定为 `False`。 |
| `OutputBufferPointerProduced` | 固定为 `False`。 |

## 禁止升级为 runtime proof

该 gate 不注册到 TensorRT，不调用 `setOutputAllocator`，不经过 build/enqueue 路径，也不返回 output buffer 或 device pointer。

smoke 输出必须保持：

- `EvidenceKind=output-allocator-internal-runtime-gate`
- `RealCallbackRuntime=False`
- `CallbackKind=output-allocator-prototype`
- `OutputBufferPointerExposed=False`
- `OutputBufferPointerProduced=False`
- `LastStatus`
- `LastDiagnostic`

因此它是 `internal-runtime-gate`，not proof，不能解除以下 deferred rows：

- `IOutputAllocator::notifyShape`
- `IOutputAllocator::reallocateOutput`
- `IDebugListener::processDebugTensor`
- `IGpuAllocator::*`
- `IGpuAsyncAllocator::*`

## 下一步门禁

进入真实 callback runtime 前仍必须完成：

- execution context attach/detach ownership。
- TensorRT 仍可能保存 callback pointer 时的 dispose 顺序。
- output buffer device pointer ledger。
- `notifyShape` 与 `reallocateOutput` 的真实 TensorRT build/enqueue 触发证据。
- full package consumer smoke 输出 `EvidenceKind=real-callback-runtime` 与完整 counters。

下一层 high-level owner 门禁记录在 [OutputAllocator Callback Owner Design](output-allocator-callback-owner-design.md)。它提供 `TensorRtOutputAllocatorCallbackOwner`、`RunDesignDiagnostic` 与 `TensorRtOutputAllocatorCallbackOwnerSnapshot`，但仍保持 `RuntimeEvidenceKind=not-present`、`RealCallbackRuntime=False` 和 `IsRealCallbackRuntimeProof=False`。

在这些条件完成前，readiness 中 `outputAllocatorInternalRuntimeGate.isRealCallbackRuntimeProof=false`，`realCallbackRuntimeEvidence.isRealCallbackRuntimeProof=false`；full package consumer 可以是 `not-present` 或环境阻塞，但都不是 runtime proof。
