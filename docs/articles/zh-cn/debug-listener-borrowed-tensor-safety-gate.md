# DebugListener Borrowed Tensor Safety Gate

> 状态：design-gate / safety-gate-ready
> readiness marker：`debug-listener-borrowed-tensor-safety-gate`
> runtime evidence：`RuntimeEvidenceKind=design-gate`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：进入真实 `IDebugListener::processDebugTensor` runtime proof 前，先固定 borrowed debug tensor pointer 与 data buffer lifetime 的 public API 安全边界。

## 目标

`debug-listener-borrowed-tensor-safety-gate` 把 DebugListener 最容易误用的 borrowed tensor 边界拆出来单独审计：TensorRT 在 `IDebugListener::processDebugTensor` 中传入的 debug tensor 与 data buffer 都是借用生命周期，C# public API 不能返回 raw `IntPtr` / `nint` / native owner pointer / debug tensor pointer / data pointer，也不能让 borrowed pointer 逃逸到用户代码。

托管 evaluator 位于 `TensorRtDebugListenerBorrowedTensorSafetyGate.cs`，pointer-free result model 位于
`TensorRtDebugListenerBorrowedTensorSafetyGateResult.cs`。readiness 与源码测试必须组合读取这两个文件。

公开 API：

- `TensorRtDebugListenerBorrowedTensorSafetyGate`
- `TensorRtDebugListenerBorrowedTensorSafetyGateResult`
- `Evaluate`

该 gate 只消费 copied `TensorRtDebugListenerCallbackOwnerSnapshot` 和可选的 `TensorRtDebugListenerAttachDetachDesignGateResult`。它复制 tensor name、data type、location、shape rank、shape summary、input/output、shape/execution tensor 标记和 `ProcessDebugTensorCount`，不消费或返回真实 TensorRT tensor pointer 或 debug tensor data pointer。

## 当前能证明什么

`TensorRtDebugListenerBorrowedTensorSafetyGate.Evaluate` 输出 pointer-free diagnostics：

| 字段 | 当前状态 | 说明 |
| --- | --- | --- |
| `AttachDetachDesignGateReady` | 来自 attach/detach gate | attach/detach copied evidence 可供 borrowed tensor gate 消费。 |
| `OwnerDesignReady` | 来自 owner snapshot | 必须是干净的 `debug-listener-callback-owner-design` 证据。 |
| `DebugTensorMetadataCopied` | `True` when owner snapshot clean | debug tensor name、类型、位置和 shape metadata 已复制。 |
| `PointerFreeSurfaceReady` | `True` | public API 未暴露 debug tensor pointer 或 data pointer。 |
| `BorrowedDebugTensorPointerEscapeBlocked` | `True` | public API 没有 borrowed debug tensor pointer 逃逸通道。 |
| `SafetyGateReady` | `True` | copied metadata 与 pointer-free surface 已可审计。 |
| `BorrowedDebugTensorLifetimeReady` | `False` | borrowed debug tensor pointer lifetime 尚未由真实 TensorRT callback 证明。 |
| `BorrowedDebugTensorDataLifetimeReady` | `False` | debug tensor data buffer lifetime 尚未由真实 TensorRT callback 证明。 |
| `ProcessDebugTensorRuntimeReady` | `False` | `IDebugListener::processDebugTensor` runtime callback 尚未实现。 |
| `CanAttemptRuntimeProof` | `False` | 不允许进入真实 runtime proof。 |
| `RuntimeProofBlocked` | `True` | deferred rows 必须继续保留。 |

smoke 输出必须保留：

- `EvidenceKind=debug-listener-borrowed-tensor-safety-gate`
- `RuntimeEvidenceKind=design-gate`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `SafetyGateReady=True`
- `BorrowedDebugTensorPointerEscapeBlocked=True`
- `BorrowedDebugTensorLifetimeReady=False`
- `BorrowedDebugTensorDataLifetimeReady=False`
- `ProcessDebugTensorRuntimeReady=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`

## 当前明确阻塞项

真实 DebugListener borrowed tensor runtime proof 仍被这些条件阻塞：

1. borrowed debug tensor pointer lifetime 尚未经过真实 TensorRT callback 验证。
2. borrowed debug tensor data buffer lifetime 尚未经过真实 TensorRT callback 验证。
3. `IDebugListener::processDebugTensor` runtime callback 尚未实现。
4. full package consumer smoke 尚未输出 `EvidenceKind=real-callback-runtime`。

这意味着当前只能证明 public C# surface 不把 borrowed pointer 泄漏出去，不能证明真实 TensorRT debug tensor/data lifetime 已安全。

## 与 Attach/Detach Gate 和 Precheck 的关系

[DebugListener Attach/Detach Design Gate](debug-listener-attach-detach-design-gate.md) 负责生命周期边界：`DetachClearControlAvailable=True`，但 `AttachControlAvailable=False`、`NativeVTableReady=False`。

本 gate 负责 borrowed tensor/data lifetime 边界：复制 debug tensor metadata，并把 pointer escape、tensor lifetime、data buffer lifetime 和 `processDebugTensor` runtime blocker 展开为独立字段。

[DebugListener Attach/VTable Safety Gate](debug-listener-attach-vtable-safety-gate.md) 会消费本 gate 和 attach/detach gate，把 `setDebugListener(non-null)`、stable native owner address、no-throw native vtable、exception-to-status mapping 与 `processDebugTensor` runtime readiness 聚合成 `debug-listener-attach-vtable-safety-gate`。该 gate 当前保持 `AttachControlAvailable=False`、`StableNativeOwnerAddressReady=False`、`NoThrowNativeVTableReady=False`、`ExceptionToStatusMappingReady=False`、`ProcessDebugTensorRuntimeReady=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`。

[DebugListener Runtime Proof Precheck](debug-listener-runtime-proof-precheck.md) 会消费本 gate，并纳入：

- `BorrowedTensorSafetyGateReady`
- `AttachVTableSafetyGateReady`
- `BorrowedDebugTensorPointerEscapeBlocked`
- `BorrowedDebugTensorLifetimeReady`
- `BorrowedDebugTensorDataLifetimeReady`
- `ProcessDebugTensorRuntimeReady`

因此 precheck 可以区分 borrowed tensor safety gate 已可审计和真实 DebugListener borrowed tensor runtime proof 尚未完成。

## 不能证明什么

该 safety gate 是 not proof。它不能解除以下 deferred rows：

- `IDebugListener::processDebugTensor`
- `IOutputAllocator::notifyShape`
- `IOutputAllocator::reallocateOutput`
- `IGpuAllocator::allocate`
- `IGpuAllocator::free`
- `IGpuAllocator::deallocate`
- `IGpuAllocator::reallocate`
- `IGpuAsyncAllocator::allocateAsync`
- `IGpuAsyncAllocator::deallocateAsync`

只有 full package consumer smoke 输出完整 `real-callback-runtime` 字段，并且 readiness 将 `realCallbackRuntimeEvidence.isRealCallbackRuntimeProof=true`，才能说明真实 TensorRT callback runtime 已触发。
