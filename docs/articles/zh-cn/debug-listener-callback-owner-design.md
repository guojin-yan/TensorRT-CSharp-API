# DebugListener Callback Owner Design

> 状态：owner-design-gate / design-ready
> readiness marker：`debug-listener-callback-owner-design`
> runtime evidence：`RuntimeEvidenceKind=not-present`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：`IDebugListener::processDebugTensor` 真实 callback 前的 owner 生命周期与 copied metadata 门禁。

## 目标

`debug-listener-callback-owner-design` 用于固定 DebugListener 的高层 C# owner 形状：

- `TensorRtDebugListenerCallbackOwner`
- `TensorRtDebugListenerCallbackRequest`
- `TensorRtDebugListenerCallbackOwnerSnapshot`
- `RunDesignDiagnostic`

该 wrapper 只返回 copied snapshot，不返回 raw `IntPtr` / `nint`、native owner handle、borrowed TensorRT debug tensor pointer、debug tensor data pointer、CUDA stream handle 或 tensor buffer ownership。

## 当前能证明什么

`RunDesignDiagnostic` 会记录以下 copied diagnostics：

| 字段 | 说明 |
| --- | --- |
| `TensorName` | 复制出的 debug tensor 名称。 |
| `DataType` / `Location` | 复制出的 `TensorRtDataType` 与 `TensorRtTensorLocation`。 |
| `ShapeRank` / `ShapeSummary` | 复制出的 debug tensor shape metadata。 |
| `IsInput` / `IsOutput` | 复制出的输入/输出标记。 |
| `IsShapeTensor` / `IsExecutionTensor` | 复制出的 shape/execution tensor 标记。 |
| `ProcessDebugTensorCount` | synthetic `processDebugTensor` 门禁计数。 |
| `InFlightCallbackCount` / `MaxInFlightCallbackCount` | 托管 gate 的 callback in-flight 诊断。 |
| `ReleaseHookCount` | dispose 后释放 keep-alive 句柄的诊断计数。 |
| `DebugTensorMetadataCopied` | 有 copied metadata 时为 `True`。 |
| `DebugTensorPointerExposed` | 固定为 `False`。 |
| `DebugTensorPointerProduced` | 固定为 `False`。 |
| `BorrowedDebugTensorPointerEscaped` | 固定为 `False`。 |

smoke 输出必须保留：

- `EvidenceKind=debug-listener-callback-owner-design`
- `RuntimeEvidenceKind=not-present`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `CallbackKind=debug-listener-prototype`
- `ProcessDebugTensorCount`
- `DebugTensorMetadataCopied`
- `DebugTensorPointerExposed=False`
- `DebugTensorPointerProduced=False`
- `BorrowedDebugTensorPointerEscaped=False`

## 不能证明什么

该 owner design gate 不调用 TensorRT `setDebugListener`，不 attach 到真实 execution context，不经过真实 build/enqueue，也不执行 TensorRT vtable callback。

因此它是 design gate，not proof，不能解除以下 deferred rows：

- `IDebugListener::processDebugTensor`
- `IOutputAllocator::notifyShape`
- `IOutputAllocator::reallocateOutput`
- `IGpuAllocator::allocate`
- `IGpuAllocator::free`
- `IGpuAllocator::deallocate`
- `IGpuAllocator::reallocate`
- `IGpuAsyncAllocator::allocateAsync`
- `IGpuAsyncAllocator::deallocateAsync`

## 下一步门禁

当前已新增 [DebugListener Attach/Detach Design Gate](debug-listener-attach-detach-design-gate.md)：它以 `debug-listener-attach-detach-design-gate` 输出 `RuntimeEvidenceKind=design-gate`、`AttachControlAvailable=False`、`DetachClearControlAvailable=True`、`LineSpecificAttachDetachReady=False` 和 `RuntimeProofBlocked=True`，用于把 attach/detach 生命周期门禁结构化。它仍是 not proof。

随后新增 [DebugListener Borrowed Tensor Safety Gate](debug-listener-borrowed-tensor-safety-gate.md)：它以 `debug-listener-borrowed-tensor-safety-gate` 输出 `RuntimeEvidenceKind=design-gate`、`BorrowedDebugTensorPointerEscapeBlocked=True`、`BorrowedDebugTensorLifetimeReady=False`、`BorrowedDebugTensorDataLifetimeReady=False`、`ProcessDebugTensorRuntimeReady=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，用于把 borrowed debug tensor/data lifetime 边界结构化。它仍是 not proof。

随后新增 [DebugListener Attach/VTable Safety Gate](debug-listener-attach-vtable-safety-gate.md)：它以 `debug-listener-attach-vtable-safety-gate` 输出 `RuntimeEvidenceKind=design-gate`、`SafetyGateReady=True`、`AttachControlAvailable=False`、`StableNativeOwnerAddressReady=False`、`NoThrowNativeVTableReady=False`、`ExceptionToStatusMappingReady=False`、`ProcessDebugTensorRuntimeReady=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，用于把 non-null attach 与 native vtable 安全边界结构化。它仍是 not proof。

随后由 [DebugListener Runtime Proof Precheck](debug-listener-runtime-proof-precheck.md) 消费 attach/detach design gate、borrowed tensor safety gate、attach/vtable safety gate 与 owner snapshot，输出 `RuntimeEvidenceKind=runtime-gate`、`BorrowedTensorSafetyGateReady=True`、`AttachVTableSafetyGateReady=True`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，用于固定进入真实 runtime proof 前的阻塞项。它仍是 not proof。

进入真实 DebugListener runtime proof 前仍必须完成：

1. line-specific execution context attach/detach。
2. native owner stable address 与 no-throw vtable。
3. TensorRT 可能保存 callback pointer 时的 dispose 顺序。
4. debug tensor pointer 与 debug tensor data buffer lifetime 设计。
5. `processDebugTensor` exception-to-status 映射。
6. full package consumer smoke 由真实 TensorRT build/enqueue 触发 callback，并输出 `EvidenceKind=real-callback-runtime`。

在这些条件完成前，readiness 中 `debugListenerCallbackOwnerDesign.isRealCallbackRuntimeProof=false`，`realCallbackRuntimeEvidence.isRealCallbackRuntimeProof=false`。
