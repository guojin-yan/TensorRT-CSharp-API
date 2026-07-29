# DebugListener Attach/VTable Safety Gate

> 状态：design-gate / safety-gate-ready
> readiness marker：`debug-listener-attach-vtable-safety-gate`
> runtime evidence：`RuntimeEvidenceKind=design-gate`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：进入真实 `IDebugListener::processDebugTensor` runtime proof 前，先固定 non-null attach、native owner stable address、no-throw vtable 和 exception-to-status 映射的 public API 安全边界。

## 目标

`debug-listener-attach-vtable-safety-gate` 位于 `debug-listener-attach-detach-design-gate` 和 `debug-listener-runtime-proof-precheck` 之间。它把 DebugListener 真正 attach 之前最危险的 native vtable 边界拆成可诊断字段，避免把 copied owner 证据、`setDebugListener(nullptr)` 清理能力或 borrowed tensor safety gate 误当成真实 callback runtime。

源码 owner 已按职责拆分：evaluation 与 blocker 构造位于
`TensorRtDebugListenerAttachVTableSafetyGate.cs`，pointer-free report 位于
`TensorRtDebugListenerAttachVTableSafetyGateResult.cs`。readiness 与源码测试必须组合读取这两个文件。

公开 API：

- `TensorRtDebugListenerAttachVTableSafetyGate`
- `TensorRtDebugListenerAttachVTableSafetyGateResult`
- `Evaluate`

该 gate 只消费 copied `TensorRtDebugListenerCallbackOwnerSnapshot`、`TensorRtDebugListenerAttachDetachDesignGateResult` 和 `TensorRtDebugListenerBorrowedTensorSafetyGateResult`。它不调用 `setDebugListener(non-null)`，不返回 raw `IntPtr` / `nint`、native listener owner pointer、debug tensor pointer、debug tensor data pointer 或 borrowed pointer。

## 当前能证明什么

`TensorRtDebugListenerAttachVTableSafetyGate.Evaluate` 输出 pointer-free diagnostics：

| 字段 | 当前状态 | 说明 |
| --- | --- | --- |
| `AttachDetachDesignGateReady` | `True` | attach/detach copied evidence 可供本 gate 消费。 |
| `BorrowedTensorSafetyGateReady` | `True` | borrowed tensor pointer escape 已有 copied safety evidence。 |
| `SafetyGateReady` | `True` | owner、metadata、pointer-free surface 和 detach clear copied evidence 已可审计。 |
| `AttachControlAvailable` | `False` | `setDebugListener(non-null)` attach bridge 尚未实现。 |
| `StableNativeOwnerAddressReady` | `False` | native DebugListener owner stable address 尚未实现。 |
| `NoThrowNativeVTableReady` | `False` | native `IDebugListener` no-throw vtable trampoline 尚未实现。 |
| `ExceptionToStatusMappingReady` | `False` | managed exception 到 native status 的映射尚未由 native bridge 证明。 |
| `BorrowedDebugTensorLifetimeReady` | `False` | borrowed debug tensor pointer lifetime 尚未由真实 TensorRT callback 证明。 |
| `BorrowedDebugTensorDataLifetimeReady` | `False` | debug tensor data buffer lifetime 尚未由真实 TensorRT callback 证明。 |
| `ProcessDebugTensorRuntimeReady` | `False` | `IDebugListener::processDebugTensor` runtime callback 尚未实现。 |
| `FullPackageConsumerRuntimeEvidenceReady` | `False` | full package consumer smoke 尚未输出 `real-callback-runtime` 证据。 |
| `CanAttemptRuntimeProof` | `False` | 不允许进入真实 runtime proof。 |
| `RuntimeProofBlocked` | `True` | deferred rows 必须继续保留。 |

smoke 输出必须保留：

- `EvidenceKind=debug-listener-attach-vtable-safety-gate`
- `RuntimeEvidenceKind=design-gate`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `SafetyGateReady=True`
- `AttachControlAvailable=False`
- `StableNativeOwnerAddressReady=False`
- `NoThrowNativeVTableReady=False`
- `ExceptionToStatusMappingReady=False`
- `ProcessDebugTensorRuntimeReady=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`

## 当前明确阻塞项

真实 DebugListener attach/vtable runtime proof 仍被这些条件阻塞：

1. line-specific execution context `setDebugListener(non-null)` attach bridge 尚未实现。
2. stable native DebugListener owner address 尚未实现。
3. native `IDebugListener` no-throw vtable trampoline 尚未实现。
4. exception-to-status mapping 尚未在 native vtable bridge 上证明。
5. borrowed debug tensor pointer 和 data buffer lifetime 尚未由真实 TensorRT callback 证明。
6. `IDebugListener::processDebugTensor` runtime callback 尚未实现。
7. full package consumer smoke 尚未输出 `EvidenceKind=real-callback-runtime`。

## 与其他 DebugListener Gate 的关系

[DebugListener Attach/Detach Design Gate](debug-listener-attach-detach-design-gate.md) 说明 detach clear 已有 copied control，但 non-null attach 和 native vtable 仍缺失。

[DebugListener Borrowed Tensor Safety Gate](debug-listener-borrowed-tensor-safety-gate.md) 说明 borrowed debug tensor pointer 不从 public API 逃逸，但不证明真实 TensorRT callback lifetime。

本 gate 将这两层 copied evidence 汇总成 attach/vtable safety 状态，并把 `AttachControlAvailable=False`、`StableNativeOwnerAddressReady=False`、`NoThrowNativeVTableReady=False`、`ExceptionToStatusMappingReady=False` 和 `ProcessDebugTensorRuntimeReady=False` 明确传递给 [DebugListener Native Attach/No-Throw Preflight](debug-listener-native-attach-nothrow-preflight.md) 与 [DebugListener Runtime Proof Precheck](debug-listener-runtime-proof-precheck.md)。

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
