# DebugListener Native Attach Entry Minimal Safety

> 状态：minimal-safety / minimal-safety-ready
> readiness marker：`debug-listener-native-attach-entry-minimal-safety`
> runtime evidence：`RuntimeEvidenceKind=minimal-safety`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：在不启用 `setDebugListener(non-null)` 的前提下，把 DebugListener native attach entry 的 source-visible no-throw 入口形状固定下来。

## 目标

`debug-listener-native-attach-entry-minimal-safety` 位于 [DebugListener Native Attach Entry Runtime Scaffold](debug-listener-native-attach-entry-runtime-scaffold.md) 之后。它把 TRT10/TRT11 attach entry 的参数形状、version guard、no-throw 边界、ownership diagnostics 和 native source-visible scaffold 合成一个 pointer-free 结果。

公开 API：

- `TensorRtDebugListenerNativeAttachEntryMinimalSafety`
- `TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult`
- `Evaluate`

native source-visible scaffold：

- `native/src/tensorrt/common/debug_listener_native_attach_entry_minimal_safety.inc`
- `DebugListenerNativeAttachEntryMinimalSafety final`
- `can_call_set_debug_listener_non_null`

## 当前能证明什么

| 字段 | 当前状态 | 说明 |
| --- | --- | --- |
| `RuntimeScaffoldReady` | `True` | 已消费 attach entry runtime scaffold。 |
| `LifecycleGateReady` | `True` | 已消费 native owner lifecycle gate 的 pointer-free evidence。 |
| `LifecyclePointerFree` | `True` | public surface 不暴露 native owner / destructor / lifecycle pointer。 |
| `NativeAttachEntryLocated` | `True` | 仅表示 minimal-safety 结果内的 source-visible attach entry shape 已定位。 |
| `NativeDetachEntryLocated` | `True` | detach/clear entry 仍可用于释放路径。 |
| `AttachEntryParameterShapeReady` | `True` | attach entry 参数 shape 已就绪。 |
| `AttachEntryNoThrowReady` | `True` | attach entry no-throw 边界已就绪。 |
| `AttachEntryVersionGuardReady` | `True` | TRT10/TRT11 version guard 已就绪。 |
| `AttachEntryOwnershipDiagnosticsReady` | `True` | ownership diagnostics 已就绪且不返回 borrowed pointer。 |
| `SetDebugListenerNonNullEnabled` | `False` | 仍不允许启用 non-null attach。 |
| `NonNullAttachStillDisabled` | `True` | non-null attach 继续被设计阻断。 |
| `NativeAttachWouldBeBlocked` | `True` | 即使入口 shape 可审查，实际 native attach 仍会被 gate 阻止。 |
| `ProcessDebugTensorRuntimeReady` | `False` | 未实现真实 callback runtime。 |
| `CanImplementNativeAttach` | `False` | 该结果不能单独授权实现/启用 attach。 |
| `CanAttemptRuntimeProof` | `False` | 不能进入真实 runtime proof。 |
| `RuntimeProofBlocked` | `True` | direct callback deferred rows 必须保留。 |

## 重要边界

`NativeAttachEntryLocated=True` 只属于 `debug-listener-native-attach-entry-minimal-safety` 自身，含义是“源码中存在可审查的 no-throw attach entry shape”。它不能传播到 `debug-listener-runtime-proof-precheck` 或 `debug-listener-runtime-proof-attempt-preflight`，也不能被解释为真实 `setDebugListener(non-null)` 已启用。

现有 precheck / attempt preflight 必须继续保持：

- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`
- `CanEnableSetDebugListenerNonNull=False`

## smoke 输出要求

`CallbackAllocatorSafeControlsSmokeRunner` 必须输出：

- `EvidenceKind=debug-listener-native-attach-entry-minimal-safety`
- `RuntimeEvidenceKind=minimal-safety`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `MinimalSafetyReady=True`
- `RuntimeScaffoldReady=True`
- `LifecycleGateReady=True`
- `LifecyclePointerFree=True`
- `NativeAttachEntryLocated=True`
- `NativeDetachEntryLocated=True`
- `AttachEntryParameterShapeReady=True`
- `AttachEntryNoThrowReady=True`
- `AttachEntryVersionGuardReady=True`
- `AttachEntryOwnershipDiagnosticsReady=True`
- `SetDebugListenerNonNullEnabled=False`
- `NonNullAttachStillDisabled=True`
- `NativeAttachWouldBeBlocked=True`
- `ProcessDebugTensorRuntimeReady=False`
- `CanImplementNativeAttach=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`
- `ReasonNativeAttachStillBlocked`

## 不能证明什么

该 evidence 是 minimal-safety，not proof。它不创建 native owner，不安装 native vtable，不调用 `setDebugListener(non-null)`，不触发 `IDebugListener::processDebugTensor`，也不能解除以下 deferred rows：

- `IDebugListener::processDebugTensor`
- `IOutputAllocator::notifyShape`
- `IOutputAllocator::reallocateOutput`
- `IGpuAllocator::allocate`
- `IGpuAllocator::free`
- `IGpuAllocator::deallocate`
- `IGpuAllocator::reallocate`
- `IGpuAsyncAllocator::allocateAsync`
- `IGpuAsyncAllocator::deallocateAsync`

下一阶段可以推进 no-throw vtable callback stub 的安全桥接，但仍不得启用真实 callback trampoline。
