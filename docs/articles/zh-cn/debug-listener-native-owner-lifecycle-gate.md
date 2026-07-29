# DebugListener Native Owner Lifecycle Gate

> 状态：lifecycle-gate / lifecycle-gate-ready
> readiness marker：`debug-listener-native-owner-lifecycle-gate`
> runtime evidence：`RuntimeEvidenceKind=lifecycle-gate`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：在真实 `setDebugListener(non-null)` attach entry 和 native `IDebugListener` vtable 实现前，提供 pointer-free 的 native owner lifecycle scaffold 证据。

## 目标

`debug-listener-native-owner-lifecycle-gate` 位于 [DebugListener Native No-Throw Destructor](debug-listener-native-nothrow-destructor.md) 和 [DebugListener Runtime Proof Precheck](debug-listener-runtime-proof-precheck.md) 之间。它只证明 source-visible scaffold 已覆盖 detach-before-release、release hook、dispose idempotency、in-flight drain 和 post-detach unpin 的结构，不创建真实 native owner，不调用 `setDebugListener(non-null)`，也不实现 `IDebugListener::processDebugTensor`。

公开 API：

- `TensorRtDebugListenerNativeOwnerLifecycleGate`
- `TensorRtDebugListenerNativeOwnerLifecycleGateResult`
- `Evaluate`

evaluator 位于 `src/JYPPX.TensorRtSharp/Callbacks/Debugging/TensorRtDebugListenerNativeOwnerLifecycleGate.cs`，
只读 result model 位于同目录的 `TensorRtDebugListenerNativeOwnerLifecycleGateResult.cs`。完整 gate evidence consumer
必须读取两份源码，不能把 evaluator 单文件当作完整实现。

native scaffold：

- `native/src/tensorrt/common/debug_listener_native_owner_lifecycle_gate.inc`
- `DebugListenerNativeOwnerLifecycleGate`
- `DebugListenerNativeOwnerLifecycleGate(const DebugListenerNativeOwnerLifecycleGate&) = delete`
- `operator=(const DebugListenerNativeOwnerLifecycleGate&) = delete`
- `DebugListenerNativeOwnerLifecycleGate(DebugListenerNativeOwnerLifecycleGate&&) = delete`
- `operator=(DebugListenerNativeOwnerLifecycleGate&&) = delete`
- `request_detach_before_release`
- `request_release`
- `can_unpin_after_detach`
- `std::is_nothrow_destructible`
- `debug_listener_native_owner_lifecycle_gate.inc`

## 当前能证明什么

| 字段 | 当前状态 | 说明 |
| --- | --- | --- |
| `NativeNoThrowDestructorGateReady` | `True` | 已消费 `debug-listener-native-nothrow-destructor`。 |
| `ManagedDisposeSnapshotReady` | `True` | 托管 copied snapshot 显示 dispose 后 release hook、in-flight 和 pin 状态干净。 |
| `LifecycleScaffoldReady` | `True` | source-visible lifecycle scaffold 已存在。 |
| `ReleaseHookOrderingGateReady` | `True` | release hook ordering 目前是 gate scaffold evidence。 |
| `DisposeIdempotencyGateReady` | `True` | dispose/release 幂等性目前是 gate scaffold evidence。 |
| `InFlightDrainGateReady` | `True` | in-flight drain 目前是 gate scaffold evidence。 |
| `CallbackStateUnpinAfterDetachGateReady` | `True` | callback state post-detach unpin 目前是 gate scaffold evidence。 |
| `DelegateUnpinAfterDetachGateReady` | `True` | delegate post-detach unpin 目前是 gate scaffold evidence。 |
| `LifecycleAddressExposed` / `LifecyclePointerProduced` | `False` | public API 不暴露、不产生 native owner pointer。 |
| `NativeAttachEntryLocated` | `False` | non-null attach entry 尚未实现。 |
| `NativeDetachEntryLocated` | `True` | detach/clear entry 仍可见。 |
| `NoThrowNativeDestructorReady` | `True` | 只表示 no-throw destructor scaffold ready。 |
| `ReleaseHookOrderingReady` | `False` | 真实 release hook ordering 尚未完成。 |
| `DisposeIdempotencyReady` | `False` | 真实 dispose/release 幂等性尚未完成。 |
| `InFlightDrainBeforeReleaseReady` | `False` | 真实 release 前 drain 尚未完成。 |
| `CallbackStateUnpinAfterDetachReady` / `DelegateUnpinAfterDetachReady` | `False` | 真实 post-detach unpin 顺序尚未完成。 |
| `LifecycleGateReady` | `True` | 表示 lifecycle-gate scaffold evidence 可供下一阶段消费。 |
| `NativeOwnerLifecycleReady` | `False` | 完整 native owner lifecycle 尚未实现。 |
| `NativeVTableDesignReady` | `False` | native `IDebugListener` vtable 尚未实现。 |
| `ProcessDebugTensorRuntimeReady` | `False` | runtime callback 尚未实现。 |
| `CanImplementNativeAttach` | `False` | 当前不允许实施 non-null attach bridge。 |
| `CanAttemptRuntimeProof` | `False` | 当前不允许进入真实 runtime proof。 |
| `RuntimeProofBlocked` | `True` | direct callback deferred rows 必须保留。 |

smoke 输出必须包含：

- `EvidenceKind=debug-listener-native-owner-lifecycle-gate`
- `RuntimeEvidenceKind=lifecycle-gate`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `NativeNoThrowDestructorGateReady=True`
- `ManagedDisposeSnapshotReady=True`
- `LifecycleScaffoldReady=True`
- `ReleaseHookOrderingGateReady=True`
- `DisposeIdempotencyGateReady=True`
- `InFlightDrainGateReady=True`
- `CallbackStateUnpinAfterDetachGateReady=True`
- `DelegateUnpinAfterDetachGateReady=True`
- `LifecycleAddressExposed=False`
- `LifecyclePointerProduced=False`
- `NativeAttachEntryLocated=False`
- `NativeDetachEntryLocated=True`
- `NoThrowNativeDestructorReady=True`
- `ReleaseHookOrderingReady=False`
- `DisposeIdempotencyReady=False`
- `InFlightDrainBeforeReleaseReady=False`
- `CallbackStateUnpinAfterDetachReady=False`
- `DelegateUnpinAfterDetachReady=False`
- `LifecycleGateReady=True`
- `NativeOwnerLifecycleReady=False`
- `NativeVTableDesignReady=False`
- `ProcessDebugTensorRuntimeReady=False`
- `CanImplementNativeAttach=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`
- `DebugListenerNativeOwnerLifecycleGate=`
- `DebugListenerNativeOwnerLifecycleGateResult`

## 当前明确阻塞项

真实 DebugListener attach/runtime proof 仍被这些条件阻塞：

1. `setDebugListener(non-null)` native attach entry 尚未实现。
2. 完整 native owner detach/release/drain/unpin lifecycle 尚未连接到真实 TensorRT owner。
3. native `IDebugListener` no-throw vtable、exception capture、status mapping 和 in-flight accounting 尚未实现。
4. `IDebugListener::processDebugTensor` runtime callback 尚未实现。
5. full package consumer smoke 尚未输出 `EvidenceKind=real-callback-runtime`。

因此：

- `LifecycleGateReady=True` 只代表 lifecycle scaffold evidence ready。
- `NativeOwnerLifecycleReady=False`
- `NativeVTableDesignReady=False`
- `CanImplementNativeAttach=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`

## 不能证明什么

该 lifecycle gate 是 source-visible / smoke-visible 的 lifecycle scaffold evidence，not proof。它不能证明 TensorRT 已持有 listener，不能证明 native vtable 已工作，不能证明 `processDebugTensor` 已被 TensorRT 调用，也不能解除以下 deferred rows：

- `IDebugListener::processDebugTensor`
- `IOutputAllocator::notifyShape`
- `IOutputAllocator::reallocateOutput`
- `IGpuAllocator::allocate`
- `IGpuAllocator::free`
- `IGpuAllocator::deallocate`
- `IGpuAllocator::reallocate`
- `IGpuAsyncAllocator::allocateAsync`
- `IGpuAsyncAllocator::deallocateAsync`

下一阶段应进入 DebugListener attach bridge / native no-throw vtable 批处理：补 line-specific non-null attach entry 形状、owner ownership diagnostics、vtable exception-to-status scaffold、package readiness 和 smoke 证据，但仍不得开启真实 runtime proof，直到完整 package-consumer `real-callback-runtime` 证据出现。
