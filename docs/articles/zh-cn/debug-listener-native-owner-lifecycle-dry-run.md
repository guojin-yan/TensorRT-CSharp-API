# DebugListener Native Owner Lifecycle Dry-Run

> 状态：dry-run / dry-run-ready
> readiness marker：`debug-listener-native-owner-lifecycle-dry-run`
> runtime evidence：`RuntimeEvidenceKind=dry-run`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：在真实 native DebugListener owner 实现前，用 pointer-free copied evidence 预演 owner lifecycle 阻塞项。

## 目标

`debug-listener-native-owner-lifecycle-dry-run` 位于 [DebugListener Native Detach Before Release Design Gate](debug-listener-native-detach-before-release-design-gate.md) 和 [DebugListener Native Attach Entry Runtime Scaffold](debug-listener-native-attach-entry-runtime-scaffold.md) 之间。它不分配 native owner，不暴露 owner pointer，不调用 `setDebugListener(non-null)`，也不触发 `IDebugListener::processDebugTensor`。

公开 API：

- `TensorRtDebugListenerNativeOwnerLifecycleDryRun`
- `TensorRtDebugListenerNativeOwnerLifecycleDryRunResult`
- `Evaluate`

该 dry-run 只复制 owner design、attach/detach、borrowed tensor、attach/vtable、native attach/no-throw、owner address、no-throw vtable、attach entry 和 detach-before-release gate 的结果。public API 不暴露 raw `IntPtr` / `nint`、native owner pointer、debug tensor pointer、debug tensor data pointer 或任何 borrowed pointer。

## 当前能证明什么

| 字段 | 当前状态 | 说明 |
| --- | --- | --- |
| `NativeDetachBeforeReleaseDesignGateReady` | `True` | detach-before-release design gate 已可供 dry-run 消费。 |
| `NativeAttachEntryDesignGateReady` | `True` | attach entry design gate 已可供 dry-run 消费。 |
| `NativeNoThrowVTableDesignGateReady` | `True` | no-throw vtable design gate 已可供 dry-run 消费。 |
| `NativeOwnerAddressDesignGateReady` | `True` | owner address design gate 已可供 dry-run 消费。 |
| `NativeDetachEntryLocated` | `True` | TRT10/TRT11 `setDebugListener(nullptr)` clear/detach entry 已定位。 |
| `NativeAttachEntryLocated` | `False` | `setDebugListener(non-null)` native attach entry 尚未实现。 |
| `ManagedCallbackKeepAliveDesignReady` | `True` | 托管 owner dispose/release/unpin/in-flight drain 证据干净。 |
| `BorrowedDebugTensorMetadataCopyDesignReady` | `True` | debug tensor metadata 复制设计已就绪。 |
| `BorrowedDebugTensorPointerEscapeBlocked` | `True` | public API 阻止 borrowed debug tensor pointer 逃逸。 |
| `DryRunReady` | `True` | copied evidence 足以进入下一阶段 `debug-listener-native-attach-entry-runtime-scaffold`。 |
| `StableNativeOwnerIdentityReady` | `False` | stable native owner identity dry-run 尚未实现。 |
| `NativeOwnerNonCopyableReady` | `False` | native owner non-copyable storage dry-run 尚未实现。 |
| `NativeOwnerDisposeOrderReady` | `False` | native owner dispose ordering dry-run 尚未实现。 |
| `NativeOwnerReleaseHookReady` | `False` | native owner release hook dry-run 尚未实现。 |
| `NativeOwnerInFlightDrainReady` | `False` | native owner in-flight drain dry-run 尚未实现。 |
| `DetachBeforeReleaseReady` | `False` | native detach-before-release dry-run 尚未实现。 |
| `ReleaseHookOrderingReady` | `False` | native release hook ordering dry-run 尚未实现。 |
| `DisposeIdempotencyReady` | `False` | native dispose/release idempotency dry-run 尚未实现。 |
| `InFlightDrainBeforeReleaseReady` | `False` | release 前 in-flight callback drain dry-run 尚未实现。 |
| `CallbackStateUnpinAfterDetachReady` | `False` | callback state post-detach unpin dry-run 尚未实现。 |
| `DelegateUnpinAfterDetachReady` | `False` | delegate post-detach unpin dry-run 尚未实现。 |
| `NoThrowNativeDestructorReady` | `False` | native owner no-throw destructor dry-run 尚未实现。 |
| `NativeOwnerLifecycleReady` | `False` | native owner lifecycle 尚未完整。 |
| `NativeVTableDesignReady` | `False` | native vtable lifecycle 设计尚未完整。 |
| `CanImplementNativeAttach` | `False` | 当前不允许实施 non-null attach bridge。 |
| `CanAttemptRuntimeProof` | `False` | 当前不允许进入真实 runtime proof。 |
| `RuntimeProofBlocked` | `True` | direct callback deferred rows 必须保留。 |

smoke 输出必须保留：

- `EvidenceKind=debug-listener-native-owner-lifecycle-dry-run`
- `RuntimeEvidenceKind=dry-run`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `DryRunReady=True`
- `NativeDetachBeforeReleaseDesignGateReady=True`
- `NativeAttachEntryDesignGateReady=True`
- `NativeNoThrowVTableDesignGateReady=True`
- `NativeOwnerAddressDesignGateReady=True`
- `NativeDetachEntryLocated=True`
- `NativeAttachEntryLocated=False`
- `StableNativeOwnerIdentityReady=False`
- `NativeOwnerNonCopyableReady=False`
- `NativeOwnerDisposeOrderReady=False`
- `NativeOwnerReleaseHookReady=False`
- `NativeOwnerInFlightDrainReady=False`
- `DetachBeforeReleaseReady=False`
- `ReleaseHookOrderingReady=False`
- `DisposeIdempotencyReady=False`
- `InFlightDrainBeforeReleaseReady=False`
- `CallbackStateUnpinAfterDetachReady=False`
- `DelegateUnpinAfterDetachReady=False`
- `NoThrowNativeDestructorReady=False`
- `NativeOwnerLifecycleReady=False`
- `NativeVTableDesignReady=False`
- `ManagedCallbackKeepAliveDesignReady=True`
- `BorrowedDebugTensorMetadataCopyDesignReady=True`
- `BorrowedDebugTensorPointerEscapeBlocked=True`
- `CanImplementNativeAttach=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`

## 当前明确阻塞项

真实 native DebugListener attach/release/runtime proof 仍被这些条件阻塞：

1. stable native DebugListener owner identity 尚未实现。
2. native owner non-copyable storage 尚未实现。
3. native owner dispose ordering 和 release hook 尚未实现。
4. native owner in-flight callback drain 尚未实现。
5. detach-before-release ordering 尚未实现。
6. release hook ordering 与 dispose/release idempotency 尚未实现。
7. callback state post-detach unpin 尚未实现。
8. delegate post-detach unpin 尚未实现。
9. native owner no-throw destructor 尚未实现。
10. native vtable lifecycle 尚未完整。
11. borrowed debug tensor pointer/data lifetime 尚未由真实 TensorRT callback 证明。
12. `IDebugListener::processDebugTensor` runtime callback 尚未实现。
13. full package consumer smoke 尚未输出 `EvidenceKind=real-callback-runtime`。

## 不能证明什么

该 dry-run 是 owner lifecycle dry-run，not proof。它不能证明 native owner 已创建，不能证明 detach-before-release 已执行，也不能解除以下 deferred rows：

- `IDebugListener::processDebugTensor`
- `IOutputAllocator::notifyShape`
- `IOutputAllocator::reallocateOutput`
- `IGpuAllocator::allocate`
- `IGpuAllocator::free`
- `IGpuAllocator::deallocate`
- `IGpuAllocator::reallocate`
- `IGpuAsyncAllocator::allocateAsync`
- `IGpuAsyncAllocator::deallocateAsync`

只有 stable native owner identity、non-copyable storage、release hook ordering、dispose idempotency、in-flight drain、post-detach unpin、no-throw destructor、native attach entry、borrowed tensor/data lifetime 和 full package consumer `real-callback-runtime` 证据全部具备，readiness 才能把 DebugListener callback runtime 视为 proof。
