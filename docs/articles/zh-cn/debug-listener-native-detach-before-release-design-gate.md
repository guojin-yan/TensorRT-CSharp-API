# DebugListener Native Detach-Before-Release Design Gate

> 状态：design-gate / design-gate-ready
> readiness marker：`debug-listener-native-detach-before-release-design-gate`
> runtime evidence：`RuntimeEvidenceKind=design-gate`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：在真实 native DebugListener owner release 前，把 detach/clear、release hook 顺序、in-flight drain 和 post-detach unpin 阻塞项结构化。

## 目标

`debug-listener-native-detach-before-release-design-gate` 位于 [DebugListener Native Attach Entry Design Gate](debug-listener-native-attach-entry-design-gate.md) 和 [DebugListener Native Owner Lifecycle Dry-Run](debug-listener-native-owner-lifecycle-dry-run.md) 之间。它不创建 native `IDebugListener` owner，不调用 `setDebugListener(non-null)`，不执行真实 detach，也不触发 `IDebugListener::processDebugTensor`；它只把 release 前必须先 detach 的安全顺序固定为 pointer-free 字段。

公开 API：

- `TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate`
- `TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult`
- `Evaluate`

该 gate 只消费 copied owner / attach-entry / no-throw vtable / owner-address evidence。public API 不暴露 raw `IntPtr` / `nint`、native owner pointer、debug tensor pointer、debug tensor data pointer 或任何 borrowed pointer。

## 当前能证明什么

| 字段 | 当前状态 | 说明 |
| --- | --- | --- |
| `NativeAttachEntryDesignGateReady` | `True` | attach entry design gate 已可供 detach-before-release gate 消费。 |
| `NativeNoThrowVTableDesignGateReady` | `True` | no-throw vtable design gate 已可供消费。 |
| `NativeOwnerAddressDesignGateReady` | `True` | owner address design gate 已可供消费。 |
| `NativeDetachEntryLocated` | `True` | TRT10/TRT11 `setDebugListener(nullptr)` clear/detach entry 已定位。 |
| `DesignGateReady` | `True` | gate 可用于下一阶段 `debug-listener-native-owner-lifecycle-dry-run`。 |
| `NativeAttachEntryLocated` | `False` | `setDebugListener(non-null)` native attach entry 尚未实现。 |
| `LineSpecificAttachEntryDesignReady` | `False` | TRT10/TRT11 line-specific attach entry 设计尚未完成。 |
| `AttachEntryNoThrowReady` | `False` | native attach entry no-throw boundary 尚未实现。 |
| `AttachEntryVersionGuardReady` | `False` | native attach entry version guard 尚未实现。 |
| `AttachEntryOwnershipReady` | `False` | native attach entry ownership contract 尚未实现。 |
| `DetachBeforeReleaseReady` | `False` | detach-before-release ordering 尚未实现。 |
| `ReleaseHookOrderingReady` | `False` | release hook 尚未保证先 detach 再 release native state。 |
| `DisposeIdempotencyReady` | `False` | native dispose/release 幂等性尚未实现。 |
| `InFlightDrainBeforeReleaseReady` | `False` | release 前 in-flight callback drain 尚未实现。 |
| `CallbackStateUnpinAfterDetachReady` | `False` | callback state 还没有 post-detach unpin 证据。 |
| `DelegateUnpinAfterDetachReady` | `False` | delegate pinning 还没有 post-detach unpin 证据。 |
| `NativeOwnerLifecycleReady` | `False` | native owner lifecycle 尚未完整。 |
| `NativeVTableDesignReady` | `False` | native no-throw vtable 尚未完整。 |
| `CanImplementNativeAttach` | `False` | 当前不允许实施 non-null attach bridge。 |
| `CanAttemptRuntimeProof` | `False` | 当前不允许进入真实 runtime proof。 |
| `RuntimeProofBlocked` | `True` | direct callback deferred rows 必须保留。 |

smoke 输出必须保留：

- `EvidenceKind=debug-listener-native-detach-before-release-design-gate`
- `RuntimeEvidenceKind=design-gate`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `DesignGateReady=True`
- `NativeAttachEntryDesignGateReady=True`
- `NativeDetachEntryLocated=True`
- `NativeAttachEntryLocated=False`
- `DetachBeforeReleaseReady=False`
- `ReleaseHookOrderingReady=False`
- `DisposeIdempotencyReady=False`
- `InFlightDrainBeforeReleaseReady=False`
- `CallbackStateUnpinAfterDetachReady=False`
- `DelegateUnpinAfterDetachReady=False`
- `NativeOwnerLifecycleReady=False`
- `NativeVTableDesignReady=False`
- `CanImplementNativeAttach=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`

下游 `debug-listener-native-owner-lifecycle-dry-run` 会继续消费该 gate，并把 stable owner identity、non-copyable storage、release hook ordering、dispose idempotency、in-flight drain、post-detach unpin 与 no-throw destructor blockers 作为 dry-run evidence 结构化。

## 当前明确阻塞项

真实 native DebugListener attach/release/runtime proof 仍被这些条件阻塞：

1. `setDebugListener(non-null)` native attach entry 尚未实现。
2. attach entry no-throw boundary、version guard 和 ownership contract 尚未实现。
3. release hook ordering 尚未保证先 detach/clear，再 release native state。
4. native dispose/release 幂等性尚未实现。
5. in-flight callback drain before release 尚未实现。
6. callback state 和 delegate pinning 尚未保证 detach 完成后再 unpin。
7. native owner lifecycle 与 native no-throw vtable 仍未完整。
8. borrowed debug tensor pointer/data lifetime 尚未由真实 TensorRT callback 证明。
9. `IDebugListener::processDebugTensor` runtime callback 尚未实现。
10. full package consumer smoke 尚未输出 `EvidenceKind=real-callback-runtime`。

## 不能证明什么

该 gate 是 design gate，not proof。它不能证明已经执行真实 detach，也不能解除以下 deferred rows：

- `IDebugListener::processDebugTensor`
- `IOutputAllocator::notifyShape`
- `IOutputAllocator::reallocateOutput`
- `IGpuAllocator::allocate`
- `IGpuAllocator::free`
- `IGpuAllocator::deallocate`
- `IGpuAllocator::reallocate`
- `IGpuAsyncAllocator::allocateAsync`
- `IGpuAsyncAllocator::deallocateAsync`

只有 line-specific attach/detach、no-throw boundary、version guard、ownership contract、release hook ordering、dispose idempotency、in-flight drain、post-detach unpin、stable native owner lifecycle、borrowed tensor/data lifetime 和 full package consumer `real-callback-runtime` 证据全部具备，readiness 才能把 DebugListener callback runtime 视为 proof。
