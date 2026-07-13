# DebugListener Native Attach Entry Design Gate

> 状态：design-gate / design-gate-ready
> readiness marker：`debug-listener-native-attach-entry-design-gate`
> runtime evidence：`RuntimeEvidenceKind=design-gate`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：在真实 `setDebugListener(non-null)` native attach entry 前，把 attach entry 的 no-throw、version guard、ownership 和 detach-before-release 阻塞项结构化。

## 目标

`debug-listener-native-attach-entry-design-gate` 位于 [DebugListener Native No-Throw VTable Design Gate](debug-listener-native-nothrow-vtable-design-gate.md) 和 [DebugListener Native Detach-Before-Release Design Gate](debug-listener-native-detach-before-release-design-gate.md) 之间。它不创建 native `IDebugListener` owner，不安装 vtable，不调用 `setDebugListener(non-null)`，也不触发 `IDebugListener::processDebugTensor`；它只把 native attach entry 进入真实实现前必须解决的边界条件固定为可审计字段。

公开 API：

- `TensorRtDebugListenerNativeAttachEntryDesignGate`
- `TensorRtDebugListenerNativeAttachEntryDesignGateResult`
- `Evaluate`

该 gate 只消费 copied `TensorRtDebugListenerCallbackOwnerSnapshot`、`TensorRtDebugListenerAttachDetachDesignGateResult`、`TensorRtDebugListenerBorrowedTensorSafetyGateResult`、`TensorRtDebugListenerAttachVTableSafetyGateResult`、`TensorRtDebugListenerNativeAttachNoThrowPreflightResult`、`TensorRtDebugListenerNativeOwnerAddressDesignGateResult` 和 `TensorRtDebugListenerNativeNoThrowVTableDesignGateResult`。public API 不暴露 raw `IntPtr` / `nint`、native listener owner pointer、debug tensor pointer、debug tensor data pointer 或任何 borrowed pointer。

## 当前能证明什么

`TensorRtDebugListenerNativeAttachEntryDesignGate.Evaluate` 输出 pointer-free diagnostics：

| 字段 | 当前状态 | 说明 |
| --- | --- | --- |
| `NativeNoThrowVTableDesignGateReady` | `True` | native no-throw vtable design gate 已有 copied evidence 可供 attach entry gate 消费。 |
| `NativeOwnerAddressDesignGateReady` | `True` | native owner address design gate 已有 copied evidence。 |
| `NativeAttachNoThrowPreflightReady` | `True` | native attach/no-throw preflight 已有 copied evidence。 |
| `NativeDetachEntryLocated` | `True` | TRT10/TRT11 clear/detach entry 可用。 |
| `ManagedCallbackKeepAliveDesignReady` | `True` | 托管 owner 的 release hook、unpin、dispose 和 in-flight drain copied 证据干净。 |
| `BorrowedDebugTensorMetadataCopyDesignReady` | `True` | debug tensor metadata copy-out 设计可审计。 |
| `BorrowedDebugTensorPointerEscapeBlocked` | `True` | public API 阻止 borrowed debug tensor pointer 逃逸。 |
| `DesignGateReady` | `True` | gate 可用于下一步 attach entry 设计，不表示 native attach 已实现。 |
| `NativeAttachEntryLocated` | `False` | `setDebugListener(non-null)` native attach entry 尚未实现。 |
| `LineSpecificAttachEntryDesignReady` | `False` | TRT10/TRT11 line-specific attach entry 设计尚未完成。 |
| `AttachEntryNoThrowReady` | `False` | native attach entry no-throw 边界尚未实现。 |
| `AttachEntryVersionGuardReady` | `False` | native attach entry version guard 尚未实现。 |
| `AttachEntryOwnershipReady` | `False` | native attach entry ownership contract 尚未实现。 |
| `DetachBeforeReleaseReady` | `False` | native detach-before-release ordering 尚未实现。 |
| `NativeOwnerLifecycleReady` | `False` | native owner lifecycle 尚未完整。 |
| `NativeVTableDesignReady` | `False` | native no-throw vtable 仍未 ready。 |
| `CanImplementNativeAttach` | `False` | 当前不允许实施 non-null attach bridge。 |
| `CanAttemptRuntimeProof` | `False` | 不允许进入真实 runtime proof。 |
| `RuntimeProofBlocked` | `True` | deferred rows 必须继续保留。 |

smoke 输出必须保留：

- `EvidenceKind=debug-listener-native-attach-entry-design-gate`
- `RuntimeEvidenceKind=design-gate`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `DesignGateReady=True`
- `NativeNoThrowVTableDesignGateReady=True`
- `NativeOwnerAddressDesignGateReady=True`
- `NativeAttachNoThrowPreflightReady=True`
- `NativeDetachEntryLocated=True`
- `NativeAttachEntryLocated=False`
- `LineSpecificAttachEntryDesignReady=False`
- `AttachEntryNoThrowReady=False`
- `AttachEntryVersionGuardReady=False`
- `AttachEntryOwnershipReady=False`
- `DetachBeforeReleaseReady=False`
- `NativeOwnerLifecycleReady=False`
- `NativeVTableDesignReady=False`
- `CanImplementNativeAttach=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`

## 当前明确阻塞项

真实 DebugListener native attach entry / callback runtime proof 仍被这些条件阻塞：

1. line-specific execution context `setDebugListener(non-null)` native attach entry 尚未实现。
2. TRT10/TRT11 attach entry 设计尚未完成。
3. attach entry no-throw 边界尚未实现。
4. attach entry version guard 尚未实现。
5. attach entry ownership contract 尚未实现。
6. detach-before-release ordering 尚未实现。
7. 下一阶段 `debug-listener-native-detach-before-release-design-gate` 还会把 release hook ordering、dispose idempotency、in-flight drain 和 post-detach unpin blocker 单独展开。
8. native owner lifecycle 与 native no-throw vtable 仍未完整。
9. borrowed debug tensor pointer 和 data buffer lifetime 尚未由真实 TensorRT callback 证明。
10. `IDebugListener::processDebugTensor` runtime callback 尚未实现。
11. full package consumer smoke 尚未输出 `EvidenceKind=real-callback-runtime`。

因此：

- `CanImplementNativeAttach=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`
- `IsRealCallbackRuntimeProof=False`

## 不能证明什么

该 gate 是 design gate，not proof。它不能解除以下 deferred rows：

- `IDebugListener::processDebugTensor`
- `IOutputAllocator::notifyShape`
- `IOutputAllocator::reallocateOutput`
- `IGpuAllocator::allocate`
- `IGpuAllocator::free`
- `IGpuAllocator::deallocate`
- `IGpuAllocator::reallocate`
- `IGpuAsyncAllocator::allocateAsync`
- `IGpuAsyncAllocator::deallocateAsync`

只有 line-specific attach entry、no-throw boundary、version guard、ownership contract、detach-before-release、stable native owner lifecycle、no-throw vtable、borrowed tensor/data lifetime、真实 TensorRT callback smoke 和 full package consumer `real-callback-runtime` 证据全部具备，readiness 才能将 DebugListener callback runtime 视为 proof。
