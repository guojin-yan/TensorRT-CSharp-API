# DebugListener Native No-Throw VTable Design Gate

> 状态：design-gate / design-gate-ready
> readiness marker：`debug-listener-native-nothrow-vtable-design-gate`
> runtime evidence：`RuntimeEvidenceKind=design-gate`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：在真实 `IDebugListener` native vtable trampoline 前，把 no-throw vtable、exception capture、status mapping 和 in-flight accounting 拆成 pointer-free 阻塞项。

## 目标

`debug-listener-native-nothrow-vtable-design-gate` 位于 [DebugListener Native Owner Address Design Gate](debug-listener-native-owner-address-design-gate.md) 和 [DebugListener Native Attach Entry Design Gate](debug-listener-native-attach-entry-design-gate.md) 之间。它不创建 native `IDebugListener` owner，不安装 vtable，不调用 `setDebugListener(non-null)`，也不触发 `IDebugListener::processDebugTensor`；它只把 no-throw native destructor、no-throw vtable、exception-to-status mapping、callback exception capture、callback status mapping 和 callback in-flight accounting 固定为可审计字段。

公开 API：

- `TensorRtDebugListenerNativeNoThrowVTableDesignGate`
- `TensorRtDebugListenerNativeNoThrowVTableDesignGateResult`
- `Evaluate`

该 gate 只消费 copied `TensorRtDebugListenerCallbackOwnerSnapshot`、`TensorRtDebugListenerAttachDetachDesignGateResult`、`TensorRtDebugListenerBorrowedTensorSafetyGateResult`、`TensorRtDebugListenerAttachVTableSafetyGateResult`、`TensorRtDebugListenerNativeAttachNoThrowPreflightResult` 和 `TensorRtDebugListenerNativeOwnerAddressDesignGateResult`。public API 不暴露 raw `IntPtr` / `nint`、native listener owner pointer、debug tensor pointer、debug tensor data pointer 或任何 borrowed pointer。

## 当前能证明什么

`TensorRtDebugListenerNativeNoThrowVTableDesignGate.Evaluate` 输出 pointer-free diagnostics：

| 字段 | 当前状态 | 说明 |
| --- | --- | --- |
| `NativeOwnerAddressDesignGateReady` | `True` | native owner address design gate 已有 copied evidence 可供 no-throw vtable gate 消费。 |
| `NativeAttachNoThrowPreflightReady` | `True` | native attach/no-throw preflight 已有 copied evidence。 |
| `ManagedCallbackKeepAliveDesignReady` | `True` | 托管 owner 的 release hook、unpin、dispose 和 in-flight drain copied 证据干净。 |
| `BorrowedDebugTensorMetadataCopyDesignReady` | `True` | debug tensor metadata copy-out 设计可审计。 |
| `BorrowedDebugTensorPointerEscapeBlocked` | `True` | public API 阻止 borrowed debug tensor pointer 逃逸。 |
| `DesignGateReady` | `True` | gate 可用于下一步 native no-throw vtable 设计，不表示 native vtable 已实现。 |
| `NativeAttachEntryLocated` | `False` | `setDebugListener(non-null)` native attach entry 尚未实现。 |
| `NativeOwnerLifecycleReady` | `False` | stable native owner address、不可复制 owner、dispose order、release hook 与 in-flight drain 尚未完整。 |
| `NoThrowNativeDestructorReady` | `False` | native owner no-throw destructor 尚未实现。 |
| `NoThrowVTableDesignReady` | `False` | native `IDebugListener` no-throw vtable design 尚未实现。 |
| `ExceptionToStatusMappingDesignReady` | `False` | managed/native exception-to-status mapping 尚未实现。 |
| `NativeVTableTrampolineReady` | `False` | native `IDebugListener` vtable trampoline 尚未实现。 |
| `CallbackExceptionCaptureReady` | `False` | callback exception capture diagnostic 尚未实现。 |
| `CallbackStatusMappingReady` | `False` | callback failure status mapping 尚未实现。 |
| `CallbackInFlightAccountingReady` | `False` | callback in-flight accounting 尚未实现。 |
| `NativeVTableDesignReady` | `False` | owner lifecycle、no-throw destructor、vtable、exception mapping 和 callback accounting 尚未同时 ready。 |
| `BorrowedDebugTensorLifetimeRuntimeReady` | `False` | borrowed debug tensor pointer lifetime 尚未由真实 TensorRT callback 证明。 |
| `BorrowedDebugTensorDataLifetimeRuntimeReady` | `False` | debug tensor data buffer lifetime 尚未由真实 TensorRT callback 证明。 |
| `ProcessDebugTensorRuntimeReady` | `False` | `IDebugListener::processDebugTensor` runtime callback 尚未实现。 |
| `FullPackageConsumerRuntimeEvidenceReady` | `False` | full package consumer 尚未输出 `real-callback-runtime` 证据。 |
| `CanImplementNativeAttach` | `False` | 当前不允许实施 non-null attach bridge。 |
| `CanAttemptRuntimeProof` | `False` | 不允许进入真实 runtime proof。 |
| `RuntimeProofBlocked` | `True` | deferred rows 必须继续保留。 |

smoke 输出必须保留：

- `EvidenceKind=debug-listener-native-nothrow-vtable-design-gate`
- `RuntimeEvidenceKind=design-gate`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `DesignGateReady=True`
- `NativeOwnerAddressDesignGateReady=True`
- `NativeAttachNoThrowPreflightReady=True`
- `NativeAttachEntryLocated=False`
- `NativeOwnerLifecycleReady=False`
- `ManagedCallbackKeepAliveDesignReady=True`
- `NoThrowNativeDestructorReady=False`
- `NoThrowVTableDesignReady=False`
- `ExceptionToStatusMappingDesignReady=False`
- `BorrowedDebugTensorMetadataCopyDesignReady=True`
- `BorrowedDebugTensorPointerEscapeBlocked=True`
- `NativeVTableTrampolineReady=False`
- `CallbackExceptionCaptureReady=False`
- `CallbackStatusMappingReady=False`
- `CallbackInFlightAccountingReady=False`
- `NativeVTableDesignReady=False`
- `CanImplementNativeAttach=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`

## 当前明确阻塞项

真实 DebugListener native no-throw vtable / callback runtime proof 仍被这些条件阻塞：

1. line-specific execution context `setDebugListener(non-null)` native attach entry 尚未实现。
2. native DebugListener owner lifecycle 尚未实现。
3. native owner no-throw destructor 尚未实现。
4. native `IDebugListener` no-throw vtable design 尚未实现。
5. native vtable trampoline 尚未实现。
6. managed/native exception-to-status mapping 尚未实现。
7. callback exception capture、status mapping 和 in-flight accounting 尚未实现。
8. borrowed debug tensor pointer 和 data buffer lifetime 尚未由真实 TensorRT callback 证明。
9. `IDebugListener::processDebugTensor` runtime callback 尚未实现。
10. full package consumer smoke 尚未输出 `EvidenceKind=real-callback-runtime`。

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

只有 stable native owner lifecycle、no-throw destructor、native vtable trampoline、exception capture/status mapping、callback in-flight accounting、borrowed tensor/data lifetime、真实 TensorRT callback smoke 和 full package consumer `real-callback-runtime` 证据全部具备，readiness 才能将 DebugListener callback runtime 视为 proof。
