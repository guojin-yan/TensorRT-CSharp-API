# DebugListener Native Owner Address Design Gate

> 状态：design-gate / design-gate-ready
> readiness marker：`debug-listener-native-owner-address-design-gate`
> runtime evidence：`RuntimeEvidenceKind=design-gate`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：在真实 `setDebugListener(non-null)` 和 native `IDebugListener` vtable 实现前，先把 native owner address 生命周期要求固定为 pointer-free、机器可读的阻塞项。

## 目标

`debug-listener-native-owner-address-design-gate` 位于 [DebugListener Native Attach/No-Throw Preflight](debug-listener-native-attach-nothrow-preflight.md) 和 [DebugListener Native No-Throw VTable Design Gate](debug-listener-native-nothrow-vtable-design-gate.md) 之间。它不实现 native owner，不返回 owner address，也不调用 `setDebugListener(non-null)`；它只把稳定 native owner address、不可复制 owner、dispose 顺序、release hook、in-flight drain、no-throw destructor、no-throw vtable 和 exception-to-status mapping 拆成可审计字段。下一步 `debug-listener-native-nothrow-vtable-design-gate` 会继续把 native vtable trampoline、exception capture、status mapping 与 callback in-flight accounting 固定为阻塞项。

公开 API：

- `TensorRtDebugListenerNativeOwnerAddressDesignGate`
- `TensorRtDebugListenerNativeOwnerAddressDesignGateResult`
- `Evaluate`

该 gate 只消费 copied `TensorRtDebugListenerCallbackOwnerSnapshot`、`TensorRtDebugListenerAttachDetachDesignGateResult`、`TensorRtDebugListenerBorrowedTensorSafetyGateResult`、`TensorRtDebugListenerAttachVTableSafetyGateResult` 和 `TensorRtDebugListenerNativeAttachNoThrowPreflightResult`。public API 不暴露 raw `IntPtr` / `nint`、native listener owner pointer、debug tensor pointer、debug tensor data pointer 或任何 borrowed pointer。

## 当前能证明什么

`TensorRtDebugListenerNativeOwnerAddressDesignGate.Evaluate` 输出 pointer-free diagnostics：

| 字段 | 当前状态 | 说明 |
| --- | --- | --- |
| `NativeAttachNoThrowPreflightReady` | `True` | native attach/no-throw preflight 已有 copied evidence 可供 owner address gate 消费。 |
| `NativeDetachEntryLocated` | `True` | TRT10/TRT11 已有 `setDebugListener(nullptr)` detach/clear 证据。 |
| `ManagedCallbackKeepAliveDesignReady` | `True` | 托管 owner 的 dispose、release hook、unpin 和 in-flight drain copied 证据干净。 |
| `BorrowedDebugTensorMetadataCopyDesignReady` | `True` | debug tensor metadata copy-out 设计可审计。 |
| `BorrowedDebugTensorPointerEscapeBlocked` | `True` | public API 阻止 borrowed debug tensor pointer 逃逸。 |
| `DesignGateReady` | `True` | gate 可用于下一步 native owner 生命周期设计，不表示 native owner 已实现。 |
| `NativeAttachEntryLocated` | `False` | `setDebugListener(non-null)` native attach entry 尚未实现。 |
| `StableNativeOwnerAddressReady` | `False` | stable native DebugListener owner address 尚未实现。 |
| `StableNativeOwnerAddressDesignReady` | `False` | stable native DebugListener owner address design 尚未实现。 |
| `NativeOwnerNonCopyableReady` | `False` | native owner 不可复制存储模型尚未实现。 |
| `NativeOwnerDisposeOrderReady` | `False` | native owner dispose 顺序尚未实现。 |
| `NativeOwnerReleaseHookReady` | `False` | native owner release hook 尚未实现。 |
| `NativeOwnerInFlightDrainReady` | `False` | native owner in-flight callback drain 尚未实现。 |
| `NoThrowNativeDestructorReady` | `False` | native owner no-throw destructor 尚未实现。 |
| `NoThrowVTableDesignReady` | `False` | native `IDebugListener` no-throw vtable design 尚未实现。 |
| `ExceptionToStatusMappingDesignReady` | `False` | managed/native exception-to-status mapping 尚未实现。 |
| `NativeOwnerLifecycleReady` | `False` | native owner address lifecycle 尚未完整。 |
| `NativeVTableDesignReady` | `False` | owner lifecycle、no-throw vtable 和 exception mapping 尚未同时 ready。 |
| `CanImplementNativeAttach` | `False` | 当前不允许实施 non-null attach bridge。 |
| `CanAttemptRuntimeProof` | `False` | 不允许进入真实 runtime proof。 |
| `RuntimeProofBlocked` | `True` | deferred rows 必须继续保留。 |

smoke 输出必须保留：

- `EvidenceKind=debug-listener-native-owner-address-design-gate`
- `RuntimeEvidenceKind=design-gate`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `DesignGateReady=True`
- `NativeAttachNoThrowPreflightReady=True`
- `NativeAttachEntryLocated=False`
- `NativeDetachEntryLocated=True`
- `StableNativeOwnerAddressReady=False`
- `StableNativeOwnerAddressDesignReady=False`
- `NativeOwnerNonCopyableReady=False`
- `NativeOwnerDisposeOrderReady=False`
- `NativeOwnerReleaseHookReady=False`
- `NativeOwnerInFlightDrainReady=False`
- `NoThrowNativeDestructorReady=False`
- `NoThrowVTableDesignReady=False`
- `ExceptionToStatusMappingDesignReady=False`
- `NativeOwnerLifecycleReady=False`
- `CanImplementNativeAttach=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`

## 当前明确阻塞项

真实 DebugListener native owner address / callback runtime proof 仍被这些条件阻塞：

1. line-specific execution context `setDebugListener(non-null)` native attach entry 尚未实现。
2. stable native DebugListener owner address 尚未实现。
3. native owner 不可复制存储、dispose 顺序、release hook 和 in-flight drain 尚未实现。
4. native owner no-throw destructor 尚未实现。
5. native `IDebugListener` no-throw vtable design 尚未实现。
6. native vtable exception-to-status mapping 尚未实现。
7. borrowed debug tensor pointer 和 data buffer lifetime 尚未由真实 TensorRT callback 证明。
8. `IDebugListener::processDebugTensor` runtime callback 尚未实现。
9. full package consumer smoke 尚未输出 `EvidenceKind=real-callback-runtime`。

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

只有 stable native owner address、不可复制 owner、no-throw destructor、no-throw vtable、exception-to-status mapping、borrowed tensor/data lifetime、真实 TensorRT callback smoke 和 full package consumer `real-callback-runtime` 证据全部具备，readiness 才能将 DebugListener callback runtime 视为 proof。
