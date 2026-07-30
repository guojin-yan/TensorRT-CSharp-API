# DebugListener Native Attach/No-Throw Preflight

> 状态：preflight / preflight-ready
> readiness marker：`debug-listener-native-attach-nothrow-preflight`
> runtime evidence：`RuntimeEvidenceKind=preflight`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：在实现真实 `setDebugListener(non-null)` 与 native `IDebugListener` no-throw vtable 前，先把可实施条件和仍阻塞 runtime proof 的条件拆成机器可读字段。

## 目标

`debug-listener-native-attach-nothrow-preflight` 位于 [DebugListener Attach/VTable Safety Gate](debug-listener-attach-vtable-safety-gate.md) 和 [DebugListener Native Owner Address Design Gate](debug-listener-native-owner-address-design-gate.md) 之间。它不实现 native bridge，而是把下一步 native attach/no-throw vtable 必须满足的条件显式化，避免把 copied owner、detach clear、borrowed tensor 或 attach/vtable design gate 误当成真实 TensorRT callback runtime。

公开 API：

- `TensorRtDebugListenerNativeAttachNoThrowPreflight`
- `TensorRtDebugListenerNativeAttachNoThrowPreflightResult`
- `Evaluate`

该 preflight 只消费 copied `TensorRtDebugListenerCallbackOwnerSnapshot`、`TensorRtDebugListenerAttachDetachDesignGateResult`、`TensorRtDebugListenerBorrowedTensorSafetyGateResult` 和 `TensorRtDebugListenerAttachVTableSafetyGateResult`。它不调用 `setDebugListener(non-null)`，不返回 raw `IntPtr` / `nint`、native listener owner pointer、debug tensor pointer、debug tensor data pointer 或 borrowed pointer。

## 当前能证明什么

`TensorRtDebugListenerNativeAttachNoThrowPreflight.Evaluate` 输出 pointer-free diagnostics：

| 字段 | 当前状态 | 说明 |
| --- | --- | --- |
| `AttachVTableSafetyGateReady` | `True` | attach/vtable safety gate 已有 copied evidence 可供 preflight 消费。 |
| `NativeDetachEntryLocated` | `True` | TRT10/TRT11 已有 `setDebugListener(nullptr)` detach/clear 证据。 |
| `ManagedCallbackKeepAliveDesignReady` | `True` | 托管 owner 的 GCHandle/delegate keep-alive、dispose、release hook 和 in-flight drain copied 证据干净。 |
| `BorrowedDebugTensorMetadataCopyDesignReady` | `True` | debug tensor metadata copy-out 设计可审计。 |
| `BorrowedDebugTensorPointerEscapeBlocked` | `True` | public API 阻止 borrowed debug tensor pointer 逃逸。 |
| `PreflightReady` | `True` | 设计预检可供下一步 native 审计使用。 |
| `NativeAttachEntryLocated` | `False` | `setDebugListener(non-null)` native attach entry 尚未实现。 |
| `StableNativeOwnerAddressDesignReady` | `False` | stable native DebugListener owner address 设计尚未实现。 |
| `NoThrowVTableDesignReady` | `False` | native `IDebugListener` no-throw vtable 设计尚未实现。 |
| `ExceptionToStatusMappingDesignReady` | `False` | native vtable exception-to-status mapping 尚未实现。 |
| `NativeVTableDesignReady` | `False` | owner address、no-throw vtable 和 exception mapping 尚未同时 ready。 |
| `CanImplementNativeAttach` | `False` | 当前不允许实施 non-null attach bridge。 |
| `BorrowedDebugTensorLifetimeRuntimeReady` | `False` | borrowed debug tensor pointer lifetime 尚未由真实 TensorRT callback 证明。 |
| `BorrowedDebugTensorDataLifetimeRuntimeReady` | `False` | debug tensor data buffer lifetime 尚未由真实 TensorRT callback 证明。 |
| `ProcessDebugTensorRuntimeReady` | `False` | `IDebugListener::processDebugTensor` runtime callback 尚未实现。 |
| `FullPackageConsumerRuntimeEvidenceReady` | `False` | full package consumer smoke 尚未输出 `real-callback-runtime` 证据。 |
| `CanAttemptRuntimeProof` | `False` | 不允许进入真实 runtime proof。 |
| `RuntimeProofBlocked` | `True` | deferred rows 必须继续保留。 |

smoke 输出必须保留：

- `EvidenceKind=debug-listener-native-attach-nothrow-preflight`
- `RuntimeEvidenceKind=preflight`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `PreflightReady=True`
- `NativeAttachEntryLocated=False`
- `NativeDetachEntryLocated=True`
- `StableNativeOwnerAddressDesignReady=False`
- `ManagedCallbackKeepAliveDesignReady=True`
- `NoThrowVTableDesignReady=False`
- `ExceptionToStatusMappingDesignReady=False`
- `BorrowedDebugTensorLifetimeRuntimeReady=False`
- `BorrowedDebugTensorDataLifetimeRuntimeReady=False`
- `ProcessDebugTensorRuntimeReady=False`
- `FullPackageConsumerRuntimeEvidenceReady=False`
- `CanImplementNativeAttach=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`

## 当前明确阻塞项

真实 DebugListener native attach/no-throw runtime proof 仍被这些条件阻塞：

1. line-specific execution context `setDebugListener(non-null)` native attach entry 尚未实现。
2. `debug-listener-native-owner-address-design-gate` 已把 stable native DebugListener owner address 设计 blocker 结构化，但 `StableNativeOwnerAddressDesignReady=False`、`NativeOwnerNonCopyableReady=False`、`NoThrowNativeDestructorReady=False` 和 `NativeOwnerLifecycleReady=False`。
3. native `IDebugListener` no-throw vtable 设计尚未实现。
4. native vtable exception-to-status mapping 尚未实现。
5. borrowed debug tensor pointer 和 data buffer lifetime 尚未由真实 TensorRT callback 证明。
6. `IDebugListener::processDebugTensor` runtime callback 尚未实现。
7. full package consumer smoke 尚未输出 `EvidenceKind=real-callback-runtime`。

因此：

- `CanImplementNativeAttach=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`
- `IsRealCallbackRuntimeProof=False`

## 不能证明什么

该 preflight 是 not proof。它不能解除以下 deferred rows：

- `IDebugListener::processDebugTensor`
- `IOutputAllocator::notifyShape`
- `IOutputAllocator::reallocateOutput`
- `IGpuAllocator::allocate`
- `IGpuAllocator::free`
- `IGpuAllocator::deallocate`
- `IGpuAllocator::reallocate`
- `IGpuAsyncAllocator::allocateAsync`
- `IGpuAsyncAllocator::deallocateAsync`

只有 native owner 生命周期、no-throw vtable、exception-to-status mapping、borrowed tensor/data lifetime、真实 TensorRT callback smoke 和 full package consumer `real-callback-runtime` 证据全部具备，readiness 才能将 DebugListener callback runtime 视为 proof。

## 文件归属

evaluator 与 result 已按职责拆开：

- `src/JYPPX.TensorRtSharp/Callbacks/Debugging/TensorRtDebugListenerNativeAttachNoThrowPreflight.cs` 只负责 `Evaluate` 与 blocker 聚合。
- `src/JYPPX.TensorRtSharp/Callbacks/Debugging/TensorRtDebugListenerNativeAttachNoThrowPreflightResult.cs` 只负责 pointer-free result 构造、属性、diagnostic 与 `ToString`。

两文件按原始 Git blob 顺序重组，保持 API、证据字段与 deferred 分类不变。
