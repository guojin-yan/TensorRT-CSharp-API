# DebugListener Native Attach Entry Runtime Scaffold

> 状态：scaffold / scaffold-ready
> readiness marker：`debug-listener-native-attach-entry-runtime-scaffold`
> runtime evidence：`RuntimeEvidenceKind=scaffold`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：在真实 `setDebugListener(non-null)` attach entry 实现前，结构化 attach entry runtime shape 的 pointer-free evidence。

## 目标

`debug-listener-native-attach-entry-runtime-scaffold` 位于 [DebugListener Native Owner Lifecycle Dry-Run](debug-listener-native-owner-lifecycle-dry-run.md) 和 [DebugListener Native Owner Stable Identity](debug-listener-native-owner-stable-identity.md) 之间。它只记录 native attach entry 的参数形状、TRT10/TRT11 version guard、C ABI no-throw/status mapping 预期和 ownership diagnostics。

公开 API：

- `TensorRtDebugListenerNativeAttachEntryRuntimeScaffold`
- `TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult`
- `Evaluate`

该 scaffold 不创建 native owner，不返回 owner pointer，不调用 `setDebugListener(non-null)`，不实现 `IDebugListener::processDebugTensor`，也不允许 public API 暴露 raw `IntPtr` / `nint`、debug tensor pointer、debug tensor data pointer 或 borrowed pointer。

## 当前能证明什么

| 字段 | 当前状态 | 说明 |
| --- | --- | --- |
| `NativeOwnerLifecycleDryRunReady` | `True` | 已消费 `debug-listener-native-owner-lifecycle-dry-run`。 |
| `NativeDetachEntryLocated` | `True` | TRT10/TRT11 `setDebugListener(nullptr)` clear/detach entry 仍可见。 |
| `NativeAttachEntryLocated` | `False` | 真实 `setDebugListener(non-null)` native attach entry 尚未实现。 |
| `AttachEntryParameterShapeReady` | `True` | attach entry 参数 shape 已被 pointer-free scaffold 记录。 |
| `AttachEntryVersionGuardReady` | `True` | TRT10/TRT11 line guard 预期已结构化。 |
| `AttachEntryNoThrowBoundaryReady` | `True` | C ABI no-throw/status mapping 预期已结构化。 |
| `AttachEntryOwnershipDiagnosticsReady` | `True` | ownership diagnostics 已结构化，但不暴露 borrowed pointer。 |
| `StableNativeOwnerIdentityReady` | `False` | stable native owner identity 尚未实现。 |
| `NativeOwnerNonCopyableReady` | `False` | native owner non-copyable storage 尚未实现。 |
| `NoThrowNativeDestructorReady` | `False` | native owner no-throw destructor 尚未实现。 |
| `NativeOwnerLifecycleReady` | `False` | native owner lifecycle 尚未完整。 |
| `RuntimeScaffoldReady` | `True` | scaffold 本身可用于下一阶段 owner identity 工作。 |
| `CanImplementNativeAttach` | `False` | 当前仍不允许实施 non-null attach bridge。 |
| `CanAttemptRuntimeProof` | `False` | 当前仍不允许进入真实 runtime proof。 |
| `RuntimeProofBlocked` | `True` | direct callback deferred rows 必须保留。 |

smoke 输出必须包含：

- `EvidenceKind=debug-listener-native-attach-entry-runtime-scaffold`
- `RuntimeEvidenceKind=scaffold`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `RuntimeScaffoldReady=True`
- `NativeOwnerLifecycleDryRunReady=True`
- `NativeDetachEntryLocated=True`
- `NativeAttachEntryLocated=False`
- `AttachEntryParameterShapeReady=True`
- `AttachEntryVersionGuardReady=True`
- `AttachEntryNoThrowBoundaryReady=True`
- `AttachEntryOwnershipDiagnosticsReady=True`
- `StableNativeOwnerIdentityReady=False`
- `NativeOwnerNonCopyableReady=False`
- `NoThrowNativeDestructorReady=False`
- `NativeOwnerLifecycleReady=False`
- `ProcessDebugTensorRuntimeReady=False`
- `CanImplementNativeAttach=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`

## 当前明确阻塞项

真实 native attach/runtime proof 仍被这些条件阻塞：

1. `setDebugListener(non-null)` native attach entry 尚未实现。
2. stable native DebugListener owner identity 尚未实现。
3. native owner non-copyable storage 尚未实现。
4. native owner no-throw destructor 尚未实现。
5. native owner lifecycle 尚未完整。
6. `IDebugListener::processDebugTensor` runtime callback 尚未实现。
7. full package consumer smoke 尚未输出 `EvidenceKind=real-callback-runtime`。

## 不能证明什么

该 scaffold 是 source-visible / smoke-visible 的 shape evidence，not proof。它不能证明 TensorRT 已持有 listener，也不能解除以下 deferred rows：

- `IDebugListener::processDebugTensor`
- `IOutputAllocator::notifyShape`
- `IOutputAllocator::reallocateOutput`
- `IGpuAllocator::allocate`
- `IGpuAllocator::free`
- `IGpuAllocator::deallocate`
- `IGpuAllocator::reallocate`
- `IGpuAsyncAllocator::allocateAsync`
- `IGpuAsyncAllocator::deallocateAsync`

下一阶段进入 [DebugListener Native Owner Stable Identity](debug-listener-native-owner-stable-identity.md)，先把 owner id / diagnostic identity 做成 pointer-free gate；随后再处理 native owner non-copyable storage，而不是直接启用真实 callback runtime。
