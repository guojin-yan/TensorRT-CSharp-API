# DebugListener Native Owner Stable Identity

> 状态：identity-gate / identity-gate-ready
> readiness marker：`debug-listener-native-owner-stable-identity`
> runtime evidence：`RuntimeEvidenceKind=identity-gate`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：在真实 native owner 和 `setDebugListener(non-null)` attach entry 实现前，结构化 owner id / diagnostic identity 的 pointer-free evidence。

## 目标

`debug-listener-native-owner-stable-identity` 位于 [DebugListener Native Attach Entry Runtime Scaffold](debug-listener-native-attach-entry-runtime-scaffold.md) 和 [DebugListener Runtime Proof Precheck](debug-listener-runtime-proof-precheck.md) 之间。它只复制 owner id、last status、last diagnostic、release diagnostic 和 pointer-free identity 诊断，用来说明托管可审计身份链已经稳定。

公开 API：

- `TensorRtDebugListenerNativeOwnerStableIdentity`
- `TensorRtDebugListenerNativeOwnerStableIdentityResult`
- `Evaluate`

该 gate 不创建 native owner，不返回 owner pointer，不调用 `setDebugListener(non-null)`，不实现 `IDebugListener::processDebugTensor`，也不允许 public API 暴露 raw `IntPtr` / `nint`、debug tensor pointer、debug tensor data pointer 或 borrowed pointer。

## 当前能证明什么

| 字段 | 当前状态 | 说明 |
| --- | --- | --- |
| `NativeAttachEntryRuntimeScaffoldReady` | `True` | 已消费 `debug-listener-native-attach-entry-runtime-scaffold`。 |
| `StableNativeOwnerIdentityReady` | `True` | owner id / diagnostic identity 已可 pointer-free 审计；这不是 stable native address。 |
| `OwnerIdentityDiagnosticsReady` | `True` | copied owner id、last status、last diagnostic 和 release diagnostic 可用。 |
| `OwnerIdentityPointerFree` | `True` | public identity surface 不暴露 raw pointer。 |
| `NativeDetachEntryLocated` | `True` | TRT10/TRT11 `setDebugListener(nullptr)` clear/detach entry 仍可见。 |
| `NativeAttachEntryLocated` | `False` | 真实 `setDebugListener(non-null)` native attach entry 尚未实现。 |
| `NativeOwnerNonCopyableReady` | `False` | 该 identity gate 自身尚未消费 storage scaffold；后续 [DebugListener Native Owner NonCopyable Storage](debug-listener-native-owner-noncopyable-storage.md) 会把 source-visible no-copy/no-move storage scaffold 提升为 `True`。 |
| `NoThrowNativeDestructorReady` | `False` | native owner no-throw destructor 尚未实现。 |
| `NativeOwnerLifecycleReady` | `False` | native owner lifecycle 尚未完整。 |
| `CanImplementNativeAttach` | `False` | 当前仍不允许实施 non-null attach bridge。 |
| `CanAttemptRuntimeProof` | `False` | 当前仍不允许进入真实 runtime proof。 |
| `RuntimeProofBlocked` | `True` | direct callback deferred rows 必须保留。 |

smoke 输出必须包含：

- `EvidenceKind=debug-listener-native-owner-stable-identity`
- `RuntimeEvidenceKind=identity-gate`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `NativeAttachEntryRuntimeScaffoldReady=True`
- `StableNativeOwnerIdentityReady=True`
- `OwnerIdentityDiagnosticsReady=True`
- `OwnerIdentityPointerFree=True`
- `NativeDetachEntryLocated=True`
- `NativeAttachEntryLocated=False`
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
2. native owner non-copyable storage 需要由 [DebugListener Native Owner NonCopyable Storage](debug-listener-native-owner-noncopyable-storage.md) 独立证明；identity gate 本身不证明 storage。
3. native owner no-throw destructor 尚未实现。
4. native owner lifecycle 尚未完整。
5. `IDebugListener::processDebugTensor` runtime callback 尚未实现。
6. full package consumer smoke 尚未输出 `EvidenceKind=real-callback-runtime`。

## 不能证明什么

该 identity gate 是 source-visible / smoke-visible 的 copied diagnostics，not proof。`StableNativeOwnerIdentityReady=True` 只表示托管 owner id / diagnostic identity 链稳定，不表示 native owner address 稳定，也不表示 TensorRT 已持有 listener。

它不能解除以下 deferred rows：

- `IDebugListener::processDebugTensor`
- `IOutputAllocator::notifyShape`
- `IOutputAllocator::reallocateOutput`
- `IGpuAllocator::allocate`
- `IGpuAllocator::free`
- `IGpuAllocator::deallocate`
- `IGpuAllocator::reallocate`
- `IGpuAsyncAllocator::allocateAsync`
- `IGpuAsyncAllocator::deallocateAsync`

下一阶段进入 [DebugListener Native Owner NonCopyable Storage](debug-listener-native-owner-noncopyable-storage.md)，只提升 source-visible no-copy/no-move storage scaffold；即使该阶段让 precheck 中的 `NativeOwnerNonCopyableReady=True`，也仍不能直接启用真实 callback runtime。

## 文件归属

evaluator 与 result 已按职责拆开：

- `src/JYPPX.TensorRtSharp/Callbacks/Debugging/TensorRtDebugListenerNativeOwnerStableIdentity.cs` 只负责 `Evaluate` 与 blocker 聚合。
- `src/JYPPX.TensorRtSharp/Callbacks/Debugging/TensorRtDebugListenerNativeOwnerStableIdentityResult.cs` 只负责 pointer-free result 构造、属性、diagnostic 与 `ToString`。

两文件按原始 Git blob 顺序重组，保持 owner identity 字段、API surface 与 deferred 分类不变。
