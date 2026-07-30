# DebugListener Native Owner NonCopyable Storage

> 状态：storage-gate / storage-gate-ready
> readiness marker：`debug-listener-native-owner-noncopyable-storage`
> runtime evidence：`RuntimeEvidenceKind=storage-gate`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：在真实 native owner lifecycle 与 `setDebugListener(non-null)` attach entry 实现前，提供 source-visible 的 non-copyable/no-move storage scaffold 证据。

## 目标

`debug-listener-native-owner-noncopyable-storage` 位于 [DebugListener Native Owner Stable Identity](debug-listener-native-owner-stable-identity.md) 和 [DebugListener Native No-Throw Destructor](debug-listener-native-nothrow-destructor.md) 之间。它只证明 native 侧已经有一个不可复制、不可移动、默认 `noexcept` 析构的 storage scaffold，并把这条证据以 pointer-free 的 C# public diagnostics 暴露给后续 destructor gate 与 precheck。

托管 evaluator 位于 `TensorRtDebugListenerNativeOwnerNonCopyableStorage.cs`，pointer-free result model 位于
`TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.cs`。readiness 与源码测试必须组合读取这两个文件。

公开 API：

- `TensorRtDebugListenerNativeOwnerNonCopyableStorage`
- `TensorRtDebugListenerNativeOwnerNonCopyableStorageResult`
- `Evaluate`

native scaffold：

- `native/src/tensorrt/common/debug_listener_native_owner_noncopyable_storage.inc`
- `DebugListenerNativeOwnerNonCopyableStorage`
- `DebugListenerNativeOwnerNonCopyableStorage(const DebugListenerNativeOwnerNonCopyableStorage&) = delete`
- `operator=(const DebugListenerNativeOwnerNonCopyableStorage&) = delete`
- `DebugListenerNativeOwnerNonCopyableStorage(DebugListenerNativeOwnerNonCopyableStorage&&) = delete`
- `operator=(DebugListenerNativeOwnerNonCopyableStorage&&) = delete`
- `std::is_nothrow_destructible`
- `debug_listener_native_owner_noncopyable_storage.inc`

该 storage gate 不分配真实 native owner，不导出新的 C ABI，不返回 owner pointer，不调用 `setDebugListener(non-null)`，不实现 `IDebugListener::processDebugTensor`，也不允许 public API 暴露 raw `IntPtr` / `nint`、debug tensor pointer、debug tensor data pointer 或 borrowed pointer。

## 当前能证明什么

| 字段 | 当前状态 | 说明 |
| --- | --- | --- |
| `NativeOwnerStableIdentityReady` | `True` | 已消费 `debug-listener-native-owner-stable-identity`。 |
| `OwnerIdentityDiagnosticsReady` | `True` | owner id、last status、last diagnostic 和 release diagnostic 已有 pointer-free copied evidence。 |
| `OwnerIdentityPointerFree` | `True` | public identity surface 不暴露 raw pointer。 |
| `NativeOwnerNonCopyableReady` | `True` | 仅表示 source-visible storage scaffold 已阻止 copy/move；不代表真实 native owner lifecycle 完成。 |
| `NativeOwnerCopyBlocked` | `True` | copy constructor 与 copy assignment 已删除。 |
| `NativeOwnerMoveBlocked` | `True` | move constructor 与 move assignment 已删除。 |
| `NativeOwnerAddressExposed` | `False` | public API 不公开 native owner address。 |
| `NativeOwnerPointerProduced` | `False` | 当前 gate 不创建也不返回 native owner pointer。 |
| `NativeAttachEntryLocated` | `False` | 真实 `setDebugListener(non-null)` native attach entry 尚未实现。 |
| `NativeDetachEntryLocated` | `True` | TRT10/TRT11 `setDebugListener(nullptr)` clear/detach entry 仍可见。 |
| `NoThrowNativeDestructorReady` | `False` | 真实 native owner destructor lifecycle 尚未实现；storage scaffold 的默认 `noexcept` 析构不是 lifecycle proof。 |
| `NativeOwnerLifecycleReady` | `False` | native owner attach/detach/release/drain 生命周期尚未完整。 |
| `CanImplementNativeAttach` | `False` | 当前仍不允许实施 non-null attach bridge。 |
| `CanAttemptRuntimeProof` | `False` | 当前仍不允许进入真实 runtime proof。 |
| `RuntimeProofBlocked` | `True` | direct callback deferred rows 必须保留。 |

smoke 输出必须包含：

- `EvidenceKind=debug-listener-native-owner-noncopyable-storage`
- `RuntimeEvidenceKind=storage-gate`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `NativeOwnerStableIdentityReady=True`
- `OwnerIdentityDiagnosticsReady=True`
- `OwnerIdentityPointerFree=True`
- `NativeOwnerNonCopyableReady=True`
- `NativeOwnerCopyBlocked=True`
- `NativeOwnerMoveBlocked=True`
- `NativeOwnerAddressExposed=False`
- `NativeOwnerPointerProduced=False`
- `NativeAttachEntryLocated=False`
- `NativeDetachEntryLocated=True`
- `NoThrowNativeDestructorReady=False`
- `NativeOwnerLifecycleReady=False`
- `ProcessDebugTensorRuntimeReady=False`
- `CanImplementNativeAttach=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`
- `DebugListenerNativeOwnerNonCopyableStorage=`
- `DebugListenerNativeOwnerNonCopyableStorageResult`

## 当前明确阻塞项

真实 native attach/runtime proof 仍被这些条件阻塞：

1. `setDebugListener(non-null)` native attach entry 尚未实现。
2. native owner no-throw destructor lifecycle 尚未实现。
3. native owner dispose/release/in-flight drain lifecycle 尚未完整。
4. `IDebugListener::processDebugTensor` runtime callback 尚未实现。
5. full package consumer smoke 尚未输出 `EvidenceKind=real-callback-runtime`。

因此：

- `NativeOwnerNonCopyableReady=True` 只代表 storage scaffold 的 copy/move 被阻止。
- `NoThrowNativeDestructorReady=False`
- `NativeOwnerLifecycleReady=False`
- `CanImplementNativeAttach=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`

## 不能证明什么

该 storage gate 是 source-visible / smoke-visible 的 storage scaffold evidence，not proof。它不能证明 TensorRT 已持有 listener，不能证明 stable native address，不能证明 no-throw destructor lifecycle，也不能解除以下 deferred rows：

- `IDebugListener::processDebugTensor`
- `IOutputAllocator::notifyShape`
- `IOutputAllocator::reallocateOutput`
- `IGpuAllocator::allocate`
- `IGpuAllocator::free`
- `IGpuAllocator::deallocate`
- `IGpuAllocator::reallocate`
- `IGpuAsyncAllocator::allocateAsync`
- `IGpuAsyncAllocator::deallocateAsync`

下一阶段由 `debug-listener-native-nothrow-destructor` 把 source-visible no-throw destructor scaffold 提升为独立 destructor gate evidence。即便下一阶段使综合 precheck 中 `NoThrowNativeDestructorReady=True`，这也仍然只代表析构 scaffold 证据，不代表 `NativeOwnerLifecycleReady=True`，不能直接启用真实 callback runtime。
