# DebugListener Native No-Throw Destructor

> 状态：destructor-gate / destructor-gate-ready
> readiness marker：`debug-listener-native-nothrow-destructor`
> runtime evidence：`RuntimeEvidenceKind=destructor-gate`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：在真实 native owner lifecycle 与 `setDebugListener(non-null)` attach entry 实现前，提供 source-visible 的 no-throw destructor scaffold 证据。

## 目标

`debug-listener-native-nothrow-destructor` 位于 [DebugListener Native Owner NonCopyable Storage](debug-listener-native-owner-noncopyable-storage.md) 和 [DebugListener Runtime Proof Precheck](debug-listener-runtime-proof-precheck.md) 之间。它只证明 native 侧已有一个不可复制、不可移动、`noexcept` 析构的 source-visible scaffold，并把这条证据以 pointer-free 的 C# public diagnostics 暴露给 precheck。

托管 evaluator 位于 `TensorRtDebugListenerNativeNoThrowDestructor.cs`，pointer-free result model 位于
`TensorRtDebugListenerNativeNoThrowDestructorResult.cs`。readiness 与源码测试必须组合读取这两个文件。

公开 API：

- `TensorRtDebugListenerNativeNoThrowDestructor`
- `TensorRtDebugListenerNativeNoThrowDestructorResult`
- `Evaluate`

native scaffold：

- `native/src/tensorrt/common/debug_listener_native_nothrow_destructor.inc`
- `DebugListenerNativeNoThrowDestructor`
- `DebugListenerNativeNoThrowDestructor(const DebugListenerNativeNoThrowDestructor&) = delete`
- `operator=(const DebugListenerNativeNoThrowDestructor&) = delete`
- `DebugListenerNativeNoThrowDestructor(DebugListenerNativeNoThrowDestructor&&) = delete`
- `operator=(DebugListenerNativeNoThrowDestructor&&) = delete`
- `~DebugListenerNativeNoThrowDestructor() noexcept = default`
- `std::is_nothrow_destructible`
- `debug_listener_native_nothrow_destructor.inc`

该 destructor gate 不分配真实 native owner，不导出新的 attach C ABI，不返回 owner pointer，不调用 `setDebugListener(non-null)`，不实现 `IDebugListener::processDebugTensor`，也不允许 public API 暴露 raw `IntPtr` / `nint`、debug tensor pointer、debug tensor data pointer 或 borrowed pointer。

## 当前能证明什么

| 字段 | 当前状态 | 说明 |
| --- | --- | --- |
| `NativeOwnerNonCopyableStorageReady` | `True` | 已消费 `debug-listener-native-owner-noncopyable-storage`。 |
| `NativeOwnerNonCopyableReady` | `True` | source-visible storage scaffold 已阻止 copy/move。 |
| `NativeOwnerCopyBlocked` / `NativeOwnerMoveBlocked` | `True` | copy/move 构造与赋值已删除。 |
| `NativeOwnerAddressExposed` / `NativeOwnerPointerProduced` | `False` | public API 不暴露、不创建 native owner pointer。 |
| `DestructorNoThrowScaffoldReady` | `True` | `DebugListenerNativeNoThrowDestructor` 的析构 scaffold 为 `noexcept`。 |
| `DestructorExceptionEscapeBlocked` | `True` | 该 gate 只暴露 source-visible no-throw scaffold，不允许异常跨 ABI。 |
| `DestructorAddressExposed` / `DestructorPointerProduced` | `False` | destructor gate 不暴露地址，也不产生 borrowed pointer。 |
| `NativeAttachEntryLocated` | `False` | 真实 `setDebugListener(non-null)` native attach entry 尚未实现。 |
| `NativeDetachEntryLocated` | `True` | TRT10/TRT11 `setDebugListener(nullptr)` clear/detach entry 仍可见。 |
| `NoThrowNativeDestructorReady` | `True` | 仅表示 source-visible no-throw destructor scaffold evidence ready。 |
| `NativeOwnerLifecycleReady` | `False` | native owner attach/detach/release/drain 生命周期尚未完整。 |
| `CanImplementNativeAttach` | `False` | 当前仍不允许实施 non-null attach bridge。 |
| `CanAttemptRuntimeProof` | `False` | 当前仍不允许进入真实 runtime proof。 |
| `RuntimeProofBlocked` | `True` | direct callback deferred rows 必须保留。 |

smoke 输出必须包含：

- `EvidenceKind=debug-listener-native-nothrow-destructor`
- `RuntimeEvidenceKind=destructor-gate`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `NativeOwnerNonCopyableStorageReady=True`
- `NativeOwnerNonCopyableReady=True`
- `NativeOwnerCopyBlocked=True`
- `NativeOwnerMoveBlocked=True`
- `NativeOwnerAddressExposed=False`
- `NativeOwnerPointerProduced=False`
- `DestructorNoThrowScaffoldReady=True`
- `DestructorExceptionEscapeBlocked=True`
- `DestructorAddressExposed=False`
- `DestructorPointerProduced=False`
- `NativeAttachEntryLocated=False`
- `NativeDetachEntryLocated=True`
- `NoThrowNativeDestructorReady=True`
- `NativeOwnerLifecycleReady=False`
- `ProcessDebugTensorRuntimeReady=False`
- `CanImplementNativeAttach=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`
- `DebugListenerNativeNoThrowDestructor=`
- `DebugListenerNativeNoThrowDestructorResult`
- `NativeNoThrowDestructorGateReady`

## 当前明确阻塞项

真实 native attach/runtime proof 仍被这些条件阻塞：

1. `setDebugListener(non-null)` native attach entry 尚未实现。
2. native owner dispose/release/in-flight drain lifecycle 尚未完整。
3. `IDebugListener::processDebugTensor` runtime callback 尚未实现。
4. full package consumer smoke 尚未输出 `EvidenceKind=real-callback-runtime`。

因此：

- `NoThrowNativeDestructorReady=True` 只代表 source-visible no-throw destructor scaffold，不代表 native owner lifecycle 已完成。
- `NativeOwnerLifecycleReady=False`
- `CanImplementNativeAttach=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`

## 不能证明什么

该 destructor gate 是 source-visible / smoke-visible 的 destructor scaffold evidence，not proof。它不能证明 TensorRT 已持有 listener，不能证明 stable native address，不能证明完整 owner lifecycle，也不能解除以下 deferred rows：

- `IDebugListener::processDebugTensor`
- `IOutputAllocator::notifyShape`
- `IOutputAllocator::reallocateOutput`
- `IGpuAllocator::allocate`
- `IGpuAllocator::free`
- `IGpuAllocator::deallocate`
- `IGpuAllocator::reallocate`
- `IGpuAsyncAllocator::allocateAsync`
- `IGpuAsyncAllocator::deallocateAsync`

下一阶段应进入 DebugListener native owner lifecycle gate，继续补 detach-before-release、dispose idempotency、release hook ordering、in-flight drain 和 post-detach unpin 证据，而不是直接启用真实 callback runtime。
