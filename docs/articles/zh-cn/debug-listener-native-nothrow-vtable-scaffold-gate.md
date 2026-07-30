# DebugListener Native No-Throw VTable Scaffold Gate

> 状态：vtable-scaffold-gate / vtable-scaffold-gate-ready
> readiness marker：`debug-listener-native-nothrow-vtable-scaffold-gate`
> runtime evidence：`RuntimeEvidenceKind=vtable-scaffold-gate`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：在真实 native `IDebugListener` vtable 完成前，审计 no-throw vtable scaffold、exception/status mapping 和 in-flight accounting 的组合 gate。

## 目标

`debug-listener-native-nothrow-vtable-scaffold-gate` 消费 `debug-listener-native-attach-bridge-shape-gate`、`debug-listener-exception-status-mapping-gate` 和 `debug-listener-inflight-accounting-gate`。它只证明 native vtable scaffold 的析构、callback stub、异常封锁和 accounting 结构已 source-visible；它不安装 native `IDebugListener`，不调用 `setDebugListener(non-null)`，也不触发真实 `IDebugListener::processDebugTensor`。

托管 evaluator 位于 `TensorRtDebugListenerNativeNoThrowVTableScaffoldGate.cs`，pointer-free result model 位于
`TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult.cs`。readiness 与源码测试必须组合读取这两个文件。

公开 API：

- `TensorRtDebugListenerNativeNoThrowVTableScaffoldGate`
- `TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult`
- `Evaluate`

native scaffold：

- `native/src/tensorrt/common/debug_listener_native_nothrow_vtable_scaffold_gate.inc`
- `DebugListenerNativeNoThrowVTableScaffoldGate`
- `DebugListenerNativeNoThrowVTableScaffoldGate(const DebugListenerNativeNoThrowVTableScaffoldGate&) = delete`
- `process_debug_tensor_stub`
- `exception_escape_blocked`

## 当前能证明什么

| 字段 | 当前状态 | 说明 |
| --- | --- | --- |
| `NativeAttachBridgeShapeGateReady` | `True` | attach bridge shape gate 已可消费。 |
| `ExceptionStatusMappingGateReady` | `True` | exception/status mapping gate 已可消费。 |
| `InFlightAccountingGateReady` | `True` | in-flight accounting gate 已可消费。 |
| `NoThrowVTableScaffoldReady` | `True` | no-throw vtable scaffold 已 source-visible。 |
| `VTableDestructorNoThrowReady` | `True` | vtable scaffold destructor 保持 no-throw。 |
| `ProcessDebugTensorCallbackStubNoThrowReady` | `True` | `process_debug_tensor_stub` 保持 no-throw。 |
| `ExceptionEscapeBlocked` | `True` | callback exception 不允许跨 ABI。 |
| `CallbackExceptionCaptureGateReady` | `True` | exception capture gate 已可消费。 |
| `CallbackStatusMappingGateReady` | `True` | status mapping gate 已可消费。 |
| `CallbackInFlightAccountingGateReady` | `True` | in-flight accounting gate 已可消费。 |
| `BorrowedDebugTensorPointerEscapeBlocked` | `True` | public API 阻止 borrowed debug tensor pointer 逃逸。 |
| `VTableAddressExposed` | `False` | public API 不暴露 vtable address。 |
| `VTablePointerProduced` | `False` | public API 不产生 vtable pointer。 |
| `NativeAttachEntryLocated` | `False` | non-null attach entry 尚未实现。 |
| `NativeVTableDesignReady` | `False` | 完整 native vtable 尚未完成。 |
| `CanImplementNativeAttach` | `False` | 当前不允许实施真实 attach bridge。 |
| `CanAttemptRuntimeProof` | `False` | 当前不允许 runtime proof。 |
| `RuntimeProofBlocked` | `True` | direct callback deferred rows 必须保留。 |

smoke/readiness marker 必须包含：

- `DebugListenerNativeNoThrowVTableScaffoldGate=`
- `DebugListenerNativeNoThrowVTableScaffoldGateResult`
- `RuntimeEvidenceKind=vtable-scaffold-gate`
- `NativeAttachBridgeShapeGateReady=True`
- `ExceptionStatusMappingGateReady=True`
- `InFlightAccountingGateReady=True`
- `NoThrowVTableScaffoldReady=True`
- `VTableDestructorNoThrowReady=True`
- `ProcessDebugTensorCallbackStubNoThrowReady=True`
- `ExceptionEscapeBlocked=True`
- `CallbackExceptionCaptureGateReady=True`
- `CallbackStatusMappingGateReady=True`
- `CallbackInFlightAccountingGateReady=True`
- `BorrowedDebugTensorPointerEscapeBlocked=True`
- `VTableAddressExposed=False`
- `VTablePointerProduced=False`
- `NativeAttachEntryLocated=False`
- `NativeVTableDesignReady=False`
- `CanImplementNativeAttach=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`

## 不能证明什么

该 gate 是 no-throw vtable scaffold evidence，not proof。它不能证明 native `IDebugListener` vtable 已经被 TensorRT 持有，不能证明 `processDebugTensor` 已在真实 build/enqueue 路径被调用，也不能解除 `IDebugListener::processDebugTensor` deferred row。
