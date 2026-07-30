# DebugListener Exception Status Mapping Gate

> 状态：exception-status-gate / exception-status-gate-ready
> readiness marker：`debug-listener-exception-status-mapping-gate`
> runtime evidence：`RuntimeEvidenceKind=exception-status-gate`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：在 native `IDebugListener` vtable 完成前，审计 callback exception capture、status mapping 和 diagnostic copy scaffold。

## 目标

`debug-listener-exception-status-mapping-gate` 消费 `debug-listener-native-attach-bridge-shape-gate`。它只证明 exception/status mapping scaffold 已 source-visible，异常不会跨 C ABI 逃逸；它不调用 `setDebugListener(non-null)`，不安装 native vtable，也不触发真实 `IDebugListener::processDebugTensor`。

公开 API：

- `TensorRtDebugListenerExceptionStatusMappingGate`
- `TensorRtDebugListenerExceptionStatusMappingGateResult`
- `Evaluate`

native scaffold：

- `native/src/tensorrt/common/debug_listener_exception_status_mapping_gate.inc`
- `DebugListenerExceptionStatusMappingGate`
- `map_exception_to_status`
- `exception_escape_blocked`

## 当前能证明什么

| 字段 | 当前状态 | 说明 |
| --- | --- | --- |
| `AttachBridgeShapeGateReady` | `True` | attach bridge shape gate 已可消费。 |
| `ManagedCallbackExceptionCaptureReady` | `True` | managed copied snapshot last status clean。 |
| `NativeCallbackExceptionCaptureReady` | `True` | native callback exception capture scaffold 已 source-visible。 |
| `CallbackStatusMappingGateReady` | `True` | callback failure status mapping scaffold 已 source-visible。 |
| `ExceptionEscapeBlocked` | `True` | callback exception 不允许跨 ABI 逃逸。 |
| `DiagnosticCopyReady` | `True` | diagnostic copy 只进入 pointer-free status records。 |
| `MappingAddressExposed` | `False` | public API 不暴露 mapping address。 |
| `MappingPointerProduced` | `False` | public API 不产生 native pointer。 |
| `NativeAttachEntryLocated` | `False` | non-null attach entry 尚未实现。 |
| `CanAttemptRuntimeProof` | `False` | 当前不允许 runtime proof。 |
| `RuntimeProofBlocked` | `True` | direct callback deferred rows 必须保留。 |

smoke/readiness marker 必须包含：

- `DebugListenerExceptionStatusMappingGate=`
- `DebugListenerExceptionStatusMappingGateResult`
- `RuntimeEvidenceKind=exception-status-gate`
- `AttachBridgeShapeGateReady=True`
- `ManagedCallbackExceptionCaptureReady=True`
- `NativeCallbackExceptionCaptureReady=True`
- `CallbackStatusMappingGateReady=True`
- `ExceptionEscapeBlocked=True`
- `DiagnosticCopyReady=True`
- `MappingAddressExposed=False`
- `MappingPointerProduced=False`
- `NativeAttachEntryLocated=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`

## 不能证明什么

该 gate 是 exception/status mapping scaffold evidence，not proof。它不能证明真实 native vtable 已安装，不能证明 TensorRT 已调用 callback，不能解除 `IDebugListener::processDebugTensor` deferred row。

## 文件归属

evaluator 与 result 已按职责拆开：

- `src/JYPPX.TensorRtSharp/Callbacks/Debugging/TensorRtDebugListenerExceptionStatusMappingGate.cs` 只负责 `Evaluate` 与 blocker 聚合。
- `src/JYPPX.TensorRtSharp/Callbacks/Debugging/TensorRtDebugListenerExceptionStatusMappingGateResult.cs` 只负责 pointer-free result 构造、属性、diagnostic 与 `ToString`。

两文件按原始 Git blob 顺序重组，exception/status 字段、API surface 与 deferred 分类不变。
