# DebugListener In-Flight Accounting Gate

> 状态：inflight-accounting-gate / inflight-accounting-gate-ready
> readiness marker：`debug-listener-inflight-accounting-gate`
> runtime evidence：`RuntimeEvidenceKind=inflight-accounting-gate`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：审计 DebugListener callback enter/leave accounting、release-after-drain 和 callback state unpin-after-drain scaffold。

## 目标

`debug-listener-inflight-accounting-gate` 消费 `debug-listener-exception-status-mapping-gate`。它只证明 copied diagnostics 中的 in-flight counter、release hook 和 pin/unpin 状态可被 precheck 审计；它不创建 native owner，不调用 `setDebugListener(non-null)`，也不证明真实 TensorRT callback 已发生。

托管 evaluator 位于 `TensorRtDebugListenerInFlightAccountingGate.cs`，pointer-free result model 位于
`TensorRtDebugListenerInFlightAccountingGateResult.cs`。readiness 与源码测试必须组合读取这两个文件。

公开 API：

- `TensorRtDebugListenerInFlightAccountingGate`
- `TensorRtDebugListenerInFlightAccountingGateResult`
- `Evaluate`

native scaffold：

- `native/src/tensorrt/common/debug_listener_inflight_accounting_gate.inc`
- `DebugListenerInFlightAccountingGate`
- `enter_callback`
- `leave_callback`
- `can_release_after_drain`

## 当前能证明什么

| 字段 | 当前状态 | 说明 |
| --- | --- | --- |
| `ExceptionStatusMappingGateReady` | `True` | exception/status mapping gate 已可消费。 |
| `CallbackEnterAccountingGateReady` | `True` | copied diagnostics 已观察到 callback enter 计数。 |
| `CallbackLeaveAccountingGateReady` | `True` | copied in-flight count 已 drain 到 0。 |
| `CallbackInFlightNeverNegativeReady` | `True` | copied in-flight count 非负。 |
| `ReleaseAfterDrainGateReady` | `True` | dispose 后 release hook 和 drain 状态可审计。 |
| `CallbackStateUnpinAfterDrainGateReady` | `True` | callback state/delegate 已在 drain 后 unpin。 |
| `AccountingAddressExposed` | `False` | public API 不暴露 accounting address。 |
| `AccountingPointerProduced` | `False` | public API 不产生 native pointer。 |
| `NativeAttachEntryLocated` | `False` | non-null attach entry 尚未实现。 |
| `CanAttemptRuntimeProof` | `False` | 当前不允许 runtime proof。 |
| `RuntimeProofBlocked` | `True` | direct callback deferred rows 必须保留。 |

smoke/readiness marker 必须包含：

- `DebugListenerInFlightAccountingGate=`
- `DebugListenerInFlightAccountingGateResult`
- `RuntimeEvidenceKind=inflight-accounting-gate`
- `ExceptionStatusMappingGateReady=True`
- `CallbackEnterAccountingGateReady=True`
- `CallbackLeaveAccountingGateReady=True`
- `CallbackInFlightNeverNegativeReady=True`
- `ReleaseAfterDrainGateReady=True`
- `CallbackStateUnpinAfterDrainGateReady=True`
- `AccountingAddressExposed=False`
- `AccountingPointerProduced=False`
- `NativeAttachEntryLocated=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`

## 不能证明什么

该 gate 是 in-flight accounting scaffold evidence，not proof。它不能证明 TensorRT 正在或曾经调用 `IDebugListener::processDebugTensor`，不能证明 native owner release ordering 完成，也不能解除 callback deferred rows。
