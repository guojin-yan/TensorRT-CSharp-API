# DebugListener Native Attach Bridge Shape Gate

> 状态：attach-bridge-shape-gate / attach-bridge-shape-gate-ready
> readiness marker：`debug-listener-native-attach-bridge-shape-gate`
> runtime evidence：`RuntimeEvidenceKind=attach-bridge-shape-gate`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：在启用真实 `setDebugListener(non-null)` 前，审计 DebugListener native attach bridge 的参数形状、TRT10/TRT11 version guard、no-throw boundary 和 ownership diagnostics。

## 目标

`debug-listener-native-attach-bridge-shape-gate` 位于 `debug-listener-native-owner-lifecycle-gate` 之后。它只证明 attach bridge 的 source-visible 形状已经可以被托管 precheck 消费，不创建 native listener owner，不返回 borrowed pointer，不调用 `setDebugListener(non-null)`，也不实现 `IDebugListener::processDebugTensor`。

公开 API：

- `TensorRtDebugListenerNativeAttachBridgeShapeGate`
- `TensorRtDebugListenerNativeAttachBridgeShapeGateResult`
- `Evaluate`

native scaffold：

- `native/src/tensorrt/common/debug_listener_native_attach_bridge_shape_gate.inc`
- `DebugListenerNativeAttachBridgeShapeGate`
- `DebugListenerNativeAttachBridgeShapeGate(const DebugListenerNativeAttachBridgeShapeGate&) = delete`
- `configure_shape`
- `std::is_nothrow_destructible`

## 当前能证明什么

| 字段 | 当前状态 | 说明 |
| --- | --- | --- |
| `NativeOwnerLifecycleGateReady` | `True` | 已消费 `debug-listener-native-owner-lifecycle-gate`。 |
| `AttachBridgeShapeReady` | `True` | native attach bridge 参数 shape scaffold 已 source-visible。 |
| `AttachBridgeNoThrowBoundaryReady` | `True` | attach bridge no-throw/status boundary 已结构化。 |
| `AttachBridgeVersionGuardReady` | `True` | TRT10/TRT11 guard 已结构化；TRT8 不开启 non-null DebugListener attach。 |
| `AttachBridgeOwnershipDiagnosticsReady` | `True` | ownership diagnostics 已结构化且不暴露 native owner pointer。 |
| `AttachBridgePointerFree` | `True` | public result 只返回 copied diagnostics。 |
| `SetDebugListenerNonNullEnabled` | `False` | 当前仍禁止启用 non-null attach。 |
| `NonNullAttachStillDisabled` | `True` | readiness 必须把该 gate 识别为非 proof。 |
| `NativeAttachEntryLocated` | `False` | line-specific non-null attach entry 尚未实现。 |
| `NativeDetachEntryLocated` | `True` | detach/clear entry 仍可审计。 |
| `NativeOwnerLifecycleReady` | `False` | 完整 native owner lifecycle 尚未完成。 |
| `NativeVTableDesignReady` | `False` | native `IDebugListener` vtable 尚未完成。 |
| `CanImplementNativeAttach` | `False` | 当前不允许实施真实 attach bridge。 |
| `CanAttemptRuntimeProof` | `False` | 当前不允许进入真实 runtime proof。 |
| `RuntimeProofBlocked` | `True` | direct callback deferred rows 必须保留。 |

smoke/readiness marker 必须包含：

- `DebugListenerNativeAttachBridgeShapeGate=`
- `DebugListenerNativeAttachBridgeShapeGateResult`
- `RuntimeEvidenceKind=attach-bridge-shape-gate`
- `NativeOwnerLifecycleGateReady=True`
- `AttachBridgeShapeReady=True`
- `AttachBridgeNoThrowBoundaryReady=True`
- `AttachBridgeVersionGuardReady=True`
- `AttachBridgeOwnershipDiagnosticsReady=True`
- `AttachBridgePointerFree=True`
- `SetDebugListenerNonNullEnabled=False`
- `NonNullAttachStillDisabled=True`
- `NativeAttachEntryLocated=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`

## 不能证明什么

该 gate 是 attach bridge shape scaffold evidence，not proof。它不能证明 TensorRT 已保存 listener，不能证明 native vtable 已工作，不能证明 `IDebugListener::processDebugTensor` 已被调用，也不能解除 `IDebugListener::processDebugTensor` deferred row。
