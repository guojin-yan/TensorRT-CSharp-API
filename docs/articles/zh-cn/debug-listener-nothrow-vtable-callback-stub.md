# DebugListener No-Throw VTable Callback Stub

> 状态：callback-stub-gate / callback-stub-gate-ready
> readiness marker：`debug-listener-nothrow-vtable-callback-stub`
> runtime evidence：`RuntimeEvidenceKind=callback-stub-gate`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：在真实 native `IDebugListener` vtable 和 `processDebugTensor` runtime 尚未启用前，审计 callback stub 的 no-throw、metadata copy、exception/status mapping 和 in-flight pairing。

## 目标

`debug-listener-nothrow-vtable-callback-stub` 消费 `debug-listener-native-attach-entry-minimal-safety` 和 `debug-listener-native-nothrow-vtable-scaffold-gate`。它只复制 DebugListener owner snapshot 中的 tensor name、data type、tensor location、rank/shape、`isInput`、`isExecutionTensor`、entry/leave count 和 failure count；不暴露 debug tensor pointer，不暴露 debug tensor data pointer，不安装 native vtable，也不调用 `setDebugListener(non-null)`。

下一层 `debug-listener-borrowed-debug-tensor-metadata-runtime-gate` 会把 copied tensor name/type/location/shape/input/output/shape/execution flags 拆成独立 metadata gate。它仍是 `RuntimeEvidenceKind=borrowed-debug-tensor-metadata-gate`，not proof，不能把 copied metadata 解释为 borrowed tensor lifetime ready。

公开 API：

- `TensorRtDebugListenerNoThrowVTableCallbackStub`
- `TensorRtDebugListenerNoThrowVTableCallbackStubResult`
- `Evaluate`

evaluator 位于 `src/JYPPX.TensorRtSharp/Callbacks/Debugging/TensorRtDebugListenerNoThrowVTableCallbackStub.cs`，
只读 result model 位于同目录的 `TensorRtDebugListenerNoThrowVTableCallbackStubResult.cs`。完整 callback-stub gate
consumer 必须读取两份源码。

native stub：

- `native/src/tensorrt/common/debug_listener_nothrow_vtable_callback_stub.inc`
- `DebugListenerNoThrowVTableCallbackStub final`
- `begin_callback`
- `complete_callback_success`
- `complete_callback_failure`
- `exception_escape_blocked`
- `can_return_status_without_throwing`
- `configure_api_line`
- `line_supports_debug_listener`

## 当前能证明什么

| 字段 | 当前状态 | 说明 |
| --- | --- | --- |
| `MinimalSafetyReady` | `True` | 已消费 native attach entry minimal-safety scoped evidence。 |
| `NoThrowVTableScaffoldGateReady` | `True` | 已消费 native no-throw vtable scaffold gate。 |
| `NoThrowVTableScaffoldReady` | `True` | no-throw vtable scaffold 形状可见。 |
| `CallbackStubGateReady` | `True` | callback stub gate 的复制诊断齐全。 |
| `CallbackStubShapeReady` | `True` | TRT10/TRT11 callback stub 参数形状可审计。 |
| `CallbackStubNoThrowReady` | `True` | stub 入口和完成路径保持 no-throw。 |
| `CallbackMetadataCopyReady` | `True` | tensor name/type/location/rank/shape/input/execution metadata 已复制。 |
| `CallbackExceptionCaptureReady` | `True` | exception capture gate 已可消费。 |
| `CallbackStatusMappingReady` | `True` | failure status mapping 已可消费。 |
| `CallbackInFlightEnterReady` | `True` | copied entry accounting 可见。 |
| `CallbackInFlightLeaveReady` | `True` | copied leave accounting 可见。 |
| `CallbackInFlightPairingReady` | `True` | copied enter/leave pairing 可见。 |
| `CallbackInFlightNeverNegativeReady` | `True` | copied in-flight 计数不为负。 |
| `BorrowedDebugTensorMetadataCopyReady` | `True` | borrowed tensor 只复制 metadata。 |
| `BorrowedDebugTensorPointerEscapeBlocked` | `True` | public API 阻止 borrowed pointer 逃逸。 |
| `DebugTensorPointerExposed` | `False` | 不暴露 debug tensor pointer。 |
| `DebugTensorDataPointerExposed` | `False` | 不暴露 debug tensor data pointer。 |
| `SetDebugListenerNonNullEnabled` | `False` | 不启用 `setDebugListener(non-null)`。 |
| `NativeAttachWouldBeBlocked` | `True` | attach 仍被门禁阻止。 |
| `NativeVTableInstalled` | `False` | native vtable 未安装。 |
| `CanInstallNativeVTable` | `False` | 当前不能安装 native vtable。 |
| `CanCallProcessDebugTensorRuntime` | `False` | 当前不能调用真实 runtime callback。 |
| `CanAttemptRuntimeProof` | `False` | 当前不能启动 runtime proof。 |
| `RuntimeProofBlocked` | `True` | direct callback deferred rows 必须保留。 |

smoke/readiness marker 必须包含：

- `DebugListenerNoThrowVTableCallbackStub=`
- `TensorRtDebugListenerNoThrowVTableCallbackStubResult`
- `RuntimeEvidenceKind=callback-stub-gate`
- `CallbackStubGateReady=True`
- `CallbackMetadataCopyReady=True`
- `DebugTensorPointerExposed=False`
- `DebugTensorDataPointerExposed=False`
- `SetDebugListenerNonNullEnabled=False`
- `NativeAttachWouldBeBlocked=True`
- `NativeVTableInstalled=False`
- `ProcessDebugTensorRuntimeReady=False`
- `CanInstallNativeVTable=False`
- `CanCallProcessDebugTensorRuntime=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`
- `ReasonCallbackRuntimeStillBlocked`

## 不能证明什么

该 gate 是 callback-stub-gate，not proof。它不能证明 TensorRT 已持有 native `IDebugListener`，不能证明真实 `IDebugListener::processDebugTensor` 已在 build/enqueue 路径执行，不能解除 `IDebugListener::processDebugTensor` deferred row，也不能作为 `real-callback-runtime`。
