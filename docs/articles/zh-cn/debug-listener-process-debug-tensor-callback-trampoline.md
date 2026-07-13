# DebugListener ProcessDebugTensor Callback Trampoline

> 状态：callback-trampoline-shape-ready
> readiness marker：`debug-listener-process-debug-tensor-callback-trampoline`
> runtime evidence：`RuntimeEvidenceKind=callback-trampoline-shape`
> 当前结论：`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`

`debug-listener-process-debug-tensor-callback-trampoline` 是 `IDebugListener::processDebugTensor` 的 private/internal callback trampoline 形状证据。它把 no-throw vtable callback stub、borrowed debug tensor metadata copy gate 和 real non-null attach runtime smoke report 合并成一个 pointer-free C# 诊断面。

该阶段仍然是 non-proof，也就是 not proof。它证明项目已经具备可审查的 callback entry shape、异常捕获/status mapping、in-flight accounting、detach-before-release sequencing 和 borrowed debug tensor metadata copy 设计，但不证明 TensorRT runtime 已经调用 `IDebugListener::processDebugTensor`。

## Public Surface

- `TensorRtDebugListenerProcessDebugTensorCallbackTrampoline`
- `TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult`
- `TensorRtDebugTensorMetadataSnapshot`

关键字段：

- `RuntimeEvidenceKind=callback-trampoline-shape`
- `TrampolineShapeReady`
- `NativeCallbackEntryLocated`
- `NoThrowCallbackEntryReady`
- `ExceptionCaptureReady`
- `CallbackStatusMappingReady`
- `InFlightAccountingReady`
- `DetachBeforeReleaseReady`
- `BorrowedDebugTensorMetadataCopyReady`
- `BorrowedDebugTensorPointerExposed=False`
- `BorrowedDebugTensorDataPointerExposed=False`
- `PointerFreeSurfaceReady=True`
- `ProcessDebugTensorRuntimeReady=False`
- `NativeVTableInstalled=False`
- `ProcessDebugTensorInvoked=False`
- `InvocationCount=0`
- `CallbackStubEntryCount`
- `CallbackStubLeaveCount`
- `CanPromoteRealCallbackRuntime=False`
- `RuntimeProofBlocked=True`

`TensorRtDebugTensorMetadataSnapshot` 只包含复制后的安全元数据，例如 tensor name、type、location、shape rank、shape summary 和 flags。它不包含 native debug tensor pointer，也不包含 debug tensor data pointer。

## Native Scaffold

native 侧提供 `debug_listener_process_debug_tensor_callback_trampoline.inc`，其中包含：

- `DebugListenerProcessDebugTensorCallbackTrampoline final`
- `DebugListenerProcessDebugTensorCallbackReport`
- `DebugListenerBorrowedDebugTensorMetadataCopy`
- `DebugListenerCallbackInFlightScope`

核心 no-throw entry/report 方法：

- `configure_api_line`
- `configure_shape`
- `begin_callback`
- `complete_callback_success`
- `complete_callback_failure`
- `can_return_status_without_throwing`
- `trampoline_shape_ready`
- `make_report`

这些方法用于固定 ABI 边界和 source-visible safety shape。当前 scaffold 不安装 native `IDebugListener` vtable，不启用默认 non-null attach，也不伪造 callback invocation。

## Smoke And Package Consumer

`CallbackAllocatorSafeControlsSmokeRunner` 和 package consumer smoke 输出：

- `DebugListenerProcessDebugTensorCallbackTrampoline=...`
- `EvidenceKind=debug-listener-process-debug-tensor-callback-trampoline`
- `RuntimeEvidenceKind=callback-trampoline-shape`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `PointerFreeSurfaceReady=True`
- `ProcessDebugTensorRuntimeReady=False`
- `CanPromoteRealCallbackRuntime=False`

`Test-RuntimePackageReadiness.ps1` 在 JSON/Markdown 中报告 `debugListenerProcessDebugTensorCallbackTrampoline`，并继续要求 `IDebugListener::processDebugTensor` deferred row 保留。

## Non-Proof Boundary

以下状态都不是 proof：

- `callback-trampoline-shape`
- `callback-stub-gate`
- `borrowed-debug-tensor-metadata-gate`
- `runtime-smoke-skipped`
- `runtime-smoke-blocked`
- `runtime-smoke-attempted`
- `runtime-smoke-failed`
- `blocked-by-cuda-driver`

只有 full package consumer smoke 同时报告 `EvidenceKind=real-callback-runtime`、`RuntimeEvidenceKind=real-callback-runtime`、`RealCallbackRuntime=True`、`IsRealCallbackRuntimeProof=True`、`CallbackKind=debug-listener-process-debug-tensor`、`ProcessDebugTensorInvoked=True`、`InvocationCount>0`、`FailureCount=0`、`InFlightCallbackCount=0` 和 `FullPackageConsumerReport=True` 时，readiness 才允许把结果归类为真实 callback runtime proof。

## 下一步

下一阶段应继续从 trampoline shape 推进到真实 runtime proof：实现受控 native owner/vtable install、line-specific `setDebugListener(non-null)` attach、真实 TensorRT callback invocation 采集，以及 full package consumer proof promotion gate。
