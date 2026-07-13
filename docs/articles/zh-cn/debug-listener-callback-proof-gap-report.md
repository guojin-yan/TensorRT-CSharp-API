# DebugListener Callback Proof Gap Report

> 状态：proof-gap-report-ready
> readiness marker：`debug-listener-callback-proof-gap-report`
> runtime evidence：`RuntimeEvidenceKind=proof-gap-report`
> 当前结论：`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`

`debug-listener-callback-proof-gap-report` 是 DebugListener callback runtime proof 的机器可读缺口报告。它把 `TensorRtDebugListenerRuntimeProofAttemptPreflight`、`TensorRtDebugListenerRealNonNullAttachRuntimeSmoke`、`TensorRtDebugListenerProcessDebugTensorCallbackTrampoline` 和 `TensorRtDebugListenerRealCallbackRuntimeProof` 的结果合并为一个 pointer-free 只读对象，帮助发布前门禁明确下一步还缺什么。

它不能替代真实 TensorRT callback runtime proof。该报告不启用 `setDebugListener(non-null)`，不安装 native `IDebugListener` vtable，不调用 TensorRT，不暴露 raw pointer，也不把 `callback-owner-closure-matrix`、`runtime-smoke-skipped`、`real-callback-runtime-blocked` 或 `attempted-no-invocation` 提升为 proof。

## Public Surface

- `TensorRtDebugListenerCallbackProofGapReport`
- `TensorRtDebugListenerCallbackProofGapReportResult`

关键字段：

- `RuntimeEvidenceKind=proof-gap-report`
- `NonNullAttachStillDisabled`
- `NativeAttachEntryReady`
- `NativeVTableInstallBlocked`
- `NoThrowCallbackEntryReady`
- `ExceptionStatusMappingReady`
- `InFlightAccountingReady`
- `BorrowedDebugTensorMetadataCopied`
- `DetachRollbackReady`
- `ProcessDebugTensorRuntimeInvoked`
- `FullPackageConsumerRuntimeProofReady`
- `PointerFreeSurfaceReady`
- `AttemptedNoInvocation`
- `InvocationCount`
- `FailureCount`
- `InFlightCallbackCount`
- `CanPromoteRealCallbackRuntime`
- `RuntimeProofBlocked`
- `DeferredRowsStillRequired`
- `GapReasonCount`
- `PrimaryGapReason`
- `RuntimeProofBlockerCategory`
- `PackageConsumerRuntimeProofRequired`
- `RuntimeInvocationRequired`
- `EvidenceSource`
- `NextOwnerAction`

默认路径必须保持：

- `NonNullAttachStillDisabled=True`
- `NativeAttachEntryReady=False`
- `NativeVTableInstallBlocked=True`
- `ProcessDebugTensorRuntimeInvoked=False`
- `FullPackageConsumerRuntimeProofReady=False`
- `InvocationCount=0`
- `CanPromoteRealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`

## Package And Readiness

`CallbackAllocatorSafeControlsSmokeRunner` 输出：

```text
DebugListenerCallbackProofGapReport=debug-listener-callback-proof-gap-report;RuntimeEvidenceKind=proof-gap-report;RealCallbackRuntime=False;IsRealCallbackRuntimeProof=False;...
```

`Test-PackageConsumer.ps1` 和 `Test-RuntimePackageReadiness.ps1` 将 `proof-gap-report`、`callback-owner-closure-matrix`、`runtime-smoke-skipped`、`runtime-smoke-blocked`、`runtime-smoke-attempted`、`runtime-smoke-failed`、`callback-trampoline-shape`、`real-callback-runtime-blocked`、`attempted-no-invocation` 与 `IsRealCallbackRuntimeProof=False` 视为非证明标记。若这些标记与 `real-callback-runtime` 同时出现，readiness 必须加入 `NoNonProofCallbackRuntimeMarker`，并保持 `isRealCallbackRuntimeProof=false`。

发布前门禁会额外检查：

- `callback-owner-closure-matrix` 必须是 `closure-matrix-ready`，且不能声称 proof。
- `debug-listener-callback-proof-gap-report` 必须是 `proof-gap-report-ready`，且不能声称 proof。
- full package consumer 真实 callback proof 仍必须来自 `EvidenceKind=real-callback-runtime`、`RuntimeEvidenceKind=real-callback-runtime`、`InvocationCount>0`、`FailureCount=0`、`InFlightCallbackCount=0` 和 `FullPackageConsumerReport=True` 的通过报告。

## Next Work

下一步可以按 `GapReasons` 收敛真实 runtime proof：

- 实现并验证 non-null attach 的 native 入口，继续保持 TRT8/TRT10/TRT11 version guard。
- 安装 no-throw native vtable owner，并保证 detach-before-release、rollback 和 dispose idempotency。
- 在 full package consumer 中用真实 TensorRT build/enqueue 触发 `IDebugListener::processDebugTensor`。
- 复制 borrowed debug tensor metadata，禁止 borrowed pointer 逃逸。
- 记录 `InvocationCount>0`、`FailureCount=0` 和 `InFlightCallbackCount=0` 后再允许 `real-callback-runtime` promotion。

`RuntimeProofBlockerCategory` 和 `NextOwnerAction` 是发布收口使用的稳定机器字段。默认阻塞分类为 `non-null-attach-disabled`，下一步动作为 `enable-and-verify-non-null-debug-listener-attach-under-version-guards`；它们只帮助 owner 定位下一批工作，不改变 `RuntimeEvidenceKind=proof-gap-report`，也不会把当前报告提升为 proof。
