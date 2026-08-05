# 真实 Callback Runtime Evidence Schema

> 状态：schema-only
> readiness marker：`real-callback-runtime-evidence-schema`
> 当前 runtime evidence：`not-present` 或 environment-blocked，`isRealCallbackRuntimeProof=false`
> 适用范围：未来 allocator/debug-listener 真实 TensorRT callback smoke。

## 目标

本 schema 固定未来 `real-callback-runtime` 的验收格式。它只定义 full package consumer smoke 需要输出和 readiness 需要解析的字段，不表示真实 TensorRT callback 已经启用。

`real-callback-runtime` 必须来自真实 TensorRT build/enqueue 路径触发的 callback。以下证据都不能升级为 runtime proof：

- `dry-run`
- `copied-state`
- bridge-only wrapper surface
- dependency probe
- `allocator-owner-ledger-design-gate`
- `allocator-owner-internal-runtime-prototype`
- `allocator-owner-ledger-safety-gate`
- `output-allocator-internal-runtime-gate`
- `output-allocator-callback-owner-design`
- `output-allocator-attach-detach-design-gate`
- `output-buffer-ownership-safety-gate`
- `output-allocator-runtime-proof-precheck`
- `debug-listener-callback-owner-design`
- `debug-listener-attach-detach-design-gate`
- `debug-listener-borrowed-tensor-safety-gate`
- `debug-listener-attach-vtable-safety-gate`
- `debug-listener-native-attach-nothrow-preflight`
- `debug-listener-native-owner-address-design-gate`
- `debug-listener-native-nothrow-vtable-design-gate`
- `debug-listener-native-attach-entry-design-gate`
- `debug-listener-native-detach-before-release-design-gate`
- `debug-listener-native-owner-lifecycle-dry-run`
- `debug-listener-native-attach-entry-runtime-scaffold`
- `debug-listener-native-attach-entry-minimal-safety`
- `debug-listener-native-owner-stable-identity`
- `debug-listener-native-owner-noncopyable-storage`
- `debug-listener-native-nothrow-destructor`
- `debug-listener-native-owner-lifecycle-gate`
- `debug-listener-native-attach-bridge-shape-gate`
- `debug-listener-exception-status-mapping-gate`
- `debug-listener-inflight-accounting-gate`
- `debug-listener-native-nothrow-vtable-scaffold-gate`
- `debug-listener-nothrow-vtable-callback-stub`
- `debug-listener-borrowed-debug-tensor-metadata-runtime-gate`
- `debug-listener-native-vtable-install-preflight`
- `debug-listener-native-owner-vtable-install-experiment`
- `debug-listener-runtime-proof-precheck`
- `debug-listener-runtime-proof-attempt-preflight`
- `debug-listener-real-non-null-attach-runtime-smoke`
- `debug-listener-process-debug-tensor-callback-trampoline`
- `debug-listener-real-callback-runtime-proof`
- `debug-listener-callback-proof-gap-report`
- `callback-owner-closure-matrix`
- `callback-trampoline-shape`
- `proof-gap-report`
- `runtime-smoke-skipped`
- `runtime-smoke-blocked`
- `runtime-smoke-attempted`
- `runtime-smoke-failed`
- `real-callback-trampoline-gate`
- `real-callback-runtime-evidence-schema`

## Full Package Consumer Smoke 输出

未来真实 callback smoke 至少输出以下行：

| 输出字段 | 示例 | 要求 |
| --- | --- | --- |
| `EvidenceKind` | `EvidenceKind=real-callback-runtime` | 必须精确标记真实 runtime callback。 |
| `RuntimeEvidenceKind` | `RuntimeEvidenceKind=real-callback-runtime` | 必须精确标记真实 runtime callback。 |
| `RealCallbackRuntime` | `RealCallbackRuntime=True` | 必须为 true。 |
| `IsRealCallbackRuntimeProof` | `IsRealCallbackRuntimeProof=True` | 必须为 true，且只能由 full package consumer smoke 成功后输出。 |
| `CallbackKind` | `CallbackKind=sync-allocator` | 标记 sync allocator、output allocator、debug listener 等。 |
| `TensorRtLine` | `TensorRtLine=11` | 必须来自真实 runtime key。 |
| `CudaLine` | `CudaLine=13.2` | 必须来自真实 runtime key。 |
| `RuntimePackageKey` | `RuntimePackageKey=win-x64-trt11.0-cuda13.2-cudnn9.22` | 必须对应 full package consumer report。 |
| `OwnerId` | `OwnerId=1` | 必须对应 native owner 诊断 ID。 |
| `InvocationCount` | `InvocationCount=1` | 必须大于 0。 |
| `AllocationCount` | `AllocationCount=1` | allocator 类 callback 必须记录。 |
| `ReleaseCount` | `ReleaseCount=1` | allocator 类 callback 必须与 ownership 语义配对。 |
| `FailureCount` | `FailureCount=0` | 非 0 时必须有可诊断失败原因。 |
| `InFlightCallbackCount` | `InFlightCallbackCount=0` | dispose 前必须可确认。 |
| `LastStatus` | `LastStatus=Ok` | 必须是 copied status，不跨 ABI 抛异常。 |
| `LastDiagnostic` | `LastDiagnostic=OK` | 必须是 copied diagnostic。 |
| `FullPackageConsumerReport` | `FullPackageConsumerReport=...package-consumer-validation-summary.json` | 必须指向 full package consumer report。 |

## Readiness 解析

`Test-RuntimePackageReadiness.ps1` 需要产生两个独立对象：

- `realCallbackRuntimeEvidenceSchema`：schema 是否被文档、README、package-consumer 和 readiness 审计。
- `realCallbackRuntimeEvidence`：当前 full package consumer smoke 是否真的输出 `real-callback-runtime`。

`debug-listener-runtime-proof-attempt-preflight` 对应 public surface `TensorRtDebugListenerRuntimeProofAttemptPreflight` / `TensorRtDebugListenerRuntimeProofAttemptPreflightResult`。该对象只复制前置条件结果，不启用 `setDebugListener(non-null)`，也不暴露 raw pointer。当前字段必须保持 `CanEnableSetDebugListenerNonNull=False`、`CanInstallNativeVTable=False`、`CanCallProcessDebugTensorRuntime=False`、`CanPromoteRealCallbackRuntime=False`，并通过 `ReasonNonNullAttachStillBlocked`、`ReasonNativeVTableStillBlocked` 和 `ReasonRuntimeProofStillBlocked` 解释阻塞原因；这些字段只能作为 proof-attempt 准入诊断，不能作为 `real-callback-runtime`。

`debug-listener-native-vtable-install-preflight` 对应 public surface `TensorRtDebugListenerNativeVTableInstallPreflight` / `TensorRtDebugListenerNativeVTableInstallPreflightResult`。该对象只汇总 owner lifecycle、attach bridge shape、no-throw vtable scaffold 和 borrowed metadata gate 的安装前诊断，不启用 `setDebugListener(non-null)`，不安装 native vtable，也不暴露 raw pointer。当前字段必须保持 `NativeVTableInstallPreflightReady=True`、`SetDebugListenerNonNullEnabled=False`、`NativeVTableInstalled=False`、`NativeVTableInstallRuntimeReady=False`、`CanEnableSetDebugListenerNonNull=False`、`CanInstallNativeVTable=False`、`CanCallProcessDebugTensorRuntime=False` 和 `RuntimeProofBlocked=True`。

`debug-listener-native-owner-vtable-install-experiment` 对应 public surface `TensorRtDebugListenerNativeOwnerVTableInstallExperiment` / `TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult`。该对象只汇总 disabled-by-default native owner/vtable install experiment 的 shape、guard、rollback、detach-before-release、failure/status mapping 和 pointer-free 诊断，不启用 `setDebugListener(non-null)`，不尝试安装 native vtable，也不暴露 raw pointer。当前字段必须保持 `ExperimentShapeReady=True`、`InstallAttemptGuardReady=True`、`NonNullAttachEnabled=False`、`RuntimeProofEnabled=False`、`NativeVTableInstallAttempted=False`、`NativeVTableInstalled=False`、`CanEnableSetDebugListenerNonNull=False`、`CanInstallNativeVTable=False`、`CanCallProcessDebugTensorRuntime=False` 和 `RuntimeProofBlocked=True`。

`debug-listener-real-non-null-attach-runtime-smoke` 对应 public surface `TensorRtDebugListenerRealNonNullAttachRuntimeSmoke` / `TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult`。该对象是 disabled-by-default、opt-in 的 attach runtime smoke report scaffold，只复制 `OptInEnabled`、`FullPackageConsumerReport`、`AttachGuardReady`、`NativeVTableReady`、`BorrowedDebugTensorRuntimeReady`、`CallbackInvocationReady`、`AttachAttempted`、`AttachSucceeded`、`DetachAttempted`、`DetachSucceeded`、`RollbackAttempted`、`RollbackSucceeded`、`NativeVTableInstalled`、`ProcessDebugTensorInvoked`、`InvocationCount`、`AllocationCount`、`ReleaseCount`、`FailureCount`、`InFlightCallbackCount`、`ReportPointerFree`、`CanAttemptRuntimeProof`、`CanPromoteRealCallbackRuntime` 和 `ReasonRuntimeProofStillBlocked`。当前默认 `RuntimeEvidenceKind=runtime-smoke-skipped`；opt-in 也只能在前置条件不完整时报告 `runtime-smoke-blocked`、`runtime-smoke-attempted` 或 `runtime-smoke-failed`。这些状态都是 not proof，不能被 `Test-RuntimePackageReadiness.ps1` 或 package consumer parser 解释为 `real-callback-runtime`。

`debug-listener-process-debug-tensor-callback-trampoline` 对应 public surface `TensorRtDebugListenerProcessDebugTensorCallbackTrampoline` / `TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult` / `TensorRtDebugTensorMetadataSnapshot`。该对象只汇总 callback trampoline shape、no-throw entry、exception/status mapping、in-flight accounting、detach-before-release、borrowed debug tensor metadata copy 和 runtime smoke report，不安装 native vtable，不启用默认 non-null attach，也不伪造 `processDebugTensor` invocation。当前 `RuntimeEvidenceKind=callback-trampoline-shape`，并必须保持 `ProcessDebugTensorRuntimeReady=False`、`NativeVTableInstalled=False`、`ProcessDebugTensorInvoked=False`、`InvocationCount=0`、`CanPromoteRealCallbackRuntime=False` 和 `IsRealCallbackRuntimeProof=False`。

`debug-listener-real-callback-runtime-proof` 对应 public surface `TensorRtDebugListenerRealCallbackRuntimeProof` / `TensorRtDebugListenerRealCallbackRuntimeProofResult`。该对象是最终 proof promotion gate，只集中检查 `OptInEnabled`、`FullPackageConsumerReport`、`RuntimeSmokeReady`、`TrampolineShapeReady`、attach/detach/rollback、`NativeVTableInstalled`、`ProcessDebugTensorInvoked`、`InvocationCount>0`、`FailureCount=0`、`InFlightCallbackCount=0`、`BorrowedDebugTensorMetadataCopied`、`PointerFreeSurfaceReady` 和 `ProcessDebugTensorRuntimeReady`。默认 `RuntimeEvidenceKind=runtime-smoke-skipped`，opt-in 后未满足条件时报告 `real-callback-runtime-blocked` 或 `attempted-no-invocation`。这些状态仍是 not proof；只有 full package consumer 真实 callback invocation 让全部字段满足时才能输出 `RuntimeEvidenceKind=real-callback-runtime`。

`debug-listener-callback-proof-gap-report` 对应 public surface `TensorRtDebugListenerCallbackProofGapReport` / `TensorRtDebugListenerCallbackProofGapReportResult`。该对象只把 proof-attempt preflight、real non-null attach runtime smoke、callback trampoline 和 final proof gate 汇总为 pointer-free 缺口图，输出 `RuntimeEvidenceKind=proof-gap-report`、`NonNullAttachStillDisabled`、`NativeAttachEntryReady`、`NativeVTableInstallBlocked`、`NoThrowCallbackEntryReady`、`ExceptionStatusMappingReady`、`InFlightAccountingReady`、`BorrowedDebugTensorMetadataCopied`、`DetachRollbackReady`、`ProcessDebugTensorRuntimeInvoked`、`FullPackageConsumerRuntimeProofReady`、`GapReasonCount`、`PrimaryGapReason`、`RuntimeProofBlockerCategory`、`PackageConsumerRuntimeProofRequired`、`RuntimeInvocationRequired`、`EvidenceSource`、`NextOwnerAction` 和 `CanPromoteRealCallbackRuntime=False`。它不能替代真实 TensorRT callback runtime proof；package consumer 与 readiness parser 必须把 `proof-gap-report` 和 `NoNonProofCallbackRuntimeMarker` 当作非证明降级条件。

`callback-owner-closure-matrix` 对应 public surface `TensorRtCallbackOwnerClosureMatrix` / `TensorRtCallbackOwnerClosureMatrixResult` / `TensorRtCallbackOwnerClosureMatrixRow`。该对象只把 `GpuAllocator`、`GpuAsyncAllocator`、`OutputAllocator`、`DebugListener` 和 `StreamReaderWriter` 的设计门与历史 deferred 缺口汇总为 pointer-free gap map，并通过 `Test-BridgePackageConsumer.ps1` 与 `Test-RuntimePackageReadiness.ps1` 进入 package-consumer/readiness 证据链。它必须保持 `RuntimeEvidenceKind=closure-matrix`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`ClosureReadyFamilyCount=0`、`RuntimeProofAttemptReadyFamilyCount=0`、`PackageConsumerRuntimeProofReadyFamilyCount=0`、`RuntimeProofBlocked=True` 和 `DeferredRowsStillRequired=True`。这些聚合计数不会自动消费后来新增的子接口实机证据；例如 `IStreamReaderV2` 已有独立的 TensorRT 10.11 本地双包 runtime record，但 legacy reader、writer 和 TRT11 实机仍未完成。矩阵不能作为真实 TensorRT callback runtime proof，也不能替代 full package consumer 的真实 invocation、detach/release、failure/in-flight 归零证据。

`Test-PackageConsumer.ps1` 的 full package consumer report 还必须写入 `RealCallbackRuntimeEvidence` 对象，readiness 优先消费这个对象，旧报告才回退到 `SmokeOutputLines` 扫描。

`RealCallbackRuntimeEvidence` 至少包含：

| 字段 | 要求 |
| --- | --- |
| `Status` | `not-present`、`blocked-by-cuda-driver`、`blocked-by-application-control`、`blocked`、`incomplete` 或 `ready`。 |
| `EvidenceKind` | `not-present`、`incomplete-real-callback-runtime` 或 `real-callback-runtime`。 |
| `RuntimeEvidenceKind` | 与 `EvidenceKind` 保持一致。 |
| `RequiredSmokeMarkers` | schema 要求的完整 marker 列表。 |
| `MissingSmokeMarkers` | `incomplete` 时列出缺失字段。 |
| `MatchedSmokeLines` | 复制出的 runtime callback 相关输出行。 |
| `IsRealCallbackRuntimeProof` | 只有 `Status=ready`、`EvidenceKind=real-callback-runtime`、`RuntimeEvidenceKind=real-callback-runtime` 且 full package consumer smoke 成功时才允许为 true。 |
| `Diagnostic` | copied diagnostic，用于区分 not-present、blocked 和 incomplete。 |

readiness 会把 full package consumer report 中的嵌套对象展平为以下审计字段，供 `realCallbackRuntimeEvidence` 和 Markdown summary 使用：

| 展平字段 | 来源 |
| --- | --- |
| `callbackRuntimeEvidenceStatus` | `RealCallbackRuntimeEvidence.Status` |
| `callbackRuntimeEvidenceKind` | `RealCallbackRuntimeEvidence.EvidenceKind` |
| `callbackRuntimeRuntimeEvidenceKind` | `RealCallbackRuntimeEvidence.RuntimeEvidenceKind` |
| `callbackRuntimeRequiredMarkers` | `RealCallbackRuntimeEvidence.RequiredSmokeMarkers` |
| `callbackRuntimeMissingMarkers` | `RealCallbackRuntimeEvidence.MissingSmokeMarkers` |
| `callbackRuntimeMatchedLines` | `RealCallbackRuntimeEvidence.MatchedSmokeLines` |
| `callbackRuntimeIsProof` | `RealCallbackRuntimeEvidence.IsRealCallbackRuntimeProof` |
| `callbackRuntimeDiagnostic` | `RealCallbackRuntimeEvidence.Diagnostic` |

在没有真实 smoke 前：

- `realCallbackRuntimeEvidenceSchema.status=schema-ready`
- `realCallbackRuntimeEvidence.status=not-present`
- `realCallbackRuntimeEvidence.evidenceKind=not-present`
- `realCallbackRuntimeEvidence.isRealCallbackRuntimeProof=false`

如果 smoke 输出 `EvidenceKind=real-callback-runtime` 但缺少必需字段：

- `realCallbackRuntimeEvidence.status=incomplete`
- `missingSmokeMarkers` 必须列出缺失字段。
- `isRealCallbackRuntimeProof=false`

如果 smoke 被环境阻塞但没有输出真实 runtime marker：

- CUDA driver/runtime 不兼容：`RealCallbackRuntimeEvidence.Status=blocked-by-cuda-driver`。
- Windows application control 阻塞：`RealCallbackRuntimeEvidence.Status=blocked-by-application-control`。
- 普通 smoke 失败且无 runtime marker：`RealCallbackRuntimeEvidence.Status=blocked`。
- 以上状态都必须保持 `IsRealCallbackRuntimeProof=false`，不得解释为 API 缺失或 callback 已启用。

只有所有必需字段都存在，`EvidenceKind=real-callback-runtime`、`RuntimeEvidenceKind=real-callback-runtime`、`RealCallbackRuntime=True`、`IsRealCallbackRuntimeProof=True` 同时存在，且 full package consumer smoke 成功执行，才能置为：

- `realCallbackRuntimeEvidence.status=ready`
- `realCallbackRuntimeEvidence.evidenceKind=real-callback-runtime`
- `realCallbackRuntimeEvidence.isRealCallbackRuntimeProof=true`

## 禁止混淆

readiness 必须保留这些负向约束：

- bridge-only report 不能产生 `real-callback-runtime`。
- `SmokeResult=passed` 只能说明 package consumer 程序执行成功，不自动说明 callback runtime 已触发。
- `allocator-owner-dry-run-diagnostics`、`allocator-owner-native-dry-run-controls`、`allocator-owner-state-ledger-dry-run-controls` 仍属于 `dry-run`。
- `allocator-owner-internal-runtime-prototype` 仍属于 `internal-runtime-prototype`。即使 dedicated smoke 输出 `EvidenceKind=allocator-owner-internal-runtime-prototype`、`RealCallbackRuntime=False`、`CallbackKind=sync-allocator-prototype`、`InFlightCallbackCount`、`ReleaseHookCount`、`CallbackStatePinned`、`DelegatePinned`、`DisposeRequested`、`LastStatus` 和 `LastDiagnostic`，也只是 owner 生命周期与 no-throw/exception-to-status 诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `allocator-owner-ledger-safety-gate` 仍属于 `ledger-safety-gate`。即使 dedicated smoke 输出 `EvidenceKind=allocator-owner-ledger-safety-gate`、`RuntimeEvidenceKind=ledger-safety-gate`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`ManagedKeepAliveReady`、`DisposeReleaseReady`、`NativeLedgerDesignReady`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是 copied diagnostics 与阻塞项汇总，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `output-allocator-internal-runtime-gate` 仍属于 `internal-runtime-gate`。即使 dedicated smoke 输出 `EvidenceKind=output-allocator-internal-runtime-gate`、`RealCallbackRuntime=False`、`CallbackKind=output-allocator-prototype`、`NotifyShapeCount`、`ReallocateOutputCount`、`ShapeRank`、`OutputBufferPointerExposed=False` 和 `LastDiagnostic`，也只是 copied diagnostics，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `output-allocator-callback-owner-design` 仍属于 `owner-design-gate`。即使 dedicated smoke 输出 `EvidenceKind=output-allocator-callback-owner-design`、`RuntimeEvidenceKind=not-present`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`NativeLedgerAvailable`、`StateTransitionCount`、`LedgerAllocationCount` 和 `LedgerReleaseCount`，也只是 owner wrapper 与 ledger intent 诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `output-allocator-attach-detach-design-gate` 仍属于 `design-gate`。即使 dedicated smoke 输出 `EvidenceKind=output-allocator-attach-detach-design-gate`、`RuntimeEvidenceKind=design-gate`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`AttachControlAvailable=False`、`DetachClearControlAvailable=True`、`LineSpecificAttachDetachReady=False`、`NativeVTableReady=False`、`OutputBufferOwnershipRuntimeReady=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是 attach/detach 生命周期门禁诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `output-buffer-ownership-safety-gate` 仍属于 `design-gate`。即使 dedicated smoke 输出 `EvidenceKind=output-buffer-ownership-safety-gate`、`RuntimeEvidenceKind=design-gate`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`SafetyGateReady=True`、`CurrentMemoryReusePolicyReady=False`、`BorrowedPointerEscapeBlocked=True`、`OwnedDevicePointerReleasePolicyReady=False`、`ShapeNotificationOrderingReady=False`、`ReallocateOutputRuntimeReady=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是 output buffer ownership 安全门禁诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `output-allocator-runtime-proof-precheck` 仍属于 `runtime-gate-precheck`。即使 dedicated smoke 输出 `EvidenceKind=output-allocator-runtime-proof-precheck`、`RuntimeEvidenceKind=runtime-gate`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`AttachDetachDesignGateReady=True`、`AttachControlAvailable=False`、`DetachClearControlAvailable=True`、`NativeVTableReady=False`、`OutputBufferOwnershipRuntimeReady=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是前置条件与阻塞项诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-callback-owner-design` 仍属于 `owner-design-gate`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-callback-owner-design`、`RuntimeEvidenceKind=not-present`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`ProcessDebugTensorCount`、`DebugTensorPointerExposed=False`、`DebugTensorPointerProduced=False` 和 `BorrowedDebugTensorPointerEscaped=False`，也只是 DebugListener owner wrapper 与 debug tensor metadata copy-out 诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-attach-detach-design-gate` 仍属于 `design-gate`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-attach-detach-design-gate`、`RuntimeEvidenceKind=design-gate`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`AttachControlAvailable=False`、`DetachClearControlAvailable=True`、`LineSpecificAttachDetachReady=False`、`NativeVTableReady=False`、`BorrowedDebugTensorLifetimeReady=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是 attach/detach 生命周期门禁诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-borrowed-tensor-safety-gate` 仍属于 `design-gate`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-borrowed-tensor-safety-gate`、`RuntimeEvidenceKind=design-gate`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`SafetyGateReady=True`、`BorrowedDebugTensorPointerEscapeBlocked=True`、`BorrowedDebugTensorLifetimeReady=False`、`BorrowedDebugTensorDataLifetimeReady=False`、`ProcessDebugTensorRuntimeReady=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是 borrowed tensor/data lifetime 安全门禁诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-attach-vtable-safety-gate` 仍属于 `design-gate`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-attach-vtable-safety-gate`、`RuntimeEvidenceKind=design-gate`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`SafetyGateReady=True`、`AttachControlAvailable=False`、`StableNativeOwnerAddressReady=False`、`NoThrowNativeVTableReady=False`、`ExceptionToStatusMappingReady=False`、`ProcessDebugTensorRuntimeReady=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是 attach/vtable 安全门禁诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-native-attach-nothrow-preflight` 仍属于 `preflight`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-native-attach-nothrow-preflight`、`RuntimeEvidenceKind=preflight`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`PreflightReady=True`、`NativeAttachEntryLocated=False`、`NativeDetachEntryLocated=True`、`NoThrowVTableDesignReady=False`、`ExceptionToStatusMappingDesignReady=False`、`CanImplementNativeAttach=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是 native attach/no-throw 前置条件诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-native-owner-address-design-gate` 仍属于 `design-gate`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-native-owner-address-design-gate`、`RuntimeEvidenceKind=design-gate`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`DesignGateReady=True`、`NativeAttachNoThrowPreflightReady=True`、`StableNativeOwnerAddressReady=False`、`StableNativeOwnerAddressDesignReady=False`、`NativeOwnerNonCopyableReady=False`、`NoThrowNativeDestructorReady=False`、`NativeOwnerLifecycleReady=False`、`CanImplementNativeAttach=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是 native owner address 生命周期门禁诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-native-nothrow-vtable-design-gate` 仍属于 `design-gate`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-native-nothrow-vtable-design-gate`、`RuntimeEvidenceKind=design-gate`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`DesignGateReady=True`、`NativeOwnerAddressDesignGateReady=True`、`NativeAttachNoThrowPreflightReady=True`、`NoThrowNativeDestructorReady=False`、`NoThrowVTableDesignReady=False`、`ExceptionToStatusMappingDesignReady=False`、`NativeVTableTrampolineReady=False`、`CallbackExceptionCaptureReady=False`、`CallbackStatusMappingReady=False`、`CallbackInFlightAccountingReady=False`、`CanImplementNativeAttach=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是 native no-throw vtable 门禁诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-native-attach-entry-design-gate` 仍属于 `design-gate`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-native-attach-entry-design-gate`、`RuntimeEvidenceKind=design-gate`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`DesignGateReady=True`、`NativeNoThrowVTableDesignGateReady=True`、`NativeDetachEntryLocated=True`、`NativeAttachEntryLocated=False`、`LineSpecificAttachEntryDesignReady=False`、`AttachEntryNoThrowReady=False`、`AttachEntryVersionGuardReady=False`、`AttachEntryOwnershipReady=False`、`DetachBeforeReleaseReady=False`、`CanImplementNativeAttach=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是 native attach entry 门禁诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-native-detach-before-release-design-gate` 仍属于 `design-gate`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-native-detach-before-release-design-gate`、`RuntimeEvidenceKind=design-gate`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`DesignGateReady=True`、`NativeAttachEntryDesignGateReady=True`、`NativeDetachEntryLocated=True`、`NativeAttachEntryLocated=False`、`DetachBeforeReleaseReady=False`、`ReleaseHookOrderingReady=False`、`DisposeIdempotencyReady=False`、`InFlightDrainBeforeReleaseReady=False`、`CallbackStateUnpinAfterDetachReady=False`、`DelegateUnpinAfterDetachReady=False`、`CanImplementNativeAttach=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是 native detach-before-release 门禁诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-native-owner-lifecycle-dry-run` 仍属于 `dry-run`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-native-owner-lifecycle-dry-run`、`RuntimeEvidenceKind=dry-run`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`DryRunReady=True`、`NativeDetachBeforeReleaseDesignGateReady=True`、`StableNativeOwnerIdentityReady=False`、`NativeOwnerNonCopyableReady=False`、`ReleaseHookOrderingReady=False`、`DisposeIdempotencyReady=False`、`InFlightDrainBeforeReleaseReady=False`、`CallbackStateUnpinAfterDetachReady=False`、`DelegateUnpinAfterDetachReady=False`、`NoThrowNativeDestructorReady=False`、`NativeOwnerLifecycleReady=False`、`CanImplementNativeAttach=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是 native owner lifecycle dry-run 诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-native-attach-entry-runtime-scaffold` 仍属于 `scaffold`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-native-attach-entry-runtime-scaffold`、`RuntimeEvidenceKind=scaffold`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`RuntimeScaffoldReady=True`、`NativeOwnerLifecycleDryRunReady=True`、`AttachEntryParameterShapeReady=True`、`AttachEntryVersionGuardReady=True`、`AttachEntryNoThrowBoundaryReady=True`、`AttachEntryOwnershipDiagnosticsReady=True`、`NativeAttachEntryLocated=False`、`StableNativeOwnerIdentityReady=False`、`NativeOwnerNonCopyableReady=False`、`NoThrowNativeDestructorReady=False`、`CanImplementNativeAttach=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是 native attach entry runtime scaffold 诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-native-attach-entry-minimal-safety` 仍属于 `minimal-safety`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-native-attach-entry-minimal-safety`、`RuntimeEvidenceKind=minimal-safety`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`MinimalSafetyReady=True`、`RuntimeScaffoldReady=True`、`LifecycleGateReady=True`、`NativeAttachEntryLocated=True`、`SetDebugListenerNonNullEnabled=False`、`NativeAttachWouldBeBlocked=True`、`CanImplementNativeAttach=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是 scoped source-visible attach entry shape 诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-native-owner-stable-identity` 仍属于 `identity-gate`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-native-owner-stable-identity`、`RuntimeEvidenceKind=identity-gate`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`NativeAttachEntryRuntimeScaffoldReady=True`、`StableNativeOwnerIdentityReady=True`、`OwnerIdentityDiagnosticsReady=True`、`OwnerIdentityPointerFree=True`、`NativeAttachEntryLocated=False`、`NativeOwnerNonCopyableReady=False`、`NoThrowNativeDestructorReady=False`、`NativeOwnerLifecycleReady=False`、`CanImplementNativeAttach=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是 pointer-free owner id / diagnostic identity 诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-native-owner-noncopyable-storage` 仍属于 `storage-gate`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-native-owner-noncopyable-storage`、`RuntimeEvidenceKind=storage-gate`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`NativeOwnerStableIdentityReady=True`、`OwnerIdentityDiagnosticsReady=True`、`OwnerIdentityPointerFree=True`、`NativeOwnerNonCopyableReady=True`、`NativeOwnerCopyBlocked=True`、`NativeOwnerMoveBlocked=True`、`NativeOwnerAddressExposed=False`、`NativeOwnerPointerProduced=False`、`NativeAttachEntryLocated=False`、`NoThrowNativeDestructorReady=False`、`NativeOwnerLifecycleReady=False`、`CanImplementNativeAttach=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是 source-visible no-copy/no-move storage scaffold 诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-native-nothrow-destructor` 仍属于 `destructor-gate`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-native-nothrow-destructor`、`RuntimeEvidenceKind=destructor-gate`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`NativeOwnerNonCopyableStorageReady=True`、`NativeOwnerNonCopyableReady=True`、`DestructorNoThrowScaffoldReady=True`、`DestructorExceptionEscapeBlocked=True`、`DestructorAddressExposed=False`、`DestructorPointerProduced=False`、`NoThrowNativeDestructorReady=True`、`NativeOwnerLifecycleReady=False`、`CanImplementNativeAttach=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是 source-visible no-throw destructor scaffold 诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-native-owner-lifecycle-gate` 仍属于 `lifecycle-gate`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-native-owner-lifecycle-gate`、`RuntimeEvidenceKind=lifecycle-gate`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`NativeNoThrowDestructorGateReady=True`、`ManagedDisposeSnapshotReady=True`、`LifecycleScaffoldReady=True`、`ReleaseHookOrderingGateReady=True`、`DisposeIdempotencyGateReady=True`、`InFlightDrainGateReady=True`、`CallbackStateUnpinAfterDetachGateReady=True`、`DelegateUnpinAfterDetachGateReady=True`、`LifecycleAddressExposed=False`、`LifecyclePointerProduced=False`、`LifecycleGateReady=True`、`NativeOwnerLifecycleReady=False`、`NativeVTableDesignReady=False`、`CanImplementNativeAttach=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是 source-visible lifecycle scaffold 诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-native-attach-bridge-shape-gate` 仍属于 `attach-bridge-shape-gate`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-native-attach-bridge-shape-gate`、`RuntimeEvidenceKind=attach-bridge-shape-gate`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`NativeOwnerLifecycleGateReady=True`、`AttachBridgeShapeReady=True`、`AttachBridgeNoThrowBoundaryReady=True`、`AttachBridgeVersionGuardReady=True`、`AttachBridgeOwnershipDiagnosticsReady=True`、`AttachBridgePointerFree=True`、`SetDebugListenerNonNullEnabled=False`、`NonNullAttachStillDisabled=True`、`NativeAttachEntryLocated=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是 attach bridge shape scaffold 诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-exception-status-mapping-gate` 仍属于 `exception-status-gate`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-exception-status-mapping-gate`、`RuntimeEvidenceKind=exception-status-gate`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`AttachBridgeShapeGateReady=True`、`ManagedCallbackExceptionCaptureReady=True`、`NativeCallbackExceptionCaptureReady=True`、`CallbackStatusMappingGateReady=True`、`ExceptionEscapeBlocked=True`、`DiagnosticCopyReady=True`、`MappingAddressExposed=False`、`MappingPointerProduced=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是 exception/status mapping scaffold 诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-inflight-accounting-gate` 仍属于 `inflight-accounting-gate`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-inflight-accounting-gate`、`RuntimeEvidenceKind=inflight-accounting-gate`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`ExceptionStatusMappingGateReady=True`、`CallbackEnterAccountingGateReady=True`、`CallbackLeaveAccountingGateReady=True`、`CallbackInFlightNeverNegativeReady=True`、`ReleaseAfterDrainGateReady=True`、`CallbackStateUnpinAfterDrainGateReady=True`、`AccountingAddressExposed=False`、`AccountingPointerProduced=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是 in-flight accounting scaffold 诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-native-nothrow-vtable-scaffold-gate` 仍属于 `vtable-scaffold-gate`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-native-nothrow-vtable-scaffold-gate`、`RuntimeEvidenceKind=vtable-scaffold-gate`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`NativeAttachBridgeShapeGateReady=True`、`ExceptionStatusMappingGateReady=True`、`InFlightAccountingGateReady=True`、`NoThrowVTableScaffoldReady=True`、`VTableDestructorNoThrowReady=True`、`ProcessDebugTensorCallbackStubNoThrowReady=True`、`ExceptionEscapeBlocked=True`、`CallbackExceptionCaptureGateReady=True`、`CallbackStatusMappingGateReady=True`、`CallbackInFlightAccountingGateReady=True`、`VTableAddressExposed=False`、`VTablePointerProduced=False`、`NativeVTableDesignReady=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是 no-throw vtable scaffold 诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-nothrow-vtable-callback-stub` 仍属于 `callback-stub-gate`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-nothrow-vtable-callback-stub`、`RuntimeEvidenceKind=callback-stub-gate`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`CallbackStubGateReady=True`、`CallbackMetadataCopyReady=True`、`DebugTensorPointerExposed=False`、`DebugTensorDataPointerExposed=False`、`SetDebugListenerNonNullEnabled=False`、`NativeAttachWouldBeBlocked=True`、`NativeVTableInstalled=False`、`CanInstallNativeVTable=False`、`CanCallProcessDebugTensorRuntime=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是 no-throw callback stub 诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-borrowed-debug-tensor-metadata-runtime-gate` 仍属于 `borrowed-debug-tensor-metadata-gate`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-borrowed-debug-tensor-metadata-runtime-gate`、`RuntimeEvidenceKind=borrowed-debug-tensor-metadata-gate`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`MetadataGateReady=True`、`TensorNameCopied=True`、`TensorTypeCopied=True`、`TensorLocationCopied=True`、`TensorShapeCopied=True`、`TensorFlagsCopied=True`、`BorrowedDebugTensorMetadataCopyReady=True`、`BorrowedDebugTensorPointerEscapeBlocked=True`、`BorrowedDebugTensorDataPointerEscapeBlocked=True`、`BorrowedDebugTensorLifetimeReady=False`、`BorrowedDebugTensorDataLifetimeReady=False`、`CanCallProcessDebugTensorRuntime=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是 copied metadata gate 诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-native-vtable-install-preflight` 仍属于 `native-vtable-install-preflight`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-native-vtable-install-preflight`、`RuntimeEvidenceKind=native-vtable-install-preflight`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`NativeVTableInstallPreflightReady=True`、`VTableInstallShapeReady=True`、`VTableInstallPointerFree=True`、`SetDebugListenerNonNullEnabled=False`、`NativeVTableInstalled=False`、`NativeVTableInstallRuntimeReady=False`、`CanEnableSetDebugListenerNonNull=False`、`CanInstallNativeVTable=False`、`CanCallProcessDebugTensorRuntime=False` 和 `RuntimeProofBlocked=True`，也只是 native vtable install preflight 诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-native-owner-vtable-install-experiment` 仍属于 `native-owner-vtable-install-experiment`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-native-owner-vtable-install-experiment`、`RuntimeEvidenceKind=native-owner-vtable-install-experiment`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`ExperimentShapeReady=True`、`InstallAttemptGuardReady=True`、`NonNullAttachEnabled=False`、`RuntimeProofEnabled=False`、`NativeVTableInstallAttempted=False`、`NativeVTableInstalled=False`、`RollbackReady=True`、`DetachBeforeReleaseReady=True`、`FailureStatusMappingReady=True`、`PointerFree=True`、`CanEnableSetDebugListenerNonNull=False`、`CanInstallNativeVTable=False`、`CanCallProcessDebugTensorRuntime=False` 和 `RuntimeProofBlocked=True`，也只是 native owner/vtable install experiment 诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-runtime-proof-precheck` 仍属于 `runtime-gate-precheck`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-runtime-proof-precheck`、`RuntimeEvidenceKind=runtime-gate`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是前置条件与阻塞项诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-runtime-proof-attempt-preflight` 仍属于 `runtime-proof-attempt-preflight`。即使 dedicated smoke 输出 `EvidenceKind=debug-listener-runtime-proof-attempt-preflight`、`RuntimeEvidenceKind=runtime-proof-attempt-preflight`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`CanEnableSetDebugListenerNonNull=False`、`CanInstallNativeVTable=False`、`CanCallProcessDebugTensorRuntime=False`、`CanPromoteRealCallbackRuntime=False` 和 `RuntimeProofBlocked=True`，也只是 proof-attempt 准入诊断，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `debug-listener-real-non-null-attach-runtime-smoke` 仍属于 runtime smoke attempt/report scaffold。即使 full package consumer 输出 `DebugListenerRealNonNullAttachRuntimeSmoke=`、`RuntimeEvidenceKind=runtime-smoke-skipped`、`RuntimeEvidenceKind=runtime-smoke-blocked`、`RuntimeEvidenceKind=runtime-smoke-attempted` 或 `RuntimeEvidenceKind=runtime-smoke-failed`，并包含 `AttachSucceeded=False`、`NativeVTableInstalled=False`、`ProcessDebugTensorInvoked=False`、`InvocationCount=0`、`ReportPointerFree=True`、`CanPromoteRealCallbackRuntime=False` 和 `RuntimeProofBlocked=True`，也只是 runtime smoke 非证明状态，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `callback-owner-closure-matrix` 仍属于 `closure-matrix`。即使 bridge package consumer 和 runtime readiness 已识别 `TensorRtCallbackOwnerClosureMatrix.Evaluate`、`FamilyCount=5`、`ClosureReadyFamilyCount=0`、`RuntimeProofAttemptReadyFamilyCount=0`、`PackageConsumerRuntimeProofReadyFamilyCount=0`、`RuntimeProofBlocked=True` 和 `DeferredRowsStillRequired=True`，也只是 owner 缺口汇总，not proof，不得设置 `isRealCallbackRuntimeProof=true`。
- `callback-interface-info-safe-controls` 与 `execution-context-callback-state-snapshot` 仍属于 `copied-state`。
- `TensorRtCallbackAllocatorReadinessSnapshot` 与 `CallbackAllocatorReadinessSnapshot=` 仍属于 `RuntimeEvidenceKind=managed-readiness`。它们只聚合 logger/profiler/progress monitor、allocator ledger、output allocator、debug listener 的 managed wrapper readiness，不代表 TensorRT 真实调用过 callback，也不能替代 package-consumer-runtime proof。
- `managed-readiness-only`、`precheck-only`、`dry-run-only` 与 `schema-only` 只能作为阶段门禁或证据采集准备项；没有兼容主机、真实 runtime smoke、真实 callback invocation、日志 hash 和 validator 通过时，必须保持 `isRealCallbackRuntimeProof=false`。
- 任何 schema-only marker 都不能作为真实 TensorRT callback 已启用的证据。

## Deferred rows

在 `realCallbackRuntimeEvidence.isRealCallbackRuntimeProof=true` 之前，以下 rows 必须继续保留 direct deferred：

- `IGpuAllocator::allocate`
- `IGpuAllocator::free`
- `IGpuAllocator::deallocate`
- `IGpuAllocator::reallocate`
- `IGpuAsyncAllocator::allocateAsync`
- `IGpuAsyncAllocator::deallocateAsync`
- `IOutputAllocator::notifyShape`
- `IOutputAllocator::reallocateOutput`
- `IDebugListener::processDebugTensor`
