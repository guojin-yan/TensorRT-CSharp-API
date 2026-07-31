# DebugListener Real Non-Null Attach Runtime Smoke

> 状态：runtime-smoke-ready
> readiness marker：`debug-listener-real-non-null-attach-runtime-smoke`
> runtime evidence：`RuntimeEvidenceKind=runtime-smoke-skipped` / `runtime-smoke-blocked` / `runtime-smoke-attempted` / `runtime-smoke-failed`
> 当前结论：`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`

> 2026-07-31 更新：本页原有 `TensorRtDebugListenerRealNonNullAttachRuntimeSmoke` 类型仍是历史 report scaffold，
> 因而默认字段保持 false。真实 owner 路径已经由 `CallbackAllocatorSafeControlsSmokeRunner
> --enable-debug-listener-runtime-smoke` 执行，并在 TRT10.11/CUDA12.9 得到 non-null attach、一次真实
> `processDebugTensor` invocation 和成功 detach。不得再把历史 scaffold 的固定 false 推导成 native vtable 尚未实现。

`debug-listener-real-non-null-attach-runtime-smoke` 是 disabled-by-default、opt-in 的 runtime smoke attempt/report scaffold。它位于 `debug-listener-runtime-proof-attempt-preflight` 之后，用来把下一步真实 `setDebugListener(non-null)` 尝试所需的前置条件、尝试状态、回滚状态和 package-consumer evidence 统一复制成 pointer-free 结果。

该对象当前不安装 native `IDebugListener` vtable，不启用默认 non-null attach，不调用 `IDebugListener::processDebugTensor`，也不暴露 native owner、vtable、debug tensor 或 data pointer。

源码 owner 已按职责拆分：evaluation 与 diagnostic/blocker 构造位于
`TensorRtDebugListenerRealNonNullAttachRuntimeSmoke.cs`，pointer-free report 位于
`TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.cs`。readiness 与源码测试必须组合读取这两个文件。

## Public Surface

- `TensorRtDebugListenerRealNonNullAttachRuntimeSmoke`
- `TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult`

关键字段：

- `RuntimeEvidenceKind=runtime-smoke-skipped`
- `OptInEnabled`
- `FullPackageConsumerReport`
- `AttachGuardReady`
- `NativeVTableReady`
- `BorrowedDebugTensorRuntimeReady`
- `CallbackInvocationReady`
- `AttachAttempted`
- `AttachSucceeded=False`
- `DetachAttempted`
- `DetachSucceeded=False`
- `RollbackAttempted`
- `RollbackSucceeded`
- `NativeVTableInstalled=False`
- `ProcessDebugTensorInvoked=False`
- `InvocationCount=0`
- `AllocationCount=0`
- `ReleaseCount=0`
- `FailureCount`
- `InFlightCallbackCount=0`
- `LastStatus`
- `LastDiagnostic`
- `ReportPointerFree=True`
- `CanAttemptRuntimeProof`
- `CanPromoteRealCallbackRuntime=False`
- `RuntimeProofBlocked=True`
- `ReasonRuntimeProofStillBlocked`
- `DeferredRowsStillRequired`

默认调用保持 `runtime-smoke-skipped`。即使设置 opt-in，在 attach/vtable/callback/full package consumer 证据不完整时也只能得到 `runtime-smoke-blocked` 或 `runtime-smoke-attempted`，不能升级为 `real-callback-runtime`。

## Native Scaffold

native 侧提供 `debug_listener_real_non_null_attach_runtime_smoke.inc`，其中 `DebugListenerRealNonNullAttachRuntimeSmokeAttempt final` 固定：

- copy/move deleted
- destructor noexcept
- `configure_api_line`
- `configure_opt_in`
- `configure_prerequisites`
- `can_attempt_attach`
- `attach_attempted`
- `attach_succeeded`
- `detach_attempted`
- `detach_succeeded`
- `rollback_attempted`
- `rollback_succeeded`
- `native_vtable_installed`
- `process_debug_tensor_invoked`
- `invocation_count`
- `failure_count`
- `in_flight_callback_count`
- `report_pointer_free`
- `can_promote_real_callback_runtime`

这些函数只提供历史 source-visible scaffold 和静态 no-throw 约束，因此其
`attach_succeeded()`、`native_vtable_installed()`、`process_debug_tensor_invoked()` 仍固定为 false。实际 vtable owner、
attach/detach C ABI 与 copied snapshot 位于独立的 `debug_listener_callback_owner.inc`，不能混用两组结果。

## Real Smoke Output

显式 opt-in 的真实 runner 构建最小 identity network，将 `debug_output` 标记为 build-time debug tensor，绑定输入/输出
CUDA memory，安装 managed owner，启用 runtime tensor debug state 后执行 enqueue。成功输出要求：

- `DebugListenerRealRuntime=Passed`
- `NativeVTableInstalled=True`
- `ProcessDebugTensorInvoked=True`
- `InvocationCount>0`
- `FailureCount=0`
- `InFlightCallbackCount=0`
- `MetadataCopied=True`
- `BorrowedPointerExposed=False`
- `DetachCount>0`
- `IsRealCallbackRuntimeProof=True`

该 marker 是 source-tree local runtime proof。历史 scaffold marker 仍可同时输出，用于验证旧 promotion gate 不会在缺少
package report 时误晋级；二者的 evidence type 必须分开解析。

最新 DLL 的 TRT10.11/CUDA12.9 复验得到一次真实 callback、零 failure、零 in-flight、一次 detach，且
`BorrowedPointerExposed=False`。native owner 的 drain wait 已用同一状态锁同步计数归零；managed context 在 native
context handle 销毁后才解除 owner borrow，嵌套 managed callback 使用 depth 保护，避免 bool 提前复位。

TRT11.0/CUDA12.9 也通过同一 owner 路径，得到相同的一次 invocation、零 failure/in-flight 与一次 detach。TRT11 使用
`--debug-listener-runtime-smoke-only`，避免旧综合 execution-context callback-state snapshot 的独立 SEH 诊断提前返回；
该模式明确输出 `Mode=DebugListenerRuntimeSmokeOnly`，不能用于声称旧综合 safe controls 已通过。

## Smoke And Readiness

`CallbackAllocatorSafeControlsSmokeRunner` 和 full package consumer smoke 输出：

- `DebugListenerRealNonNullAttachRuntimeSmoke=...`
- `EvidenceKind=debug-listener-real-non-null-attach-runtime-smoke`
- `RuntimeEvidenceKind=runtime-smoke-skipped`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `OptInEnabled`
- `FullPackageConsumerReport`
- `AttachSucceeded=False`
- `NativeVTableInstalled=False`
- `ProcessDebugTensorInvoked=False`
- `ReportPointerFree=True`
- `CanPromoteRealCallbackRuntime=False`
- `RuntimeProofBlocked=True`

`Test-RuntimePackageReadiness.ps1` 在 JSON/Markdown 中报告 `debugListenerRealNonNullAttachRuntimeSmoke`，并继续要求 `IDebugListener::processDebugTensor` deferred row 保留。

后续 `debug-listener-process-debug-tensor-callback-trampoline` 会消费本报告，并把它与 `callback-stub-gate`、`borrowed-debug-tensor-metadata-gate` 合并成 `RuntimeEvidenceKind=callback-trampoline-shape`。该合并结果仍然是 not proof，只有 full package consumer 的真实 callback invocation 才能提升为 `real-callback-runtime`。

## Non-Proof Boundary

以下状态都不是 proof：

- `runtime-smoke-skipped`
- `runtime-smoke-blocked`
- `runtime-smoke-attempted`
- `runtime-smoke-failed`

这些状态只能说明 runtime smoke report 已被生成，不能说明：

- `setDebugListener(non-null)` 已启用。
- native `IDebugListener` vtable 已安装。
- borrowed debug tensor 或 data pointer 生命周期已由真实 TensorRT callback 证明。
- `IDebugListener::processDebugTensor` 已被 TensorRT 调用。
- `real-callback-runtime` 可以被 promotion。

只有 full package consumer smoke 同时报告 `EvidenceKind=real-callback-runtime`、`RuntimeEvidenceKind=real-callback-runtime`、`RealCallbackRuntime=True`、`IsRealCallbackRuntimeProof=True`、`ProcessDebugTensorInvoked=True`、`InvocationCount>0`、`FailureCount=0` 和完整 package report evidence，readiness 才允许把结果归类为真实 callback runtime proof。
