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

`eng/Test-BridgePackageRuntimeConsumer.ps1` 还会在仓库外创建只引用本地 managed 与 `.Bridge` nupkg 的 clean consumer，
在同一次 identity enqueue 中执行 debug tensor 标记、真实 callback 和 detach。该报告把证据域固定拆成
`source-tree`、`local-package`、`public-package`、`post-publish`；只有 `local-package` 可以由本次运行晋级。晋级同时要求
consumer 无 `ProjectReference`、直接程序集引用和源码探测，并满足 invocation>0、failure/in-flight=0、copied metadata、
`BorrowedPointerExposed=False`、detach>0。缺少任一 marker 都按失败处理，不能用默认的零值代替证据。
TRT10.11/CUDA12.9 与 TRT11.0/CUDA12.9 已使用本地 `4.0.10000-local.callback` managed/bridge nupkg 通过该路径，
两条线均得到 invocation=1、failure/in-flight=0、detach=1；package policy 同时确认 managed 包无 native asset，
每个 bridge 包只有项目自有 `jyppxtrtbridge.dll`。使用 `-SkipInstalledVendorAssetHashing` 的诊断运行不会晋级该证明。

同一 package consumer 支持 `-DebugListenerScenario callback-return-false`、`callback-throw`、
`attempted-no-invocation` 与 `missing-vendor-dependency` 四个受控负例。前两项必须观察到 invocation>0、failure>0、
in-flight=0 和成功 detach；no-invocation 必须观察到 attach/detach 但 invocation=0；缺依赖必须在隔离 loader PATH
下取得真实 `DllNotFoundException`/loader failure，或精确的 Windows guarded module-not-found code
`3228369022`（`0xC06D007E`）。callback false/throw 时 TensorRT 仍可能完成 enqueue 且子进程退出 0，因此判定必须读取
failure、LastStatus、handler outcome、in-flight 与 detach marker。负例通过只表示预期失败被拒绝，四级 proof 均保持 false。

最新 DLL 的 TRT10.11/CUDA12.9 复验得到一次真实 callback、零 failure、零 in-flight、一次 detach，且
`BorrowedPointerExposed=False`。native owner 的 drain wait 已用同一状态锁同步计数归零；managed context 在 native
context handle 销毁后才解除 owner borrow，嵌套 managed callback 使用 depth 保护，避免 bool 提前复位。

TRT11.0/CUDA12.9 也通过同一 owner 路径，得到相同的一次 invocation、零 failure/in-flight 与一次 detach。
后续 getter matrix 在 TRT10.11 与 TRT11.0 都定位到 vendor 默认 debug listener 的 `getInterfaceInfo()` 可能触发
`0xC0000005`；output allocator、temporary-storage allocator 与三项 presence getter 并不是本次首个失败阶段。
native 现已把 borrowed getter 求值和 `getInterfaceInfo()` 一并放入 SEH guard，并在不扩展旧 caller-allocated struct
的前提下用 `last_status`、`last_operation`、`last_diagnostic` 返回 partial phase。`GetCallbackStateSnapshot` 会保留已复制
字段，`TryGetCallbackStateSnapshot` 对 partial 返回 false；TRT10/TRT11 的 phase 均为
`snapshot-debug-listener-interface-info-partial`。TRT11 综合 safe controls 随后通过，且同一进程中的真实 callback 仍得到
invocation=1、failure/in-flight=0、detach=1。`--debug-listener-runtime-smoke-only` 仍可用于单独隔离 owner 路径，但不再是
绕开 callback-state snapshot 异常的必要条件。

四个 fail-closed 负例由 `eng/Test-BridgePackageRuntimeNegativeControlMatrix.ps1` 分别运行，并写入按 runtime key/scenario
隔离的目录，避免覆盖 success report。矩阵会按 manifest 验证本机 TensorRT 根中的必需 DLL；默认开发根不完整时，只使用
主机上已有的 assembled runtime，不把 vendor DLL 放入 bridge 包。callback-return-false、callback-throw、attempted-no-invocation 必须观察到 coherent、
pointer-free callback-state；missing-vendor-dependency 在 context 创建前故意失败，必须记录 `observed=false`，且不能声称
取得 snapshot。aggregate 要求所有 runtime、local callback、public-package、post-publish、publish 与 release-close proof
标志均为 false；expected failure 只能证明所选错误被观察并被拒绝。

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

只有对应证据域的 consumer smoke 同时报告 `EvidenceKind=real-callback-runtime`、真实 invocation、零 failure/in-flight、
pointer-free copied metadata、成功 detach 和完整 package identity，readiness 才能在该证据域内归类为真实 callback runtime
proof。`local-package` 结果不能替代 `public-package` 或 `post-publish` 结果。
