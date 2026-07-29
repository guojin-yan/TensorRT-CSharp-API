# DebugListener Real Callback Runtime Proof

> 状态：real-callback-runtime-proof-gate-ready
> readiness marker：`debug-listener-real-callback-runtime-proof`
> 默认 runtime evidence：`RuntimeEvidenceKind=runtime-smoke-skipped`
> 当前结论：`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`

`debug-listener-real-callback-runtime-proof` 是 DebugListener callback 链路的最终 proof promotion gate。它消费 `TensorRtDebugListenerRealNonNullAttachRuntimeSmoke` 与 `TensorRtDebugListenerProcessDebugTensorCallbackTrampoline` 的 copied report，把 opt-in、full package consumer、attach/detach/rollback、native vtable install、`processDebugTensor` invocation、metadata copy、pointer-free surface 与 counters 合并成一个机器可读结论。

该 gate 本身仍然是 not proof。它不默认启用 `setDebugListener(non-null)`，不安装 native `IDebugListener` vtable，不伪造 TensorRT 调用 `IDebugListener::processDebugTensor`。只有 full package consumer 真实输出 `real-callback-runtime` 且 `InvocationCount>0` 时，它才允许 `IsRealCallbackRuntimeProof=True`。

源码 owner 已按职责拆分：evaluation 与 blocker 构造位于
`TensorRtDebugListenerRealCallbackRuntimeProof.cs`，pointer-free report 位于
`TensorRtDebugListenerRealCallbackRuntimeProofResult.cs`。readiness 与源码测试必须组合读取这两个文件。

## Public Surface

- `TensorRtDebugListenerRealCallbackRuntimeProof`
- `TensorRtDebugListenerRealCallbackRuntimeProofResult`

关键字段：

- `RuntimeEvidenceKind`
- `OptInEnabled`
- `FullPackageConsumerReport`
- `RuntimeSmokeReady`
- `TrampolineShapeReady`
- `AttachAttempted`
- `AttachSucceeded`
- `DetachAttempted`
- `DetachSucceeded`
- `RollbackAttempted`
- `RollbackSucceeded`
- `NativeVTableInstalled`
- `ProcessDebugTensorInvoked`
- `InvocationCount`
- `FailureCount`
- `InFlightCallbackCount`
- `BorrowedDebugTensorMetadataCopied`
- `PointerFreeSurfaceReady`
- `ProcessDebugTensorRuntimeReady`
- `AttemptedNoInvocation`
- `CanPromoteRealCallbackRuntime`
- `RuntimeProofBlocked`

默认路径保持：

- `OptInEnabled=False`
- `FullPackageConsumerReport=False`
- `NativeVTableInstalled=False`
- `ProcessDebugTensorInvoked=False`
- `InvocationCount=0`
- `CanPromoteRealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`

## Native Scaffold

native 侧提供 `debug_listener_real_callback_runtime_proof.inc`，包含：

- `DebugListenerRealCallbackRuntimeProofGate final`
- `DebugListenerRealCallbackRuntimeProofReport`
- `configure_api_line`
- `configure_prerequisites`
- `configure_attempt`
- `configure_invocation`
- `can_attempt_runtime_proof`
- `can_promote_real_callback_runtime`
- `make_report`

这些方法用于固定 no-throw proof gate shape。当前 scaffold 只提供 source-visible proof gate，不触发真实 attach，也不创建 native vtable owner。

## Promotion Rules

以下条件缺一不可：

- `OptInEnabled=True`
- `FullPackageConsumerReport=True`
- `RuntimeSmokeReady=True`
- `TrampolineShapeReady=True`
- `AttachSucceeded=True`
- `DetachSucceeded=True`
- rollback 路径成功或未触发
- `NativeVTableInstalled=True`
- `ProcessDebugTensorInvoked=True`
- `InvocationCount>0`
- `FailureCount=0`
- `InFlightCallbackCount=0`
- `BorrowedDebugTensorMetadataCopied=True`
- `PointerFreeSurfaceReady=True`
- `ProcessDebugTensorRuntimeReady=True`
- lower-level reports 均为 `IsRealCallbackRuntimeProof=True`

如果 attach 已尝试但 invocation 仍为 0，必须输出 `RuntimeEvidenceKind=attempted-no-invocation`，proof=false。若 opt-in 后 prerequisites 不满足，输出 `RuntimeEvidenceKind=real-callback-runtime-blocked`，proof=false。CUDA error 35 仍由 package consumer 分类为 `blocked-by-cuda-driver`，不是 API proof。

## Smoke And Package Consumer

`CallbackAllocatorSafeControlsSmokeRunner` 和 package consumer smoke 输出：

- `DebugListenerRealCallbackRuntimeProof=...`
- `EvidenceKind=debug-listener-real-callback-runtime-proof`
- `RuntimeEvidenceKind=runtime-smoke-skipped`、`real-callback-runtime-blocked` 或 `attempted-no-invocation`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `InvocationCount=0`
- `CanPromoteRealCallbackRuntime=False`

package consumer parser 还要求未来 `real-callback-runtime` 输出中 `InvocationCount>0`。仅有字段存在、`SmokeResult=passed`、bridge consumer 编译通过、dependency probe 成功或 callback trampoline shape ready，都不能提升 proof。

## Deferred Boundary

`IDebugListener::processDebugTensor` deferred row 必须继续保留，直到 full package consumer 在真实 TensorRT runtime path 中观察到 callback invocation，并满足完整 proof schema。
