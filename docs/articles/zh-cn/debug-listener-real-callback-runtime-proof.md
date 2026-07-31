# DebugListener Real Callback Runtime Proof

> 状态：real-callback-runtime-proof-gate-ready
> readiness marker：`debug-listener-real-callback-runtime-proof`
> 默认 runtime evidence：`RuntimeEvidenceKind=runtime-smoke-skipped`
> 当前结论：`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`

> 2026-07-31 更新：上面的结论只描述本页既有的历史 promotion-gate 默认输入。仓库现已增加真实
> `TensorRtDebugListenerCallbackOwner(TensorRtApiLine, TensorRtDebugListenerHandler)` 路径，并在本机
> TensorRT 10.11 / CUDA 12.9 上完成 `MarkDebugTensor -> setDebugListener(non-null) ->
> setTensorDebugState -> enqueue -> processDebugTensor -> detach`。真实 owner snapshot 得到
> `InvocationCount=1`、`FailureCount=0`、`InFlightCallbackCount=0`、`TensorName=debug_output` 和
> `IsRealCallbackRuntimeProof=True`。这属于 source-tree local TensorRT callback runtime，不自动等同于公开包、
> Linux、TRT11 runtime、post-publish 或 Owner release proof。

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

## Real Owner Runtime Path

真实路径位于 `native/src/tensorrt/common/debug_listener_callback_owner.inc`，由 TRT10/TRT11 共用宏展开：

- `DebugListenerCallbackOwner final : public nvinfer1::IDebugListener` 使用稳定 heap address，copy/move deleted，析构为 `noexcept`。
- C ABI 提供 create、attach、detach 和 copied snapshot；owner handle 由通用 bridge destroy 释放。
- callback 入口先把 tensor name、type、location 和最多 8 维 shape 复制到固定本地存储，再进入 managed handler。
- TensorRT-owned `addr` 与 `cudaStream_t` 不进入 C ABI，不写入 snapshot，也不暴露给 public C# API。
- detach 清除 TensorRT borrowed listener 后等待 native in-flight counter 归零；managed context 随后解除 owner borrow，最后才释放 native vtable、delegate 和 GCHandle。
- callback 自身线程禁止 clear/replace/dispose execution context，避免 detach 等待当前 callback 造成重入死锁。
- native 编译期断言固定 owner 不可复制/移动、析构不抛异常以及 `processDebugTensor` 的 no-throw vtable 合同；
  in-flight 归零与 condition-variable wait 由同一状态锁同步。
- managed 重入状态按线程记录 callback depth，嵌套 callback 不会提前解除保护；context teardown 会在 native context
  handle 释放后才解除 owner borrow，即使显式 detach 诊断失败也不会提前释放 vtable。

`TensorRtDebugListenerRuntimeSnapshot` 是真实 owner 的 pointer-free runtime report；旧
`TensorRtDebugListenerRealCallbackRuntimeProofResult` 仍保留为历史 package promotion gate，不再代表仓库是否具备实际 owner。

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

历史 direct-method deferred row 继续保留用于兼容审计；owner-safe counterpart 已实现。只有 clean package consumer 在真实
TensorRT runtime path 中同样观察到 callback invocation，并满足完整 proof schema，才可提升 package/release proof。

## Local Validation Snapshot

当前 source-tree TRT10.11/CUDA12.9 与 TRT11.0/CUDA12.9 均实测为 `InvocationCount=1`、`FailureCount=0`、
`InFlightCallbackCount=0`、`DetachCount=1`、`IsRealCallbackRuntimeProof=True`。生成器幂等结果为
`203 manifests / 4009 API records`；TRT10/TRT11 PE export parity 分别为 `1091/1091` 与 `1238/1238`，
missing 均为 0。本批合同与布局测试 `25/25`，完整 solution Debug build `0 warning / 0 error`。

TRT11 使用 `--debug-listener-runtime-smoke-only` 隔离真实 owner 路径，因为同一 vendor build 的旧综合 callback-state
snapshot 由 SEH guard 捕获 `0xC0000005`。该模式仍执行 environment/adapter/safe-surface 检查，只是不让独立旧诊断阻断
真实 callback；它不会把旧诊断写成通过。

这些数字只说明本机源码树、ABI 与真实 TRT10/TRT11 callback 路径闭合，不替代 Linux、bridge-only clean package
consumer、公开包、post-publish 或 Owner release acceptance。
