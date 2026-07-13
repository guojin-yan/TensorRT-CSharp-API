# DebugListener Real Callback Runtime Proof Preflight

> 状态：runtime-proof-attempt-preflight / preflight-ready
> readiness marker：`debug-listener-runtime-proof-attempt-preflight`
> runtime evidence：`RuntimeEvidenceKind=runtime-proof-attempt-preflight`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：真实 `IDebugListener::processDebugTensor` runtime proof 前的最后一层准入审计。

## 目标

`TensorRtDebugListenerRuntimeProofAttemptPreflight` 消费 `TensorRtDebugListenerRuntimeProofPrecheckResult`，只回答一个问题：当前证据是否足够开始真实 runtime proof 实现。

当前答案仍然是否定的。该 preflight 不调用 TensorRT，不启用 `setDebugListener(non-null)`，不安装 native vtable，不调用 `IDebugListener::processDebugTensor`，也不把任何 gate/scaffold/precheck evidence 提升为 `real-callback-runtime`。

## Public API

- `TensorRtDebugListenerRuntimeProofAttemptPreflight`
- `TensorRtDebugListenerRuntimeProofAttemptPreflightResult`
- `Evaluate(TensorRtDebugListenerCallbackOwnerSnapshot)`
- `Evaluate(TensorRtDebugListenerRuntimeProofPrecheckResult)`

关键字段：

| 字段 | 当前含义 |
| --- | --- |
| `CanEnableSetDebugListenerNonNull` | 当前为 `False`。native line-specific non-null attach entry、owner lifecycle、ownership contract 仍未完成。 |
| `CanInstallNativeVTable` | 当前为 `False`。native `IDebugListener` vtable trampoline、exception capture、status mapping、in-flight accounting 尚未完成。 |
| `CanCallProcessDebugTensorRuntime` | 当前为 `False`。不能在真实 TensorRT build/enqueue 路径触发 `processDebugTensor`。 |
| `CanPromoteRealCallbackRuntime` | 当前为 `False`。没有 full package consumer `real-callback-runtime` 证据。 |
| `ReasonNonNullAttachStillBlocked` | copied string，解释为什么 non-null attach 仍被阻塞。 |
| `ReasonNativeVTableStillBlocked` | copied string，解释为什么 native vtable 仍不能安装。 |
| `ReasonRuntimeProofStillBlocked` | copied string，解释为什么 runtime proof 仍不能 promotion。 |

所有字段都是 pointer-free。public API 不暴露 `IntPtr`、`UIntPtr`、`nint`、native owner address、debug tensor pointer、callback pointer 或 vtable pointer。

## Smoke 输出

`CallbackAllocatorSafeControlsSmokeRunner` 输出：

- `DebugListenerRuntimeProofAttemptPreflight=...`
- `EvidenceKind=debug-listener-runtime-proof-attempt-preflight`
- `RuntimeEvidenceKind=runtime-proof-attempt-preflight`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `CanEnableSetDebugListenerNonNull=False`
- `CanInstallNativeVTable=False`
- `CanCallProcessDebugTensorRuntime=False`
- `CanPromoteRealCallbackRuntime=False`
- `RuntimeProofBlocked=True`

这些输出是准入审计，不是 runtime proof。

## 下一阶段准入条件

进入 native attach entry 最小安全实现前，必须同时满足：

- stable owner lifetime：native owner identity 稳定，release 前不会移动或复制。
- non-copyable native owner：native owner storage 禁止 copy/move，且不会把 address 暴露到 public API。
- no-throw destructor：析构和 release 路径不跨 ABI 抛异常。
- no-throw vtable callback：`processDebugTensor` callback stub 必须捕获异常并映射 status。
- exception-to-status mapping：managed exception 只进入 copied diagnostic/status，不跨 C ABI。
- in-flight drain before release：release 前等待 callback in-flight 计数归零。
- borrowed debug tensor/data lifetime：只复制 metadata，不暴露 borrowed pointer。
- full package consumer smoke：真实 TensorRT invocation 必须输出完整 `EvidenceKind=real-callback-runtime`、`RuntimeEvidenceKind=real-callback-runtime`、`RealCallbackRuntime=True`、`IsRealCallbackRuntimeProof=True` 和 counters。

在这些条件满足前，`IDebugListener::processDebugTensor` deferred row 必须保留。

## 不能做的事

- 不能用 `debug-listener-runtime-proof-attempt-preflight` 替代 `real-callback-runtime`。
- 不能因为 `SmokeResult=passed` 就设置 `IsRealCallbackRuntimeProof=True`。
- 不能公开 raw pointer 或 borrowed pointer。
- 不能跨 ABI 抛异常。
- 不能破坏 TRT8/TRT10/TRT11 version guard。
