# Deferred Boundary Risk Tier Gate

本文是 `deferred 边界提升` 的发布质量门。它把剩余 deferred 接口分成可执行层级，避免把 manifest/source 匹配、generated P/Invoke 或 deferred history 误判成真实可用 API。

核心标记：

- `deferred-boundary-risk-tier-gate`
- `manifest-source-match-not-release-proof`
- `no-public-raw-pointer`
- `package-consumer-smoke-required`
- `algorithm-selector-borrowed-pointer`

## 完成度定义

当前项目不能再用“missing 接口清零”判断完成。真实完成必须同时满足：

1. 非 deferred native ABI 已实现，并且不是 no-arg placeholder。
2. manifest 参数、native source、generated binding、public native header 一致。
3. public C# wrapper 返回 typed copied value 或 owning wrapper。
4. public API 不暴露裸 `IntPtr`、`nint`、borrowed TensorRT pointer、plugin creator pointer、algorithm pointer 或 error recorder pointer。
5. 至少有 quality test 或 package-consumer compile/smoke 证据。
6. 若涉及 runtime 行为，还必须有真实 runtime smoke；文档、候选包、precheck、dependency probe 都不能替代 runtime proof。

## 风险层级

| 层级 | 名称 | 可直接批量提升 | 说明 |
| --- | --- | --- | --- |
| A | `A-tier copied value` | 是 | count、exists、scalar getter、caller buffer string、copied snapshot、copied metadata。 |
| B | `B-tier safe alternative` | 条件允许 | 已有 safe alternative 或 alias wrapper，可做 proof 收口，但不能删除 deferred history。 |
| C | `C-tier design-gate-required` | 否 | 需要先设计 owner/lifetime/callback/no-throw/pointer-free snapshot。 |
| D | `D-tier keep-deferred` | 否 | plugin instance、allocator callback、enqueue、borrowed pointer、register/deregister、raw ownership API。 |

## 当前必须继续设计门的典型组

### Algorithm Selector

`IAlgorithm`、`IAlgorithmContext`、`IAlgorithmIOInfo`、`IAlgorithmVariant` 来自 TensorRT algorithm selector callback 期间的短生命周期 borrowed 对象。当前没有 public `JYPPX_TensorRtAlgorithm*` / `AlgorithmContext` owning wrapper，也没有 callback result lifetime 模型。

因此以下接口不能只因为是 getter 就机械提升：

- `IAlgorithm::getTimingMSec`
- `IAlgorithm::getWorkspaceSize`
- `IAlgorithm::getAlgorithmIOInfo`
- `IAlgorithm::getAlgorithmVariant`
- `IAlgorithmContext::getName`
- `IAlgorithmContext::getDimensions`
- `IAlgorithmIOInfo::getStrides`
- `IAlgorithmVariant::getTactic`
- `IAlgorithmVariant::getImplementation`

进入实现前必须先完成 pointer-free copied snapshot 设计，并证明 snapshot 在 callback 生命周期内完成复制，不让 borrowed pointer 逃逸。

### Plugin V2/V3 与 Registry Mutation

Plugin creator inventory 已经有 count/copy 类只读能力，但以下仍是 C/D 层：

- `registerCreator` / `deregisterCreator`
- `loadLibrary` / `deregisterLibrary`
- plugin resource acquire/release
- plugin instance create/clone/enqueue
- Plugin V2/V3 callback trampoline
- borrowed plugin creator/resource pointer public exposure

下一步只能继续做 copied metadata、lookup result、field metadata，不做 ownership 不清的对象暴露。

### Allocator / OutputAllocator / DebugListener Callback

Allocator owner dry-run、ledger state、execution context callback state snapshot 属于 A/B 层证据；它们证明 copied diagnostic 和 no-public-pointer surface，不证明真实 TensorRT callback runtime 已启用。

以下继续保持 C/D 层：

- `IGpuAllocator::allocate/free/deallocate/reallocate`
- `IGpuAsyncAllocator::allocateAsync/deallocateAsync`
- `IOutputAllocator::notifyShape/reallocateOutput`
- `IDebugListener::processDebugTensor`
- callback trampoline invocation proof
- native owner attach 到真实 TensorRT runtime 后的生命周期释放

只有当 no-throw vtable、in-flight accounting、detach-before-release、device pointer ledger 和 runtime smoke 全部闭环后，才能提升为真实 callback runtime proof。

### Execution / NoCopy / Deserialize

`IExecutionContext::execute/executeV2/enqueueV2/INoCopy`、direct `deserializeCudaEngineV2/loadRuntime` 涉及 binding buffer、stream、device memory、profile state、plugin dependency、engine/runtime ownership。它们不能作为普通 getter 处理。

这些接口必须继续走 runtime boundary precheck、dependency diagnostics、package-consumer smoke 和 owner proof。

## 执行规则

- 不删除 deferred history 来制造完成度。
- 不把 `-IncludeMediumRisk` 输出当成低风险实现清单。
- 不只改 manifest；必须同步 native/source/header/interop/wrapper/tests。
- 不向 public API 暴露裸 pointer。
- 不跨 ABI 抛异常。
- 不把 docs/readiness/precheck/package candidate 当成 runtime proof。

## 下一批推荐

优先从 A/B 层继续推进：

1. 已有 copied value native 入口但 public header 或 wrapper 缺口的接口。
2. Plugin registry/creator inventory 的 remaining copied metadata。
3. ErrorRecorder copied snapshot 与 presence/clear 组合。
4. Allocator/callback state 的 dry-run diagnostic、ledger copied state 和 package-consumer compile evidence。
5. B-tier work package 中已有 safe alternative 的 alias proof 收口。

暂不直接推进 Algorithm selector borrowed objects，除非先完成 pointer-free snapshot object model。
