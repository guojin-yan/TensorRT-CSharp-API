# Callback 与 Allocator 安全桥接路线

TensorRT 的 callback、allocator、debug listener 等接口很有价值，但也是跨语言桥接中最容易出错的区域。本文解释为什么项目不直接暴露裸指针回调，以及从 deferred boundary 走向真实 runtime proof 需要哪些步骤。

## 为什么要谨慎

跨 C++/C# callback 涉及：

- 对象生命周期。
- borrowed pointer。
- vtable 稳定性。
- 异常不能跨 ABI 抛出。
- attach/detach 顺序。
- inflight callback 计数。
- 线程和释放时机。

如果直接把 `IntPtr` 暴露给 public API，短期看似可用，长期会造成不可诊断的崩溃。

## 当前策略

| 类型 | 当前策略 | 证明要求 |
| --- | --- | --- |
| allocator | owner ledger + safety gate | runtime smoke 与释放顺序 |
| output allocator | attach/detach design gate | buffer ownership proof |
| debug listener | nothrow callback + borrowed tensor copy | real callback runtime proof |
| plugin callback | 暂不直接 trampoline | ABI 和 ownership 设计 |

## 从 deferred 到真实 API

1. 先定义 pointer-free 或 copy-out metadata。
2. 建 owner ledger，记录 attach/detach/release。
3. native 层保证 nothrow。
4. C# 层避免暴露 ownership 不明的裸 `IntPtr`。
5. 增加 smoke，不只验证 source/manifest。
6. 再进入 package-consumer-runtime 或 real callback runtime proof。

## 证据边界

- design gate 不是 runtime proof。
- build-only 和 parse-only 不能证明 callback 可安全运行。
- sidecar-only 不适用于 callback proof。
- `blocked-by-cuda-driver` 是兼容主机 owner action。
- real callback runtime proof 不能由文档或 template 伪造。
- `package-consumer-runtime` 是 release proof record，不是 callback design note。
- `real-model-runtime` 是样例模型证据，不是 allocator 生命周期 proof。

## 推荐对外说法

可以说：项目已经为 callback/allocator 安全桥接建立 owner ledger、design gate 和 runtime proof 路线。

不要说：所有 callback 已经真实 runtime proof 完成。

## 下一步

下一批接口提升应选择低 ownership 风险的 API。涉及 callback、borrowed pointer、external resource 的接口必须先补 nothrow native bridge、C# wrapper、smoke 和质量测试，再考虑移出 deferred。
## Managed Readiness 聚合

新增 `TensorRtCallbackAllocatorReadinessSnapshot` 后，callback / allocator 安全桥接路线被拆成两个更清楚的层次：

1. `managed readiness`：由 `TensorRtCallbackAllocatorReadiness.Evaluate(...)` 聚合 logger、profiler、progress monitor、allocator ledger、OutputAllocator precheck 和 DebugListener precheck 的复制式证据，输出 `IsPublishSafeForManagedCallbacks`、`BlockedReasonCount`、`RuntimeProofBlocked` 等字段。
2. `real callback runtime proof`：仍需要兼容 TensorRT/CUDA 主机、真实 attach、真实 callback invocation、device pointer ledger、stream lifetime、exception/status mapping 和 full package consumer 报告，完成后才能让 `IsRuntimeInvocationProofComplete` 为真。

这个 snapshot 的价值是减少发布前检查时的文件往返：它给出一个高层 C# API、smoke、package consumer 和质量测试都能引用的统一摘要。但它不改变 deferred 边界，不删除 `IGpuAllocator::allocate`、`IOutputAllocator::reallocateOutput`、`IDebugListener::processDebugTensor` 等 callback 行的 deferred 状态，也不能把 blocked / skipped proof 说成 passed。

`TensorRtCallbackOwnerClosureMatrixResult` 则把 managed readiness 往下拆成 family-level closure matrix：`GpuAllocator`、`GpuAsyncAllocator`、`OutputAllocator`、`DebugListener`、`StreamReaderWriter` 分别输出 `ReadyClosureColumnCount`、`PackageConsumerRuntimeProofRequired`、`PackageConsumerRuntimeProofReady`、`RuntimeProofBlocked` 和 `DeferredRowsStillRequired`。该矩阵可以作为下一阶段排工入口，但仍是 `RuntimeEvidenceKind=closure-matrix`，不能作为真实 callback runtime proof。

下一批如果继续推进，应优先沿矩阵中缺失列补真实 runtime proof 的外部证据输入、兼容主机执行记录、native vtable install、detach/release 归零和 invocation 记录，而不是继续扩展 managed-only gate 的字段数量。
