# CUDA Graph Borrowed Handle 安全门

状态：设计门禁
readiness marker：`cuda-graph-borrowed-handle-safety-gate`
适用范围：CUDA Graph borrowed graph/event/node/params 查询边界

本文用于明确 CUDA Graph deferred API 的下一步提升边界。当前目标不是把所有 deferred 直接清零，而是在不公开裸句柄、不误建 owner、不破坏生命周期的前提下，把能复制成托管快照的只读能力逐批提升。

## 当前必须保留 deferred 的接口

| CUDA API | 风险 | 当前结论 |
| --- | --- | --- |
| `cudaGraphChildGraphNodeGetGraph` | 返回 child graph borrowed handle；调用方不拥有 graph | 继续 deferred，除非改为 copied graph identity snapshot 或 owner-referenced view |
| `cudaGraphEventRecordNodeGetEvent` | 返回 borrowed event handle；不能当作 caller-owned event | 继续 deferred，优先设计 event presence/status snapshot |
| `cudaGraphEventWaitNodeGetEvent` | 返回 borrowed event handle；不能跨 wrapper 释放或长期保存 | 继续 deferred，优先设计 event presence/status snapshot |
| `cudaGraphNodeGetContainingGraph` | 返回 containing graph borrowed handle | 保留 direct deferred；已存在 `graph-node-is-in-graph-safe` copied boolean 替代路径 |
| `cudaGraphNodeGetParams` | 返回 discriminated union 参数；含指针、数组和节点类型相关布局 | 继续 deferred，必须先设计 typed copied descriptor |

## Public API 禁止项

这些 API 解锁前，public C# 层不得出现以下形状：

- 返回或接收 `public IntPtr` / `public nint` 表示 graph、event、node params 或 borrowed CUDA handle。
- 把 borrowed graph/event handle 包装成 owning `SafeHandle`。
- 从 graph node getter 直接返回 `CudaGraph`、`CudaEvent` 或可释放对象。
- 暴露可转交 ownership、可释放、可长期保存的 native pointer。
- 通过删除 deferred manifest/source 记录来制造完成度。

## 可提升路径

安全提升应优先采用 copied metadata/snapshot，而不是暴露 handle：

| 能力 | 安全替代设计 |
| --- | --- |
| child graph getter | 返回 child graph 是否存在、graph id/local id/tools id、node type、节点关系摘要 |
| event record/wait getter | 返回 event 是否存在、event node kind、record/wait 方向、必要时返回 copied flags/status |
| containing graph getter | 使用 `CudaGraph.ContainsNode(node)` 或 node id 查询，不返回 graph handle |
| node params getter | 按 node kind 返回 typed copied descriptor，例如 memcpy/memset/kernel/event/child-graph 的只读字段 |
| graph view | 若未来必须返回 view，应保存 owner reference，XML 注释写明生命周期，且 view 不可释放 borrowed handle |

## 解锁检查清单

任一 borrowed handle API 从 deferred 提升前，必须同时满足：

1. native ABI 使用 caller buffer、count/copy 或值类型输出，不返回悬空指针。
2. C# interop 不公开 `IntPtr` / `nint`，仅 internal 传递。
3. public wrapper 返回 copied snapshot 或 owner-referenced borrowed view。
4. XML 注释说明 ownership、生命周期、不可释放语义。
5. smoke 或 quality test 覆盖至少一条可执行路径或静态边界。
6. coverage alias 保留历史 deferred 记录，只把安全替代 API 标记为 implemented。

## 当前边界

本门禁只代表 borrowed handle 风险已被文档化并纳入质量测试。它不是 runtime proof，不是 package-consumer proof，不是 post-publish proof，也不代表这些 deferred API 已经可调用。下一阶段如推进，应优先选择 copied snapshot 设计较清晰的 `cudaGraphNodeGetParams` typed descriptor 或 event node presence/status snapshot。

当前已经允许的安全替代形状是 `EventRecordNodeHasEvent` / `EventWaitNodeHasEvent` 这类 presence snapshot：native 可调用 `cudaGraphEventRecordNodeGetEvent` / `cudaGraphEventWaitNodeGetEvent`，但只把 borrowed event handle 折叠为 `bool`，public C# 不返回 `CudaEvent`、`IntPtr` 或 `nint`。

Boundary: not runtime proof, not post-publish proof, not publish approval, not release close approval, not package push.
