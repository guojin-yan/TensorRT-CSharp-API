# Deferred 边界：为什么 manifest 100% 不等于可发布 100%

TensorRtSharp4.0 已经把大量 CUDA / TensorRT 接口纳入 manifest 和 source 覆盖，但发布候选完成度不能只看 manifest/source 匹配。真实完成度取决于非 deferred native 实现、高层 C# wrapper、smoke 验证和 package-consumer proof。

## 适合

- 想理解项目从 missing 清零转向 deferred 边界提升的读者。
- 需要审查 `artifacts/interface-coverage/project-completion-review.md` 的维护者。
- 负责决定某个 API 是否能进入 public wrapper 的发布负责人。

## 关键路径

- 总体复审：`artifacts/interface-coverage/project-completion-review.md`。
- 接口矩阵：`artifacts/interface-coverage/tensorrt-interface-comparison.csv`。
- 风险门：`docs/articles/zh-cn/deferred-boundary-risk-tier-gate.md`。
- 设计分组：`docs/articles/zh-cn/deferred-manual-design-groups.md`。

## proof 边界

以下状态都不能单独证明 API 可发布：

- manifest/source 匹配。
- no-arg deferred stub 存在。
- build-only。
- dry-run。
- copied-state。
- schema-only。
- preflight。
- owner-action-required。

Public API 必须避免暴露裸 `IntPtr` / `nint` borrowed object。字符串、数组和 metadata 应通过 count/copy、caller buffer 或 immutable snapshot 方式返回。

## 配图建议

- 一张梯子图：manifest/source match -> non-deferred bridge -> C# wrapper -> smoke -> package-consumer-runtime proof。
- 将 callback trampoline、borrowed pointer、plugin instance create/enqueue 放在高风险 deferred 区。

## 下一步

优先推进只读、可复制、无 ownership 争议的 API；allocator、debug listener、Plugin V2/V3 callback 和 borrowed pointer 相关接口继续走设计门、native lifetime gate 和真实 callback/runtime proof。
