# TensorRtSharp 4.0 常见问题：安装、版本、模型、性能与 Proof 边界

## 适用读者

这篇 FAQ 面向首次接触 TensorRtSharp 4.0 的用户、准备接入样例的模型工程师、以及需要解释 proof 状态的维护者。

## 解决问题

项目很大，用户常常不知道先看哪里、如何选择 runtime、为什么 sample 需要自备模型、为什么 release 还显示 blocked。本文把常见问题集中回答。

## 常见问题

**Q：我应该从哪里开始？**
先看 `README.zh-CN.md`、`docs/index.md`、`docs/articles/zh-cn/getting-started.md`。如果目标是模型转换，看 `samples/OnnxToEngine` 和 `applications/TensorRtExec`；如果目标是 YOLO，使用 `samples/YoloVision`。

**Q：旧检测样例命名还在吗？**
不建议恢复早期过窄的检测样例命名。当前统一视觉样例是 `samples\YoloVision`，覆盖 YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom，以及 det、cls、seg、obb、pose、sem。

**Q：为什么 sample 需要我自己提供模型？**
模型、labels、输入图片、license 和 SHA256 涉及授权与可复现性。项目可以提供 runner、manifest 和模板，但真实模型 proof 需要 owner 提供可审计资产。

**Q：TensorRtExec report 能不能当 proof？**
不能。TensorRtExec report 是 build/precheck/diagnostics 证据，不是 package-consumer-runtime proof。

**Q：为什么 release close 还 blocked？**
因为 clean owner input、package-consumer-runtime proof、post-publish verification 等真实证据仍未全部通过。dashboard 是状态聚合，不是 close approval。

**Q：我能用 local feed 验证包吗？**
可以用于开发排查，但不能作为 public package consumer proof。release proof 必须来自 public package source。

## 推荐阅读顺序

```text
docs/articles/zh-cn/tensorrtsharp-4-project-overview-campaign.md
docs/articles/zh-cn/tensorrtsharp-4-architecture-abi-wrapper.md
docs/articles/zh-cn/tensorrtsharp-nuget-runtime-package-guide.md
docs/articles/zh-cn/package-consumer-runtime-proof-clean-consumer-guide.md
```

## 边界说明

FAQ 是说明文档，不是 runtime proof。`build-only`、`dry-run`、`template`、local feed、ProjectReference、direct `.nupkg`、GUI screenshot、TensorRtExec report、YoloVision matrix、OnnxToEngine report、sidecar、readonly diagnostics 都不能替代 package-consumer-runtime proof 或 post-publish verification。

## 下一步

如果你是普通用户，按 runtime package 和 sample 文档运行；如果你是 release owner，优先补齐 clean consumer proof 输入；如果你是贡献者，优先选择只读、查询型、低 ownership 风险的 deferred API 批量提升。
