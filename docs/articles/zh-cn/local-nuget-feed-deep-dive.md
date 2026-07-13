# Local NuGet Feed Deep Dive

Local NuGet feed 验证用于证明消费端通过 PackageReference 使用包，而不是通过源码工程引用绕过打包问题。

## 核心规则

`Test-LocalNuGetFeedConsumer.ps1` 必须保持：

- 使用 `local-release-candidate-feed`。
- 禁止 `ProjectReference`。
- restore source 只指向本地 feed 或明确允许的源。
- native asset copy 结果写入 summary。
- driver 阻塞时记录 `blocked-by-cuda-driver`。

## 为什么重要

源码 build 通过不代表 NuGet consumer 可用。真实消费端还需要验证：

- managed package 是否可 restore。
- runtime split package 是否可解析。
- native bridge 和 vendor runtime 是否复制到输出目录。
- `DependencyProbe BridgeInitialized` 是否出现。

## 推荐证据

- `artifacts/package-consumer/package-consumer-validation-summary.json`
- `artifacts/release-candidate/runtime-package-matrix.json`
- `artifacts/final-release/final-release-dry-run-summary.json`

## 边界

Local feed consumer 可以证明打包消费路径，不等于公开发布完成。公开渠道仍需要 release owner 选择、签名策略、NVIDIA 再分发确认和发布后 restore 验证。
