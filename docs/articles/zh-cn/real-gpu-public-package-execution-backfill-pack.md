# 真实 GPU 与公开包源执行回填包

`real-gpu-public-package-execution-backfill-pack.json` 为 Owner 提供真实 GPU 运行和公开包源 clean consumer 执行时的目录结构、命令模板、日志重定向规则、hash 采集命令和 strict validator 归档路径。它不是 proof，也不执行发布。

## 当前状态

- `backfillState=blocked-owner-action-required`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `performsPublish=false`
- `approvesPublicRelease=false`

## 证据目录

- `artifacts/final-release/owner-evidence/real-model-runtime/{task}/`
- `artifacts/final-release/owner-evidence/package-consumer-runtime/`
- `artifacts/final-release/owner-evidence/post-publish-verification/`
- `artifacts/final-release/owner-evidence/release-issue-close/`

## 关键执行要求

- YoloVision 真实模型运行必须覆盖 `det/cls/seg/obb/pose/sem`。
- 每个模型任务都要记录模型源、license、ONNX hash、engine hash、输入输出 hash、stdout/stderr hash、截图 hash 和 host metadata。
- Package consumer 必须使用仓库外 clean consumer、公开 package source、`dotnet restore --no-cache --force-evaluate`，并检查无 ProjectReference、无 local feed、无 direct `.nupkg`。
- Post-publish verification 必须发生在真实公开发布之后。
- Release issue close 必须等上游 strict validators 全部通过后最后执行。

命令包本身不是 proof，hash 示例不是真实 hash，任何 lane 都不能因为本文件存在而标记为 passed。
