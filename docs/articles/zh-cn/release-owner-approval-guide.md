# Release Owner Approval Guide

本文给 release owner 一张最终审批用的边界清单。它不替代自动脚本，而是说明哪些状态可以接受，哪些状态必须保持为人工审批项。

## 当前 Dry Run 读法

当前 `Test-FinalReleaseDryRun.ps1` 汇总状态：

- `overallStatus=ready-needs-manual-approval`
- `blockingIssueCount=0`
- `bilingualDocumentationFindingCount=0`
- `packageConsumerSmokeStatus=blocked-by-cuda-driver`
- `runtimeProofStatus=blocked-by-cuda-driver`
- `runtimeProofRequiredForRelease=true`
- `allowRuntimeSmokeBlocked=true`
- `realCallbackRuntimeProof=false`
- `signingStatus=unsigned-or-not-requested`

这表示发布候选没有自动 blocker，双语 API 文档 gate 已清零，但仍不能直接宣称正式发布完成。

## 必须人工确认

release owner 需要明确处理：

- Linux handoff：当前只有 handoff/dry-run evidence，真实 Linux runner proof 仍需目标 runner 产出。
- Signing and trust：当前是 `unsigned-or-not-requested`，正式渠道是否允许 unsigned 必须审批。
- Full package runtime smoke：当前 Windows 主机是 `blocked-by-cuda-driver`，这不是 API proof。
- Runtime proof：`runtimeProofStatus=blocked-by-cuda-driver` 且 `runtimeProofRequiredForRelease=true`，公开发布前必须由兼容 CUDA driver/runtime 环境补齐 proof，或由 release owner 明确记录处置方式。
- External runtime proof：真实记录必须包含 CUDA driver supported runtime、CUDA runtime version、TensorRT line/runtime version、目标 runtime package key、managed/runtime nupkg SHA256、真实 smoke 命令、stdout/stderr 摘要和 smoke log SHA256；`precheck`、`dependency-probe-only`、`blocked-by-cuda-driver` 都不能晋级。
- Post-publish verification：真实发布后还必须记录 package id/version、package URL、下载后的 nupkg SHA256、干净 consumer restore/build/smoke 日志和 `postPublishProofClassification=post-publish-package-consumer-runtime`；template-only 不能关闭 release issue。
- Allow runtime smoke blocked：`allowRuntimeSmokeBlocked=true` 只说明 dry run 允许环境阻塞继续汇总，不是 smoke passed。
- NVIDIA CUDA/cuDNN/TensorRT 再分发：公开发布前必须确认条款、包体积和发布渠道。

## 可以视为已完成的事项

- Public API bilingual documentation audit：`findingCount=0`。
- Public API bilingual documentation backlog：空工作队列，`backlogFindingCount=0`。
- Local feed consumer：禁止 `ProjectReference` 的 PackageReference 路径已纳入证据。
- Runtime package matrix：保留 Windows TRT8/TRT10/TRT11 与 Linux runtime line 的发布视图。

## 不允许的审批写法

- 不要把 `ready-needs-manual-approval` 写成正式发布完成。
- 不要把 Linux handoff 写成 Linux runner proof。
- 不要把 `blocked-by-cuda-driver` 写成 smoke passed。
- 不要把 `runtimeProofRequiredForRelease=true` 写成 runtime proof 已完成。
- 不要把 `allowRuntimeSmokeBlocked=true` 写成 smoke ready 或 smoke passed。
- 不要把 `postPublishProofClassification=template-only`、`owner-action-required`、`dependency-probe-only` 或 `blocked-by-cuda-driver` 写成 post-publish proof。
- 不要把 `InvocationCount=0`、schema-ready、precheck 或 design gate 写成真实 callback proof。
- 不要把 collection package、runbook、template、draft、example、local inventory 或 local feed 写成真实 proof。
- 不要把缺少 `stdoutSummary` / `stderrSummary` 复核摘要或缺少 `-RequireExistingLog` SHA256 校验的记录写成通过。
- 不要在真实 external runtime proof 与真实 post-publish verification proof 缺失时把 `canCloseReleaseIssue` 改为 `true`。
- 不要删除 deferred rows 来制造完成度。

## Owner 冻结前最小复核

冻结前 owner 至少复核以下三条：

1. `external-runtime-proof-validation.json` 是否为 `validationState=real-runtime-proof`，且 `canPromoteRuntimeProof=true`。
2. `post-publish-verification-validation.json` 是否只在真实发布后变为 `real-post-publish-verification-proof`，且 `canCloseReleaseIssue=true`。
3. `release-candidate-full-acceptance-summary.json` 在 proof 不齐时必须继续保持 `canPublishPublicly=false` 与 `canCloseReleaseIssue=false`。

## 推荐审批记录

发布前在 release issue 或 release note 中记录：

- final release dry run summary 路径和生成时间。
- release owner 对 unsigned、渠道、NVIDIA 再分发的决定。
- Windows runtime smoke 的当前状态和后续复测计划。
- runtime proof required 的处置方式：补跑兼容环境、继续 RC 限制发布，或明确阻塞 public promotion。
- Linux runner proof 的目标 runner、runtime key 和证据路径。
- callback proof 是否仍为 false，以及为什么不阻塞当前 RC。
