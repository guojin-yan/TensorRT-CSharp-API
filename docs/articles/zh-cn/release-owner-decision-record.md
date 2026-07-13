# Release Owner Decision Record

本文说明 `release-owner-decision-record` 的用途。它是从当前 dry run、Linux handoff、双语文档审计和 stale claim audit 聚合出来的发布负责人记录，不是自动批准结果。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseOwnerDecisionRecord.ps1
```

默认 runtime key：

- Windows RC 线：`win-x64-trt11.0-cuda13.2-cudnn9.22`
- Linux handoff 线：`linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22`

生成产物：

- `artifacts/final-release/release-owner-decision-record.json`
- `artifacts/final-release/release-owner-decision-record.md`

## 当前语义

记录默认保持：

- `recordState=pending-release-owner-approval`
- `canPublishPublicly=false`
- `requiresHumanOwner=true`
- `runtimeProofStatus=blocked-by-cuda-driver`
- `runtimeProofRequiredForRelease=true`
- `realCallbackRuntimeProof=false`
- `isRealLinuxRunnerProof=false`
- `postPublishVerificationState=template-only`
- `postPublishCommandsReady=false`
- `postPublishStdoutStderrSummaryReady=false`
- `externalRuntimeProofBackfillPlanState=blocked-compatible-host-proof-required`
- `externalRuntimeProofBackfillCanPromoteRuntimeProof=false`
- `postPublishVerificationBackfillPlanState=blocked-real-post-publish-proof-required`
- `postPublishVerificationBackfillCanCloseReleaseIssue=false`
- `allowRuntimeSmokeBlocked` 只记录 final dry run 允许环境阻塞继续汇总。

这表示自动证据已经被整理成 owner 可读的决策项，但发布动作仍需人工确认。该记录不会推送 NuGet、不会创建 GitHub Release，也不会把 dry-run 证据提升为正式发布证明。

## 必须保留的边界

- `ready-needs-manual-approval` 不是公开发布批准。
- Linux handoff 不是 Linux runner proof。
- `blocked-by-cuda-driver` 不是 smoke passed，也不是 API proof。
- `runtimeProofRequiredForRelease=true` 表示公开发布前仍需要兼容 runtime proof 或明确 release owner 处置。
- `allowRuntimeSmokeBlocked=true` 不是 smoke ready，也不是 smoke passed。
- post-publish verification 只用于真实渠道发布后的回填；缺 clean consumer identity、host metadata、commands、`--runtime-package-key` smoke command、stdout/stderr summary 或 nupkg hash 时，不能关闭 release issue。
- `external-runtime-proof-backfill-plan.json/.md` 和 `post-publish-verification-backfill-plan.json/.md` 只是下一步执行指引，不能提升为 runtime proof、post-publish proof、publication approval 或 release close approval。
- `schema-ready`、precheck、design gate 和 `InvocationCount=0` 都不是真实 callback runtime proof。
- NVIDIA CUDA、cuDNN、TensorRT runtime 文件是否可公开再分发，必须由 release owner 或 legal 口径确认。

## 与 Template 的区别

`release-owner-decision-template` 是空白审批表，适合复制到 release issue 中逐项填写。

`release-owner-decision-record` 是当前仓库证据快照，适合在候选发布前重新生成，并作为 release owner 决策输入。

## Backfill Plan 决策项

记录中必须保留 `backfill-plan-boundary` 决策项，直到真实 proof 回填完成。该项会展示：

- `externalRuntimeProofBackfillPlanState=blocked-compatible-host-proof-required`
- `externalRuntimeProofBackfillStepCount>=7`
- `externalRuntimeProofBackfillCanPromoteRuntimeProof=false`
- `postPublishVerificationBackfillPlanState=blocked-real-post-publish-proof-required`
- `postPublishVerificationBackfillStepCount>=9`
- `postPublishVerificationBackfillCanCloseReleaseIssue=false`

这个决策项只要求 owner 确认 guidance-only 边界；它不会批准发布，也不会关闭 release issue。

## 推荐接入点

正式发布前建议按顺序执行：

1. 刷新 package/readiness/final dry run 相关脚本。
2. 刷新 `Export-ReleaseOwnerDecisionTemplate.ps1`。
3. 刷新 `Export-ReleaseOwnerDecisionRecord.ps1`。
4. 运行 `Test-StaleReleaseClaims.ps1`，确保新增文字没有过度声明。
5. 由 release owner 在 release issue 中写入明确决定。
