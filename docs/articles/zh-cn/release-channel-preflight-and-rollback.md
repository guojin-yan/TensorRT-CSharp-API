# Release Channel Preflight And Rollback

本文把 release owner 从 dry run 推进到真实渠道前必须检查的事项集中起来。它不是发布脚本，也不会执行 `dotnet nuget push` 或 `gh release upload`。

## 当前边界

- `ready-needs-manual-approval` 不是公开发布批准。
- local feed consumer 是打包消费验证，不是 nuget.org 或 GitHub Packages 已发布。
- `blocked-by-cuda-driver` 不是 smoke passed。
- `runtimeProofStatus=blocked-by-cuda-driver` 与 `runtimeProofRequiredForRelease=true` 必须保留到 release issue，不能被 `ready-needs-manual-approval` 覆盖。
- `allowRuntimeSmokeBlocked=true` 只允许 dry run 继续汇总完整证据，不是 smoke ready。
- Linux handoff 与 `template-only` 文件不是 Linux runner proof。
- 真实 callback proof 仍需要 `InvocationCount>0` 和 `IsRealCallbackRuntimeProof=True`。

## 推荐预检顺序

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateChecklist.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalReleaseDryRun.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -AllowRuntimeSmokeBlocked -WarnOnly
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseOwnerApprovalInputTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseOwnerApprovalInput.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseOwnerDecisionRecord.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofRecordTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofRecordInputTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofRecordDraft.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofRecordExample.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofOwnerHandoff.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CompatibleHostRuntimeProofRunbook.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationRecordTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleasePackageProofBundle.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-DocsPublishReadinessBundle.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleasePublishExecutionChecklist.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleasePromotionIssueRecord.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StaleReleaseClaims.ps1
```

生成的 `artifacts/final-release/release-package-proof-bundle.md` 是包实证审阅视图，`artifacts/final-release/docs-publish-readiness-bundle.md` 是文档外发准备审阅视图，`artifacts/final-release/release-evidence-bundle.md` 是证据聚合视图，`artifacts/final-release/release-publish-execution-checklist.md` 是 owner-gated 发布执行清单，`artifacts/final-release/release-promotion-issue-record.md` 可作为 release issue 初稿；这些文件都必须由 release owner 审阅，且都不会执行真实发布。

## 渠道检查

| 渠道 | 预检 | 回滚 |
| --- | --- | --- |
| local feed | 确认消费端无 `ProjectReference`，native assets copy 完整 | 删除 local feed 目录并重新 pack |
| nuget.org | 确认 package owner、API key、签名策略、包体积和 NVIDIA 再分发审批 | 已发布版本通常不能覆盖，只能 unlist 或发布修正版本 |
| GitHub Packages | 确认 token 权限、source URL、组织 retention 和 restore 凭据 | 按组织权限删除、deprecate 或 supersede |
| GitHub Release assets | 确认 tag、release notes、SHA256 和本地 source 使用说明 | 删除资产或发布修正版 notes/assets |
| private feed | 确认访问控制、retention、包体积和客户 NuGet.config | 按组织 feed 策略回滚 |

## 发布后验证

公开或私有渠道推送后，必须新建干净 consumer：

1. 只从目标 channel restore。
2. 不允许 `ProjectReference`。
3. 记录 `consumerProjectName` 和指向 clean consumer `.csproj` 的 `consumerProjectPath`。
4. 验证 managed package 和 runtime package 都来自 channel。
5. 记录 smoke host 的 OS、GPU、driver、CUDA driver/runtime、TensorRT runtime/line 和 cuDNN version。
6. 验证 native bridge 和 vendor runtime copy。
7. 记录 `DependencyProbe BridgeInitialized`。
8. 只有在兼容 CUDA driver/GPU host 上跑通，才写 runtime smoke passed；`smokeCommand` 必须包含 `--runtime-package-key <runtime package key>`。
9. 对 restore log、native asset listing、DependencyProbe log 和 smoke log 计算 SHA256，并回填 post-publish verification record。
10. 记录 restore/build/smoke 的 `stdoutSummary` 与 `stderrSummary`，便于 release issue 审阅。
11. 记录 managed/runtime package id、version、channel URL 和下载后的 nupkg SHA256；本地 nupkg 路径不能替代 channel proof。
12. 只有 `postPublishProofClassification=post-publish-package-consumer-runtime`、runtime smoke passed、日志 SHA256 齐全、干净 consumer 无 `ProjectReference`、consumer project identity 齐全、host metadata 齐全、smoke command 带 runtime key 时，才能把 post-publish verification 作为关闭 release issue 的 proof。
13. 如果 full package smoke 仍被 CUDA driver 阻塞，保持 `blocked-by-cuda-driver` 和 `owner-action-required`，不要关闭 runtime proof。

External runtime proof 的兼容主机回填应先生成：

- `artifacts/final-release/external-runtime-proof-owner-handoff.json`
- `artifacts/final-release/external-runtime-proof-owner-handoff.md`
- `artifacts/final-release/compatible-host-runtime-proof-runbook.json`
- `artifacts/final-release/compatible-host-runtime-proof-runbook.md`

handoff 只记录命令、目标 runtime key、日志路径约定和 validator 命令。它不是 `package-consumer-runtime` proof；只有填好的 `external-runtime-proof-record.json` 包含 clean consumer project identity、CUDA driver/runtime、TensorRT line/runtime、cuDNN version、nupkg SHA256、带 `--runtime-package-key` 的真实 smoke command、真实 smoke log SHA256、stdout/stderr 摘要，并通过 `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof` 后，才能作为 external runtime proof。

`compatible-host-runtime-proof-runbook.md` 是 handoff 的执行版，给外部兼容主机逐步运行 package consumer smoke、计算 nupkg/log SHA256、回填 record 字段和执行 `-FailOnNotProof`。它同样不会发布包，也不会批准 public release。

## Release Issue 最小字段

- 目标版本和 package ID。
- 目标渠道。
- owner approval input validation 状态。
- 签名决定。
- NVIDIA 再分发决定。
- Linux proof 状态。
- runtime smoke 状态。
- runtime proof 状态和 `runtimeProofRequiredForRelease` 处置。
- callback proof 状态。
- rollback 方案。
- 发布后 restore/build/native-copy 验证结果。
