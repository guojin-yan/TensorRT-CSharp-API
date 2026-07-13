# Release Candidate Full Acceptance Summary

`release-candidate-full-acceptance-summary` 是发布候选全量验收的非发布聚合视图。它把最终包审阅、package proof、docs readiness、release evidence、promotion issue、freeze summary/checklist/validation、owner command plan、External Runtime Proof Collection Package、Post Publish Verification Collection Package、stale claims audit 和样例交付面放到一份 owner-facing 摘要中。

它的目标是回答一个问题：当前项目是否已经整理到可以交给 owner 去采集真实 external runtime proof 与真实 post-publish proof。它不是 proof，也不是发布授权。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalPackageReviewBundle.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleasePackageProofBundle.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-DocsPublishReadinessBundle.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofBackfillPlan.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationBackfillPlan.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofCollectionPackage.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationCollectionPackage.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleasePromotionIssueRecord.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateFreezeSummary.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateFreezeChecklist.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCandidateFreezeSummary.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerAuthorizedPublishCommandPlan.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerAuthorizedPublishCommandPlan.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StaleReleaseClaims.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-UserAcceptanceSampleCatalog.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateFullAcceptanceSummary.ps1
```

## 输出文件

- `artifacts/final-release/release-candidate-full-acceptance-summary.json`
- `artifacts/final-release/release-candidate-full-acceptance-summary.md`

## 关键字段

- `recordKind=release-candidate-full-acceptance-summary`
- `acceptanceState=ready-for-owner-proof-collection` 表示本地非发布验收面已经可交给 owner 执行真实 proof collection。
- `performsPublish=false` 表示脚本不会发布包。
- `canPromoteToOwnerProofCollection=true` 只表示可以交给 owner 采集真实 proof，不表示 proof 已存在。
- `canPublishPublicly=false` 必须保持到 owner 明确授权。
- `canUseAsPublicPackageProof=false` 必须保持到真实 channel package proof 存在。
- `canPublishDocsExternally=false` 必须保持到 owner 审阅并授权外发文档。
- `canCloseReleaseIssue=false` 必须保持到真实 external runtime proof、真实 post-publish proof 与 owner 授权齐全。

## 验收项

summary 的 `acceptanceItems` 会逐项记录：

- `final-package-review-bundle`：本地 `.nupkg`、native asset 和 SHA256 inventory。
- `release-package-proof-bundle`：package layout、local feed、native asset copy 和 package-consumer 证据。
- `docs-publish-readiness-bundle`：中文文档与 release docs readiness。
- `release-evidence-bundle`：release evidence 聚合状态。
- `release-promotion-issue-record`：release issue draft 状态。
- `release-candidate-freeze`：freeze summary/checklist/validation 状态。
- `owner-authorized-publish-command-plan`：owner-gated command plan 状态。
- `external-runtime-proof-collection-package`：compatible-host proof collection guidance。
- `post-publish-verification-collection-package`：真实发布后的 clean-consumer verification guidance。
- `backfill-plans`：详细回填步骤。
- `real-proof-boundary`：确认当前仍未伪造真实 proof。
- `stale-release-claims-audit`：确认 release-facing 文案没有过期完成度声明。
- `sample-surface-yolovision`：确认样例交付面仍使用 `YoloVision`，且 asset-required 样例不被当作 smoke passed。

## 边界

以下内容都不能作为 release close proof：

- collection package
- backfill plan
- runbook
- template / draft / example
- local package inventory
- docs readiness
- dependency-probe-only
- `blocked-by-cuda-driver`
- YoloVision asset candidates

`blocked-by-cuda-driver` 不是 smoke passed。`YoloVision` 资产候选不是 real model smoke passed。

## Owner 下一步

1. 在 CUDA/TensorRT 兼容主机运行 External Runtime Proof Collection Package。
2. 使用 `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof` 校验填好的真实 external runtime proof record。
3. 在 owner 授权并真实发布后，从 clean consumer 运行 Post Publish Verification Collection Package。
4. 先使用 `Test-PostPublishCleanConsumerProject.ps1` 扫描源码仓库外的 clean consumer，再用 `Export-PostPublishVerificationRecordInputDraft.ps1` 汇总 package/channel/log SHA256 输入草稿。
5. 使用 `Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof` 校验填好的真实 post-publish verification record。
6. 真实 proof 改变后，重新生成 release evidence、freeze、promotion issue、owner command plan、full acceptance summary 和 `release-close-preflight`。
