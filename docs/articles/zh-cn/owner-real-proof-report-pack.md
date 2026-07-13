# Owner Real Proof Report Pack

`owner-real-proof-report-pack` 是 Owner 真实 proof 填报包。它读取 `real-proof-execution-record-projection` 的 6 条 execution record，将每条记录整理成更贴近 Owner 执行和复查的 proof report item。

它不是 proof，不执行 package publish，不代表 post-publish verification，也不能关闭 release issue。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerRealProofReportPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerRealProofReportPack.ps1 -Strict
```

## 输出

- `artifacts/final-release/owner-real-proof-report-pack.json`
- `artifacts/final-release/owner-real-proof-report-pack.md`
- `artifacts/final-release/owner-real-proof-report-pack-validation.json`
- `artifacts/final-release/owner-real-proof-report-pack-validation.md`

## 默认状态

- `recordKind=owner-real-proof-report-pack`
- `packState=blocked-owner-real-proof-report-input-required`
- `reportItemCount=6`
- `blockedReportItemCount=6`
- `readyForOwnerReviewCount=0`
- `readyForPromotionCount=0`
- `failedBlockerCount=0`
- `failedActionRequiredCount>=1`
- `canPromoteRuntimeProof=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isReleaseCloseProof=false`

## Proof Lanes

- `package-consumer-runtime`
- `post-publish-verification`
- `linux-runner-proof`
- `real-model-runtime`
- `release-close-owner-input`
- `strict-close-validation`

## Report Item 内容

每条 report item 都包含：

- `requiredOwnerInputs`
- `requiredEvidenceFiles`
- `requiredHashes`
- `requiredCommands`
- `requiredValidators`
- `forbiddenSubstituteChecklist`
- `reviewChecklist`
- `readyForOwnerReview=false`
- `readyForPromotion=false`
- `promotionFlags`

## 边界

- report pack 是 Owner 填报包，不是 proof。
- `readyForOwnerReview` 不等于 `readyForPromotion`。
- hash match 只是完整性信号，不是 runtime proof。
- Windows handoff 不可替代 Linux runner proof。
- 本地 feed、`ProjectReference`、direct `.nupkg`、`DependencyProbe`、build-only、template、hash-only audit 都不可替代真实 proof。
- 只有 Owner 后续填入真实证据、日志、hash、validator output 并完成复查后，后续阶段才能进入严格 proof input candidate。
