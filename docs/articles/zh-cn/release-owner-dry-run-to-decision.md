# Release Owner From Dry Run To Decision

本文给 release owner 一条从 dry run 到发布决策的操作路线。目标是减少临门一脚时的口径漂移。

## 推荐顺序

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateChecklist.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalReleaseDryRun.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -AllowRuntimeSmokeBlocked -WarnOnly
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseOwnerDecisionTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseOwnerApprovalInputTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseOwnerApprovalInput.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseOwnerDecisionRecord.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StaleReleaseClaims.ps1
```

## 决策输入

release owner 至少读取：

- `artifacts/final-release/final-release-dry-run-summary.md`
- `artifacts/final-release/release-owner-decision-template.md`
- `artifacts/final-release/release-owner-approval-input-template.md`
- `artifacts/final-release/release-owner-approval-input-validation.md`
- `artifacts/final-release/release-owner-decision-record.md`
- `artifacts/final-release/stale-release-claims-audit.md`
- `artifacts/linux-dry-run/linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22/linux-runner-evidence-template.md`

## 必须回答的问题

- 选择哪个发布渠道。
- 是否允许当前 RC unsigned。
- NVIDIA runtime 文件是否允许在目标渠道再分发。
- 是否接受当前 package consumer smoke 的 driver 阻塞状态。
- 是否填写并通过 `release-owner-approval-input-record.json`。
- Linux runner proof 是否在本次发布前补齐。
- callback proof=false 是否作为 known limitation 保留。

## 输出建议

最终 release issue 应包含 owner 决定、证据路径、未完成项和回滚策略。默认模板校验结果必须保持 `blocked-owner-input-required` 和 `canPublishPublicly=false`；只有真实 owner 输入校验通过后，才能继续执行发布渠道动作。只有实际推送并完成消费端验证后，才能写发布完成。
