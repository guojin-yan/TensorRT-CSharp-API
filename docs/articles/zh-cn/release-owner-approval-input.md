# Release Owner Approval Input

`release-owner-approval-input` 是 release owner 从 dry run 走向真实发布前的人工输入层。它不替代 `final-release-dry-run-summary.json`，也不替代 `release-owner-decision-record.json`；它只回答一个更严格的问题：是否已经有明确 owner 决定允许进入 public promotion。

## 生成模板

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseOwnerApprovalInputTemplate.ps1
```

输出：

- `artifacts/final-release/release-owner-approval-input-template.json`
- `artifacts/final-release/release-owner-approval-input-template.md`

模板默认保持：

- `recordKind=release-owner-approval-input-template`
- `templateOnly=true`
- `approvalState=pending-release-owner-input`
- `canPublishPublicly=false`
- `runtimeProofStatus=blocked-by-cuda-driver`
- `runtimeProofRequiredForRelease=true`
- `externalRuntimeProofState=template-only`
- `externalRuntimeProofRuntimePackageKeyMatches=true`
- `externalRuntimeProofLogSha256FormatReady=false`
- `externalRuntimeProofLogSha256Matches=false`
- `externalRuntimeProofOwnerActionStatus=owner-action-required`
- `postPublishVerificationState=template-only`
- `postPublishCommandsReady=false`
- `postPublishStdoutStderrSummaryReady=false`
- `externalRuntimeProofBackfillPlanState=blocked-compatible-host-proof-required`
- `externalRuntimeProofBackfillCanPromoteRuntimeProof=false`
- `postPublishVerificationBackfillPlanState=blocked-real-post-publish-proof-required`
- `postPublishVerificationBackfillCanCloseReleaseIssue=false`

模板本身不能批准发布。

## Backfill Plan 边界

owner approval input 会显式读取并展示两份回填计划：

- `artifacts/final-release/external-runtime-proof-backfill-plan.json`
- `artifacts/final-release/external-runtime-proof-backfill-plan.md`
- `artifacts/final-release/post-publish-verification-backfill-plan.json`
- `artifacts/final-release/post-publish-verification-backfill-plan.md`

这些 backfill plan 只是 owner guidance，不是 proof、不是发布批准、不是 release close approval，也不会执行 package push。默认情况下必须继续保持 `canPromoteRuntimeProof=false` 和 `canCloseReleaseIssue=false`。

## 填写真实输入

release owner 如果要继续推进，需要复制模板为：

- `artifacts/final-release/release-owner-approval-input-record.json`

并至少填写：

- `recordKind=release-owner-approval-input-record`
- `templateOnly=false`
- `approvalState=approved-for-publication`
- `canPublishPublicly=true`
- 顶层 `ownerName`
- 每个 `decisionInputs[*].decisionState`
- 每个 `decisionInputs[*].rationale`
- 每个 decision 的 owner 信息或顶层 owner 信息

可以先生成一个不会批准发布的示例：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseOwnerApprovalInputExample.ps1
```

输出：

- `artifacts/final-release/release-owner-approval-input-record.example.json`
- `artifacts/final-release/release-owner-approval-input-record.example.md`

示例使用 `recordKind=release-owner-approval-input-example`、`approvalState=example-not-for-publication`、`canPublishPublicly=false`，用于说明字段形状。它不是 owner approval record，校验时必须继续 blocked。

必需决策包括：

| ID | 允许通过的状态 | 说明 |
| --- | --- | --- |
| `release-channel` | `approved-public-channel` / `approved-private-channel` | 选择发布渠道和 rollback 策略。 |
| `signing-policy` | `approved-signed` / `approved-unsigned-rc` | 明确签名策略。 |
| `nvidia-redistribution` | `approved-for-selected-channel` / `approved-private-only` | 明确 NVIDIA runtime 再分发边界。 |
| `runtime-proof-disposition` | `approved-runtime-proof-ready` / `approved-known-limitation-for-rc` | 处理 `runtimeProofRequiredForRelease=true`。 |
| `linux-runner-proof-disposition` | `approved-real-linux-proof` / `approved-windows-only-rc` | 处理 Linux proof 缺口。 |
| `post-publish-verification-disposition` | `acknowledge-post-publish-required` / `require-before-closing-issue` | 处理真实发布后验证和 release issue 关闭门槛。 |
| `callback-proof-disposition` | `approved-real-callback-proof` / `approved-known-limitation-for-rc` | 处理 callback proof=false。 |
| `backfill-plan-boundary-acknowledgement` | `acknowledged-guidance-only` / `require-proof-backfill-before-publish` | 明确 external runtime proof backfill plan 和 post-publish verification backfill plan 只是 guidance。 |

## 校验输入

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseOwnerApprovalInput.ps1 `
  -InputPath artifacts\final-release\release-owner-approval-input-record.json
```

输出：

- `artifacts/final-release/release-owner-approval-input-validation.json`
- `artifacts/final-release/release-owner-approval-input-validation.md`

如果直接校验模板，结果必须是：

- `overallStatus=blocked-owner-input-required`
- `canPublishPublicly=false`

这是预期行为。只有非模板输入、显式 owner、全部必需决策和 clean stale claim audit 同时满足时，校验才允许 `ready-for-owner-approved-publication`。

## 不能误读

- `release-owner-approval-input-template.json` 不是 approval record。
- `approved-known-limitation-for-rc` 不是 smoke passed。
- `runtimeProofRequiredForRelease=true` 必须被明确处置，不能被 `ready-needs-manual-approval` 覆盖。
- `externalRuntimeProofRuntimePackageKeyMatches=true` 只说明 release target key 对齐；缺 `logSha256` 时仍然不是 runtime proof。
- `externalRuntimeProofLogSha256FormatReady=false` 或 `externalRuntimeProofLogSha256Matches=false` 必须保持 `owner-action-required`。
- `external-runtime-proof-backfill-plan.json` 不是 runtime proof；即使 step count 完整，也不能把 `canPromoteRuntimeProof` 改成 true。
- `postPublishCommandsReady=false`、`postPublishConsumerProjectIdentityReady=false`、`postPublishHostReady=false` 或 `postPublishStdoutStderrSummaryReady=false` 时，不能把 release issue 写成 post-publish verified。
- `post-publish-verification-backfill-plan.json` 不是 post-publish proof；真实渠道发布和 clean consumer smoke 缺失时，`canCloseReleaseIssue` 必须保持 false。
- `approved-windows-only-rc` 不等于 Linux runner proof。
- `approved-known-limitation-for-rc` 不等于 callback proof complete。
- 校验通过也不执行 `dotnet nuget push`、GitHub Packages 上传或 GitHub Release 上传。

## 与发布材料的关系

`Export-ReleaseOwnerDecisionRecord.ps1` 和 `Export-ReleasePromotionIssueRecord.ps1` 会读取 `release-owner-approval-input-validation.json`。在默认模板状态下，它们会继续显示：

- `ownerApprovalInputValidationStatus=blocked-owner-input-required`
- `ownerApprovalCanPublishPublicly=false`

这表示发布材料已经准备好接收 owner 输入，但还没有得到公开发布批准。
