# Release Issue Close Final Owner Decision Audit

`release-issue-close-final-owner-decision-audit` 是最终关闭 release issue 前的 Owner 决策聚合审计。它把公开包 proof 输入、post-publish proof owner confirmation、public proof bridge、close record candidate、final close decision、strict close validator 和 release evidence classification audit 串成 7 个 gate。

该审计默认保持 blocked/non-proof：它只回答“最终 Owner 决策还差哪些真实 gate”，不执行发布、不上传包、不生成 runtime proof、不生成 post-publish proof，也不能批准公开发布或关闭 release issue。

## 产物

- `artifacts/final-release/release-issue-close-final-owner-decision-audit.json`
- `artifacts/final-release/release-issue-close-final-owner-decision-audit.md`
- `artifacts/final-release/release-issue-close-final-owner-decision-audit-validation.json`
- `artifacts/final-release/release-issue-close-final-owner-decision-audit-validation.md`

## 当前边界

- `performsPublish=false`
- `canPromoteRuntimeProof=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isReleaseCloseProof=false`
- `isPostPublishProof=false`

## Gate 范围

| Gate | 作用 |
|---|---|
| `public-package-proof-owner-input` | Owner 必须填写真实公开包 URL、registry、nupkg SHA256、发布时间和 review 字段。 |
| `post-publish-proof-owner-confirmation` | Owner 必须确认 public package proof、post-publish clean consumer proof、结果导入和 close owner bridge。 |
| `release-close-public-proof-bridge` | 所有最终公开 proof bridge gate 必须 ready。 |
| `release-issue-close-record-candidate` | close record candidate 必须包含真实 post-publish proof、rollback owner input 和 candidate validation。 |
| `release-issue-final-close-decision` | Owner 必须填写最终 close decision、rollback review、公开包来源、clean consumer 和 log/hash review 字段。 |
| `release-issue-close-record` | strict close validator 必须通过 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`。 |
| `release-evidence-classification-audit` | 所有 non-proof 边界必须保持 intact，不能把候选、模板、runbook 或审计包当 proof。 |

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseFinalOwnerDecisionAudit.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseFinalOwnerDecisionAudit.ps1 -Strict
```

Strict 校验只允许 blocker/action-required 的真实状态暴露，不允许通过修改字段伪造成 `canCloseReleaseIssue=true` 或 release-close proof。
