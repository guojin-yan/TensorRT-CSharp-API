# Final Close Owner Approval Boundary Audit

`final-close-owner-approval-boundary-audit` 是最终关闭前的 Owner approval 边界审计。它把公开发布 Owner result、post-publish clean consumer proof、rollback/final decision、strict release issue close record、最终 readiness 和分类审计汇总为 6 条 approval lane。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalCloseOwnerApprovalBoundaryAudit.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalCloseOwnerApprovalBoundaryAudit.ps1 -Strict
```

## 产物

- `artifacts/final-release/final-close-owner-approval-boundary-audit.json`
- `artifacts/final-release/final-close-owner-approval-boundary-audit.md`
- `artifacts/final-release/final-close-owner-approval-boundary-audit-validation.json`
- `artifacts/final-release/final-close-owner-approval-boundary-audit-validation.md`

## 边界

默认状态为 `blocked-final-close-owner-approval-boundary-owner-action-required`，`Lanes=6` 且 `Blocked=6`。它只是 Owner approval 边界审计，不发布包、不批准公开发布、不生成 runtime proof、post-publish proof 或 release close proof，也不关闭 release issue。

必须保持：

- `notExecutedByAutomation=true`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isPostPublishProof=false`
- `isReleaseCloseProof=false`
- `isReleaseCloseRecordProof=false`

## Owner 下一步

Owner 必须完成真实公开发布结果、仓库外 clean consumer proof、rollback review、最终 close decision、strict close validation 和分类审计复核。任一 approval lane 仍 blocked 时，release issue 必须保持打开。
