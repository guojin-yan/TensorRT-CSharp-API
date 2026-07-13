# Final Owner Release Close Record Projection

`final-owner-release-close-record-projection` 是最终关闭记录的 Owner 投影。它把真实字段合同、strict record candidate、release issue close validation 和最终 Owner readiness 汇总为 4 条关闭记录 lane。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalOwnerReleaseCloseRecordProjection.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalOwnerReleaseCloseRecordProjection.ps1 -Strict
```

## 产物

- `artifacts/final-release/final-owner-release-close-record-projection.json`
- `artifacts/final-release/final-owner-release-close-record-projection.md`
- `artifacts/final-release/final-owner-release-close-record-projection-validation.json`
- `artifacts/final-release/final-owner-release-close-record-projection-validation.md`

## 边界

默认状态为 `blocked-final-owner-release-close-record-projection-owner-input-required`，`Lanes=4` 且 `Blocked=4`。它只是关闭记录投影，不发布包、不批准公开发布、不生成 runtime proof、post-publish proof 或 release close proof，也不关闭 release issue。

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

Owner 必须先让真实字段合同、strict record candidate、release issue close validation 和最终 readiness 全部通过，再把投影结果作为最终关闭记录的输入。任一 lane 仍 blocked 时，release issue 必须保持打开。
