# Release Issue Close Strict Owner Decision Import

`release-issue-close-strict-owner-decision-import` 将最终 release issue 关闭前的 Owner 决策输入拆成严格 lane。它聚合真实公开发布结果、post-publish clean consumer proof、rollback review、最终 Owner decision 和 strict close validator 前置条件，确保任何缺失字段都会继续阻断关闭。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseStrictOwnerDecisionImport.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseStrictOwnerDecisionImport.ps1 -Strict
```

## 产物

- `artifacts/final-release/release-issue-close-strict-owner-decision-import.json`
- `artifacts/final-release/release-issue-close-strict-owner-decision-import.md`
- `artifacts/final-release/release-issue-close-strict-owner-decision-import-validation.json`
- `artifacts/final-release/release-issue-close-strict-owner-decision-import-validation.md`

## 边界

该导入面默认状态为 `blocked-release-issue-close-strict-owner-decision-required`。它只映射 Owner 决策 lane 和缺失项，不批准公开发布、不关闭 issue、不把候选记录提升为 proof。

必须保持：

- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isPostPublishProof=false`
- `isReleaseCloseProof=false`

## Owner 下一步

Owner 需要先补齐真实公开发布结果、真实仓库外 clean consumer proof、rollback review、final close decision 和 strict validator 通过记录。只有这些真实记录全部通过后，release issue close record 才能进入最终关闭校验；本导入面本身不能作为 close approval。
