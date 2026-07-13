# Release Issue Close Record Owner Input

`release-issue-close-record-owner-input` 是 release issue close record candidate 的 owner 回填模板与验证面。它把 release evidence bundle、release close preflight、stale claims audit、post-publish proof validation、rollback plan、rollback owner/trigger、owner final close decision、release issue id/url 固定为可审计字段。

该 surface 只用于 close candidate overlay，不是 close proof，不执行发布，不关闭 release issue，也不能替代 post-publish proof validation、rollback approval、owner final decision 或最终 strict close validator。

## 产物

- `artifacts/final-release/release-issue-close-record-owner-input.template.json`
- `artifacts/final-release/release-issue-close-record-owner-input.template.md`
- `artifacts/final-release/release-issue-close-record-owner-input-validation.json`
- `artifacts/final-release/release-issue-close-record-owner-input-validation.md`

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseRecordOwnerInputTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecordOwnerInput.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseRecordCandidate.ps1 -OwnerInputPath artifacts/final-release/release-issue-close-record-owner-input.template.json
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecordCandidate.ps1 -Strict
```

## 边界

- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- strict 模式只因 blocker 失败而失败；无真实 close 输入时保持 `blocked-owner-input-required` 与 action-required。
