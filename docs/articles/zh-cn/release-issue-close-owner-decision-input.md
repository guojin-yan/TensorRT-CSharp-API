# Release Issue Close Owner Decision Input

`release-issue-close-owner-decision-input` 是 release issue 最终关闭前的 Owner 决策输入模板。它用于承载真实发布后 Owner 对 rollback、公开包来源、clean consumer proof、runtime smoke 和最终关闭结论的人工确认。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseOwnerDecisionInput.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseOwnerDecisionInput.ps1 -Strict
```

## 产物

- `artifacts/final-release/release-issue-close-owner-decision-input.template.json`
- `artifacts/final-release/release-issue-close-owner-decision-input.template.md`
- `artifacts/final-release/release-issue-close-owner-decision-input-validation.json`
- `artifacts/final-release/release-issue-close-owner-decision-input-validation.md`

## 边界

默认状态为 `blocked-release-issue-close-owner-decision-input-required`。模板字段通过 shape validation 不代表可以关闭 release issue；只有真实 post-publish proof、clean consumer smoke、rollback review、Owner final decision 和 strict close validator 全部通过后，才能进入关闭判断。

该模板不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

## Owner 下一步

Owner 需要用真实公开发布后的证据替换占位字段，并确保 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady` 在最终 close record 上通过。模板或本地候选记录不能替代真实外部 proof。
