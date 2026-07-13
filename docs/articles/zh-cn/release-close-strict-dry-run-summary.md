# Release Close Strict Dry-Run Summary

`release-close-strict-dry-run-summary` 把 runtime proof lane dry-run、release close strict bridge 和 release evidence bundle 合并成关闭前 dry-run gap summary。它用于说明 release issue 为什么仍不能关闭。

## 覆盖范围

- 6 条 proof lane 均生成 close dry-run item。
- 每条 item 合并缺失真实证据字段、bridge blocker、strict close blocker 和 release evidence bundle gap。
- `strictCloseCommand` 固定指向 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`。
- 默认 `closeDryRunState=blocked-release-close-real-proof-required`，并保持 publish/proof/close flag 全部为 false。

## 命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCloseStrictDryRunSummary.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCloseStrictDryRunSummary.ps1 -Strict
```

## 产物

- `artifacts/final-release/release-close-strict-dry-run-summary.json`
- `artifacts/final-release/release-close-strict-dry-run-summary.md`
- `artifacts/final-release/release-close-strict-dry-run-summary-validation.json`
- `artifacts/final-release/release-close-strict-dry-run-summary-validation.md`

## 边界

该 summary 是 blocked release-close gap analysis，不是 close approval。strict dry-run gate、remaining gap、bridge output 和 evidence bundle state 都不能替代真实 post-publish proof、rollback approval、final owner decision、package publication 或 release issue close approval。
