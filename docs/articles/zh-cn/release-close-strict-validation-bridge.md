# Release Close Strict Validation Bridge

`release-close-strict-validation-bridge` 串联 runtime proof input validation、Owner runtime proof runbook、post-publish verification、release issue close record validation、release close final owner runbook 和 release evidence bundle。

## 覆盖范围

- 6 条 proof lane 均映射为 bridge item。
- 每条 bridge item 记录 runtime proof input、Owner runbook、post-publish、close record、final runbook、evidence bundle 的 readiness。
- 默认所有 item 保持 `blocked-release-close-strict-validation-required`。
- strict close command 固定指向 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`。

## 命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCloseStrictValidationBridge.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCloseStrictValidationBridge.ps1 -Strict
```

## 产物

- `artifacts/final-release/release-close-strict-validation-bridge.json`
- `artifacts/final-release/release-close-strict-validation-bridge.md`
- `artifacts/final-release/release-close-strict-validation-bridge-validation.json`
- `artifacts/final-release/release-close-strict-validation-bridge-validation.md`

## 边界

Bridge 是 blocked prerequisite aggregator，不是 runtime proof、不是 publish approval、不是 post-publish proof、不是 rollback approval，也不是 release issue close approval。它只能说明离严格关闭还缺哪些真实输入。
