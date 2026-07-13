# Owner External Proof Execution Bundle

`owner-external-proof-execution-bundle` 将 release close strict dry-run 中仍 blocked 的 6 条 proof lane 转成 Owner 可执行的外部 proof 命令包。它的目标是减少 Owner 真机执行时的上下文切换：每条 lane 都包含命令、hash 命令、host metadata、导入字段映射、non-substitute checklist 和 remaining close gaps。

## 覆盖范围

- 6 条 proof lane 均生成 execution bundle item。
- 每条 item 保留 `remainingCloseGaps`，用于说明为什么仍不能 release close。
- 每条 item 保留 `performsPublish=false`、`canPromoteRuntimeProof=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`。
- 默认 `bundleState=blocked-owner-external-proof-execution-required`。

## 命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerExternalProofExecutionBundle.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerExternalProofExecutionBundle.ps1 -Strict
```

## 产物

- `artifacts/final-release/owner-external-proof-execution-bundle.json`
- `artifacts/final-release/owner-external-proof-execution-bundle.md`
- `artifacts/final-release/owner-external-proof-execution-bundle-validation.json`
- `artifacts/final-release/owner-external-proof-execution-bundle-validation.md`

## 边界

该 bundle 是 blocked owner execution guidance，不是 runtime proof、publish approval、post-publish proof 或 release close approval。命令、hash 槽、host metadata 提示和 remaining gap 统计都不能替代真实外部执行记录与严格 validator。
