# Runtime Proof Lane Dry-Run Summary

`runtime-proof-lane-dry-run-summary` 汇总 runtime proof execution input、Owner result input、Owner runbook 和 release close strict bridge 的每条 lane 状态，用于提前看清真实 proof 仍缺哪些字段。

## 覆盖范围

- 6 条 proof lane 均映射为 dry-run item。
- 每条 item 汇总 runtime input validation、Owner result input validation、Owner runbook validation 和 close bridge validation。
- `realEvidenceFieldsMissing` 列出 host、package、command、log、hash、validator、exit code、reviewer 等真实证据缺口。
- 默认 `dryRunState=blocked-runtime-proof-real-evidence-required`，并保持 `canPromoteRuntimeProof=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`。

## 命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RuntimeProofLaneDryRunSummary.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RuntimeProofLaneDryRunSummary.ps1 -Strict
```

## 产物

- `artifacts/final-release/runtime-proof-lane-dry-run-summary.json`
- `artifacts/final-release/runtime-proof-lane-dry-run-summary.md`
- `artifacts/final-release/runtime-proof-lane-dry-run-summary-validation.json`
- `artifacts/final-release/runtime-proof-lane-dry-run-summary-validation.md`

## 边界

该汇总是 dry-run 分析，不是 runtime proof。lane readiness、missing field count、bridge blocker、validator command 和本地 aggregation 都不能替代真实外部日志、匹配 SHA256、Owner review、post-publish verification 或 release close strict validation。
