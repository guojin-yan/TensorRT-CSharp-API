# Owner Runtime Proof Execution Runbook

`owner-runtime-proof-execution-runbook` 将 runtime proof execution input record 转为 Owner 可执行的命令序列。每条 lane 都列出准备目录、执行命令、hash 采集、validator command 和 expected artifacts。

## 覆盖范围

- 6 条 proof lane 均保留 blocked 状态。
- 每条 runbook item 至少包含 4 步命令序列和 5 个 required hash。
- validator commands 来自上游 closure/input record，便于 Owner 按 lane 执行。
- 默认 `runbookState=blocked-owner-runtime-proof-execution-required`，不提升 runtime proof，不允许发布，不允许关闭 release issue。

## 命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerRuntimeProofExecutionRunbook.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerRuntimeProofExecutionRunbook.ps1 -Strict
```

## 产物

- `artifacts/final-release/owner-runtime-proof-execution-runbook.json`
- `artifacts/final-release/owner-runtime-proof-execution-runbook.md`
- `artifacts/final-release/owner-runtime-proof-execution-runbook-validation.json`
- `artifacts/final-release/owner-runtime-proof-execution-runbook-validation.md`

## 边界

Runbook 是执行手册，不是 proof。命令序列、expected artifacts、required hashes 和 validator slots 不能替代真实 runtime 执行、post-publish verification、rollback approval 或 release close strict validation。
