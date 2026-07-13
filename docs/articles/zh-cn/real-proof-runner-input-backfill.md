# Real Proof Runner Input Backfill

`real-proof-runner-input-backfill` 是 Owner 输入模板与验证层，用于把 `real-external-proof-backfill-execution-bundle` 的 6 条 execution track 转成可填写字段。

它不是 proof，不执行发布，不批准公开发布，不关闭 release issue。模板默认必须保持 blocked，直到 Owner 用真实 runner 路径、hash、host metadata、commands、stdout/stderr summary 和 validator output 回填。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealExternalProofBackfillExecutionBundle.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealProofRunnerInputBackfill.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealProofRunnerInputBackfill.ps1 -Strict
```

输出：

- `artifacts/final-release/real-proof-runner-input-backfill.template.json`
- `artifacts/final-release/real-proof-runner-input-backfill.template.md`
- `artifacts/final-release/real-proof-runner-input-backfill-validation.json`
- `artifacts/final-release/real-proof-runner-input-backfill-validation.md`

## 默认状态

- `recordKind=real-proof-runner-input-backfill`
- `inputState=blocked-owner-runner-input-required`
- `trackCount=6`
- `canPromoteRuntimeProof=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isReleaseCloseProof=false`

## 不能误读

- owner input template 不是 runtime proof。
- validator output 只能说明缺口，不能 promotion。
- local feed、ProjectReference、direct `.nupkg`、DependencyProbe、build-only、template-only、Windows handoff for Linux proof、hash-only audit 必须保持 forbidden substitute。
