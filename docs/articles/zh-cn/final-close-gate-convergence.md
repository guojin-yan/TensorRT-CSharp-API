# Final Close Gate Convergence

`final-close-gate-convergence` 是最终关闭 gate 的收敛视图。它把真实公开发布结果、post-publish clean consumer proof、runtime proof、rollback/final owner decision、strict close validation 和 release evidence classification audit 汇总成一个 owner-action-required 看板。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalCloseGateConvergence.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalCloseGateConvergence.ps1 -Strict
```

## 产物

- `artifacts/final-release/final-close-gate-convergence.json`
- `artifacts/final-release/final-close-gate-convergence.md`
- `artifacts/final-release/final-close-gate-convergence-validation.json`
- `artifacts/final-release/final-close-gate-convergence-validation.md`

## 边界

该收敛视图默认状态为 `blocked-final-close-gate-owner-proof-required`。它只展示最终关闭 gate 的阻断 lane，不发布包、不运行 proof、不批准公开发布，也不关闭 release issue。

必须保持：

- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isPostPublishProof=false`
- `isReleaseCloseProof=false`

## Owner 下一步

Owner 必须用真实公开渠道和仓库外 clean consumer 结果补齐所有 blocker，并通过 release evidence classification audit 与 strict close validator。只要任一 lane 仍为 blocked、template-only、local-only、dry-run 或 owner-input-required，最终关闭 gate 就必须继续保持 blocked。
