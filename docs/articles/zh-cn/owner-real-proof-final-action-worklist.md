# Owner 真实 Proof 最终执行清单

本页对应 `owner-real-proof-final-action-worklist`。它直接从 `release-final-blocker-convergence.json` 派生，不是新的 proof，也不是发布批准。

## 当前状态

- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `worklistState=blocked-owner-action-required`
- source: `artifacts/final-release/release-final-blocker-convergence.json`

当前四条 proof lane 仍需要 Owner 在真实环境执行并回填证据。没有真实日志、SHA256、host metadata 和 strict validator output 时，不允许公开发布，也不允许关闭 release issue。

## 四条工作项

| Lane | 当前缺口 |
|---|---|
| `real-model-runtime` | 缺真实 runtime record、stdout/stderr logs 和 validator output |
| `package-consumer-runtime` | record 存在，但缺公开包源 clean consumer 日志和 validator output |
| `post-publish-verification` | record 存在，但缺发布后 clean consumer 日志和 validator output |
| `release-issue-close` | 缺 close record，并依赖前三条 lane 全部 strict-validator passed |

## 最短执行顺序

1. 在真实兼容 CUDA/TensorRT host 上执行 real model runtime 用例，归档日志、hash、host metadata 和 strict validator output。
2. 从公开包源创建外部 clean consumer，执行 restore/build/runtime smoke，归档日志、hash、host metadata 和 strict validator output。
3. 公开发布后执行 post-publish clean consumer verification，归档日志、hash、host metadata 和 strict validator output。
4. 只有前三条 lane 全部通过后，才生成 `release-issue-close-record.json` 并运行 close validator。

## 重新生成

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseProofOwnerBackfillSummary.ps1 -OutputPath artifacts\final-release\release-proof-owner-backfill-summary-validation.json
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseFinalBlockerConvergence.ps1 -SummaryPath artifacts\final-release\release-proof-owner-backfill-summary-validation.json -OutputPath artifacts\final-release\release-final-blocker-convergence.json
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerRealProofFinalActionWorklist.ps1
```

## 不可替代项

template、report、matrix、article、command pack、summary pack、dry-run、build-only、sidecar-only、screenshot-only、`Skipped=True`、`blocked-by-cuda-driver`、local feed、ProjectReference、direct `.nupkg` 都不能作为 release proof。
