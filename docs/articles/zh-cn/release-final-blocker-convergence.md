# 发布阻塞项最终收敛与可发布判定

本页说明 `release-final-blocker-convergence` gate。它的目标不是继续新增说明性材料，而是把公开发布前剩余阻塞项压缩成可以执行、可以校验、可以停止反复循环的最终清单。

## 当前判定

- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `performsPublish=false`
- `approvesPublicRelease=false`
- source summary: `artifacts/final-release/release-proof-owner-backfill-summary-validation.json`

当前仍处于 `blocked-owner-action-required`。原因是四条 release proof lane 尚未全部具备真实 Owner 执行输入和 strict validator output。

## 四条 Lane

| Lane | 当前状态 | 是否可提升 |
|---|---|---:|
| `real-model-runtime` | 缺真实 record、validator output 和运行日志 | false |
| `package-consumer-runtime` | record 存在，但缺 owner evidence logs 和 validator output | false |
| `post-publish-verification` | record 存在，但缺公开发布后 clean consumer 日志和 validator output | false |
| `release-issue-close` | 缺 close record，且依赖前三条 lane 全部通过 | false |

## 发布边界

该 gate 是 read-only convergence runner，只汇总 strict validator 阻塞项和 Owner 下一步动作。它不会：

- 创建真实 proof。
- 生成或补造 SHA256。
- 生成或补造 host metadata。
- 执行 `dotnet nuget push`。
- 将 lane 标记为 passed。
- 关闭 release issue。

## 不可替代项

以下内容不能作为 release proof：template、report、matrix、article、command pack、summary pack、dry-run、build-only、sidecar-only、screenshot-only、`Skipped=True`、`blocked-by-cuda-driver`、local feed、ProjectReference、direct `.nupkg`。

## 最短 Owner 动作

1. 在兼容 GPU / CUDA / TensorRT host 上运行真实模型 runtime 用例，归档 stdout/stderr、hash、host metadata、record 和 strict validator output。
2. 从公开包源创建外部 clean consumer，运行 restore/build/runtime smoke，归档日志、hash、host metadata、record 和 strict validator output。
3. 公开发布后再次运行 clean consumer post-publish verification，归档日志、hash、host metadata、record 和 strict validator output。
4. 只有前三条 lane 全部 strict-validator passed 后，才生成 release issue close record 并运行 close validator。

## 重新生成

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseProofOwnerBackfillSummary.ps1 -OutputPath artifacts\final-release\release-proof-owner-backfill-summary-validation.json
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseFinalBlockerConvergence.ps1 -SummaryPath artifacts\final-release\release-proof-owner-backfill-summary-validation.json -OutputPath artifacts\final-release\release-final-blocker-convergence.json
```
