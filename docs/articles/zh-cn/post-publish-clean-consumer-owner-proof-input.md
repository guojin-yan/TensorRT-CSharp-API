# Post-Publish Clean Consumer Owner Proof Input

`post-publish-clean-consumer-owner-proof-input` 是发布后 clean consumer proof 的 Owner 输入说明。它不生成新的发布产物，而是说明真实公开包发布后，Owner 需要提供哪些外部信息，才能让 `post-publish-clean-consumer-result-convergence` 和 strict close gate 从 blocked 状态继续推进。

## 必填输入

- 公开渠道 package ID、version、source URL 和 SHA256。
- 仓库外 clean consumer 项目路径或归档。
- clean consumer restore/build/test/smoke 命令。
- stdout/stderr 摘要、exit code、host metadata 和运行时间。
- 禁用 ProjectReference、本地 feed、direct nupkg 或仓库内示例替代真实公开包。
- reviewer 复核结论和 rollback 处理状态。

## 对应验证链路

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishCleanConsumerResultConvergence.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishCleanConsumerResultConvergence.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-StrictCloseReadyConvergenceDashboard.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StrictCloseReadyConvergenceDashboard.ps1 -Strict
```

## 边界

这是一份 Owner 输入说明，不是可执行 proof。任何 local `.nupkg`、local feed、ProjectReference、direct nupkg、template、dry-run、runbook、dashboard 或本地 artifact scan 都不能替代真实公开渠道 clean consumer 运行记录。

在真实公开包、仓库外 clean consumer、runtime smoke、post-publish proof 和 release close decision 全部回填并通过 validator 前，`canCloseReleaseIssue=false` 必须保持不变。
