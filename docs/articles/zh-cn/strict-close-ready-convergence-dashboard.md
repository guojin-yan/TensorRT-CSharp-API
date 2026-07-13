# StrictCloseReady Convergence Dashboard

`strict-close-ready-convergence-dashboard` 汇总 final release close blocker dashboard、public publish result import、post-publish clean consumer convergence、final owner decision audit、release issue close record validation 和 classification audit。

该 dashboard 只是 Owner action 聚合层。它不能发布、不能证明 runtime/post-publish、不能批准公开发布，也不能关闭 release issue。最终关闭仍必须以真实 Owner 记录和 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady` 为准。

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-StrictCloseReadyConvergenceDashboard.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StrictCloseReadyConvergenceDashboard.ps1 -Strict
```
