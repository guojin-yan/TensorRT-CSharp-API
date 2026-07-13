# Release Close Final Owner Runbook

`release-close-final-owner-runbook` 是 release close 最后一公里的 Owner 执行手册。它基于 `owner-proof-real-input-convergence`，把真实公开包源、包 SHA256、仓库外 clean consumer runtime smoke、日志 SHA256、host metadata、rollback review、final close decision 和 strict close validator 串成一个顺序化执行面。

## 当前边界

- `runbookState=blocked-release-close-final-owner-action-required`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

该 runbook 只指导 Owner 执行和回填真实 proof，不执行 publish / push / upload，不关闭 release issue，不把 template、draft、candidate、hash match、local feed、ProjectReference、direct `.nupkg`、schema-only、preflight-only、dependency-probe-only 或 blocked-by-cuda-driver 当作 proof。

## 覆盖步骤

- 确认真实 public package channel。
- 从公开渠道下载包并记录 SHA256。
- 在仓库外准备 clean consumer。
- 运行 clean consumer restore/build/runtime smoke。
- 捕获 smoke log、SHA256、exit code、stdout/stderr summary 和 host metadata。
- 回填 post-publish verification owner input。
- 回填 package consumer runtime proof candidate。
- 回填 release issue id/url。
- 回填 rollback owner/trigger/plan。
- 回填 final close decision。
- 刷新 close record overlay candidate。
- 刷新 release close strict candidate。
- 刷新 owner proof convergence。
- 最后运行 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`。

## 失败恢复映射

- public package source 失败：回到 `post-publish-verification-owner-input`。
- clean consumer runtime smoke 失败：回到 `package-consumer-runtime-proof-candidate`。
- rollback review 失败：回到 `release-issue-final-close-decision`。
- close candidate 失败：回到 `release-issue-close-record-candidate` / `release-issue-close-record-overlay-candidate`。
- strict validator 失败：回到 `release-issue-close-record-validation`，release issue 必须继续保持 open。

## 产物

- `artifacts/final-release/release-close-final-owner-runbook.json`
- `artifacts/final-release/release-close-final-owner-runbook.md`
- `artifacts/final-release/release-close-final-owner-runbook-validation.json`
- `artifacts/final-release/release-close-final-owner-runbook-validation.md`

## 生成与验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCloseFinalOwnerRunbook.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCloseFinalOwnerRunbook.ps1 -Strict
```

严格验证只证明 runbook 结构合法、步骤齐全、不可替代 proof 类型完整，并保持 `failedBlockerCount=0`。只要真实 Owner proof、clean consumer runtime smoke、rollback approval、final close decision 和最终 close validator 还没有全部通过，`blockedStepCount>=1`、`ownerActionStepCount>=1` 与 `failedActionRequiredCount>=1` 就是预期状态。
