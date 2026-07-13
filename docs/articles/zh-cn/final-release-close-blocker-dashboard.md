# Final Release Close Blocker Dashboard

`final-release-close-blocker-dashboard` 是最终 release close 前的 Owner 可读 blocker 看板。它把 final freeze manifest、public publish handoff、public package proof、post-publish confirmation、public proof bridge、final post-publish audit、final owner decision audit、strict close validator 和 classification audit 聚合到一张紧凑表。

该 dashboard 只显示 owner action、required proof、validator 和 non-substitute 原因，不执行发布、不提升 proof、不批准 close，也不能替代 strict close validator。

## 产物

- `artifacts/final-release/final-release-close-blocker-dashboard.json`
- `artifacts/final-release/final-release-close-blocker-dashboard.md`
- `artifacts/final-release/final-release-close-blocker-dashboard-validation.json`
- `artifacts/final-release/final-release-close-blocker-dashboard-validation.md`

## 当前边界

- `performsPublish=false`
- `canPromoteRuntimeProof=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isReleaseCloseProof=false`
- `isPostPublishProof=false`

## Blocker 范围

看板覆盖 9 个最终 blocker：release candidate final freeze manifest、public publish owner manual command handoff、public package proof owner input、post-publish proof owner confirmation、release close public proof bridge、final post-publish audit pack、release issue close final owner decision audit、strict release close validator 和 release evidence classification audit。

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalReleaseCloseBlockerDashboard.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalReleaseCloseBlockerDashboard.ps1 -Strict
```

当前看板可以 ready，但 `blockedBlockerCount` 仍大于 0；这表示看板结构可用，不表示 release issue 可以关闭。
