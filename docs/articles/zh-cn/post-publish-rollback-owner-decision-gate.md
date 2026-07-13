# 发布后 Rollback 与 Owner 决策 Gate

`post-publish-rollback-owner-decision-gate` 聚焦发布后 verification、known limitations、rollback plan review、Owner post-publish decision 和 Owner close decision。

它默认 `Passed=false`，只记录发布后 Owner 决策所需字段，不执行发布、不关闭 issue，也不把 rollback review 或本地记录解释为 release close approval。

## 边界

- 不是 runtime proof。
- 不是 post-publish proof。
- 不是 publish approval。
- 不是 release close approval。
- 不是 package push。

## Owner 输入

- `postPublishVerificationRecord`
- `knownLimitationsUrl`
- `rollbackPlanReviewed`
- `rollbackPlanSha256`
- `ownerPostPublishDecision`
- `ownerCloseDecision`
- `strictCloseValidatorCommand`

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishRollbackOwnerDecisionGate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishRollbackOwnerDecisionGate.ps1 -Strict
```
