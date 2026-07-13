# Release Issue Close Record Real Input Map

`release-issue-close-record-real-input-map` 是 `owner-proof-real-backfill-execution-pack` 之后的 Owner 输入映射层。它把 8 个 Owner 输入任务映射到最终 release close 记录所需字段、目标 artifact、首个刷新命令和严格 validator。

## 当前边界

- `mapState=blocked-release-close-real-input-required`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

该映射只减少 Owner 回填歧义，不执行 publish / push / upload，不关闭 release issue，不把 template、draft、candidate、hash match、local feed、ProjectReference、direct `.nupkg`、schema-only、preflight-only、dependency-probe-only 或 blocked-by-cuda-driver 当作 proof。

## 映射范围

- `rollback-plan` -> `rollbackPlan`
- `rollback-owner` -> `rollbackOwner`
- `rollback-trigger` -> `rollbackTrigger`
- `owner-final-close-decision` -> `ownerFinalCloseDecision`
- `release-issue-id` -> `releaseIssueId`
- `release-issue-url` -> `releaseIssueUrl`
- `public-channel-package-source` -> `publicChannelPackageSource`
- `clean-consumer-runtime-smoke-log` -> `cleanConsumerRuntimeSmokeLog`

## 产物

- `artifacts/final-release/release-issue-close-record-real-input-map.json`
- `artifacts/final-release/release-issue-close-record-real-input-map.md`
- `artifacts/final-release/release-issue-close-record-real-input-map-validation.json`
- `artifacts/final-release/release-issue-close-record-real-input-map-validation.md`

## 生成与验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseRecordRealInputMap.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecordRealInputMap.ps1 -Strict
```

严格验证只证明映射结构合法。只要真实 Owner 输入、真实 post-publish proof、clean consumer runtime smoke log 和 final close approval 没有回填，`missingRealInputCount>=1`、`blockedMappingCount>=1` 与 `failedActionRequiredCount>=1` 就是预期状态。
