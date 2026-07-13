# Owner External Execution Result Backfill Kit

`owner-external-execution-result-backfill-kit` 是真实 Owner 外部执行结果回填包。它把 package-consumer runtime proof、post-publish verification、release issue close record owner input 和 final close decision 仍缺失的真实执行结果集中到同一份回填清单，方便 Owner 在 release close 前按字段、日志和 SHA256 一次性补齐。

它不会执行发布，不会采集 proof，不会上传包，也不会关闭 release issue。当前状态必须保持：

- `kitState=blocked-owner-external-execution-results-required`
- `validationState=blocked-owner-external-execution-results-required`
- `failedBlockerCount=0`
- `failedActionRequiredCount>=1`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

## 生成与校验

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerExternalExecutionResultBackfillKit.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerExternalExecutionResultBackfillKit.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
```

输出：

- `artifacts/final-release/owner-external-execution-result-backfill-kit.json`
- `artifacts/final-release/owner-external-execution-result-backfill-kit.md`
- `artifacts/final-release/owner-external-execution-result-backfill-kit-validation.json`
- `artifacts/final-release/owner-external-execution-result-backfill-kit-validation.md`

## 覆盖范围

backfill kit 聚焦四条真实 Owner 外部执行结果线：

- `package-consumer-runtime-proof`
- `post-publish-verification`
- `release-issue-close-record-owner-input`
- `release-issue-final-close-decision`

每条线都必须保留真实外部执行结果要求：仓库外 clean consumer、真实包源、runtime key smoke command、stdout/stderr 摘要、日志 SHA256、host metadata、Owner 决策和 rollback approval。placeholder、missing log、missing SHA256、local feed、ProjectReference、direct `.nupkg`、`blocked-by-cuda-driver` 都只能作为 blocker 或修复输入，不能晋级为 proof。

## 不能误读

- kit 是 Owner 执行指导，不是 runtime proof。
- strict validator command 只是校验入口，不代表外部执行结果已经存在。
- `failedBlockerCount=0` 只表示 kit 形状合法，不表示 release ready。
- `failedActionRequiredCount>=1` 说明仍需 Owner 回填真实执行结果。
- hash、path、validator 状态全部满足前，`canPublishPublicly=false` 和 `canCloseReleaseIssue=false` 必须保持不变。
