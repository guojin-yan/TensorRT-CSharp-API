# Real External Proof Backfill Execution Bundle

`real-external-proof-backfill-execution-bundle` 是 Owner 执行包，用来把 package-consumer runtime、post-publish verification、Linux runner、real-model runtime、release close owner input 和 strict close validation 的真实回填动作集中到一个地方。

它不是 proof，不执行发布，不批准公开发布，不关闭 release issue。它只减少 Owner 来回翻找 worklist/runbook/candidate 的成本。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofWorklist.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCloseProofWorklist.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealExternalProofBackfillExecutionBundle.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
```

输出：

- `artifacts/final-release/real-external-proof-backfill-execution-bundle.json`
- `artifacts/final-release/real-external-proof-backfill-execution-bundle.md`

## 默认状态

- `recordKind=real-external-proof-backfill-execution-bundle`
- `bundleState=blocked-real-external-proof-execution-required`
- `trackCount=6`
- `blockedTrackCount=6`
- `performsPublish=false`
- `canPromoteRuntimeProof=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isReleaseCloseProof=false`

## Tracks

- `package-consumer-runtime-proof-execution`
- `post-publish-verification-execution`
- `linux-runner-proof-execution`
- `real-model-runtime-proof-execution`
- `release-close-owner-input-execution`
- `strict-close-validation-execution`

## 不能误读

- execution bundle 是执行交接，不是 proof。
- worklist、runbook、candidate、scaffold、precheck、dry-run、hash audit 不能替代真实 proof。
- local feed、ProjectReference、direct `.nupkg`、DependencyProbe、build-only、template-only、Windows handoff for Linux proof 都是 forbidden substitute。
- release issue close 仍必须等真实 proof records、Owner final decision 和 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady` 通过。
