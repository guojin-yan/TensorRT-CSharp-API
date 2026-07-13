# Release Close Proof Worklist

`release-close-proof-worklist` 是 release issue 最终关闭前的 blocker map。它把 package-consumer runtime proof、post-publish verification、Linux runner proof、real-model runtime proof、release close owner input、close candidate、final close decision 和 strict close validator 聚合为一份 Owner 可执行清单。

它不是 proof，不执行发布，不批准公开发布，不关闭 release issue。只要任一真实 proof 或 Owner 决策仍缺失，`canCloseReleaseIssue=false` 必须保持。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofWorklist.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCloseProofWorklist.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
```

输出：

- `artifacts/final-release/release-close-proof-worklist.json`
- `artifacts/final-release/release-close-proof-worklist.md`

## 默认状态

- `recordKind=release-close-proof-worklist`
- `worklistState=blocked-release-close-real-proof-required`
- `workItemCount=8`
- `blockedWorkItemCount>=1`
- `failedActionRequiredCount>=1`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `performsPublish=false`
- `isRuntimeExecutionProof=false`
- `isReleaseCloseProof=false`

## 聚合范围

- `package-consumer-runtime-proof-worklist.json`
- `post-publish-verification-validation.json`
- `linux-runner-evidence-validation.json`
- `sample-run-evidence-record-validation.json`
- `real-model-owner-handoff.json`
- `release-issue-close-record-owner-input-validation.json`
- `release-issue-close-record-candidate-validation.json`
- `release-issue-final-close-decision-validation.json`
- `release-issue-close-record-validation.json`

## 不能误读

- worklist 是 Owner 执行清单，不是 release-close proof。
- `template-only`、`owner-action-required`、candidate、scaffold、runbook、precheck、dashboard、hash audit 都不能替代真实 proof。
- Linux runner proof 必须来自真实 linux-x64 runner，不是 Windows handoff。
- real-model runtime proof 必须由真实模型、真实输入和 validator-passing runtime log/hash 支撑。
- post-publish verification 只能在真实渠道发布后，用 package identity/hash、clean consumer identity、host metadata、commands、stdout/stderr 和 log hash 回填。
- release issue close 仍以 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady` 为最终 gate。
