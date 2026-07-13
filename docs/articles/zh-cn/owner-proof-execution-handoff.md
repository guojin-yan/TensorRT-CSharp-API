# Owner Proof Execution Handoff

`owner-proof-execution-handoff` 是 release owner 的真实外部 proof 采集/发布执行交接看板。它不是 proof，也不是发布脚本，而是把 `owner-proof-backfill-execution-pack` 中 6 条 proof line 转成 owner 可以逐项执行的状态表。

生成命令：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerProofExecutionHandoff.ps1
```

输出：

- `artifacts/final-release/owner-proof-execution-handoff.json`
- `artifacts/final-release/owner-proof-execution-handoff.md`

默认边界必须保持：

- `recordKind=owner-proof-execution-handoff`
- `handoffState=blocked-real-proof-required`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `readyHandoffLineCount=0`

## 覆盖范围

handoff 固定覆盖 6 条真实 proof line：

- `owner-authorization`
- `package-consumer-runtime`
- `linux-runner-proof`
- `real-model-runtime`
- `post-publish-verification`
- `release-issue-close-record`

每条 line 都会输出：

- `currentState`
- `ownerNextAction`
- `firstCommand`
- `validatorCommand`
- `requiredRealInputs`
- `missingRealInputs`
- `expectedArtifacts`
- `existingCandidateArtifacts`
- `missingExpectedArtifacts`
- `sourceArtifacts`
- `cannotUse`
- `canPromoteProof`

## 使用方式

Owner 应按 handoff 的 `ownerExecutionOrder` 执行，而不是从最后的 close record 开始：

1. 先刷新 stale release claims 和 release close preflight。
2. 回填 owner authorization 和 selected-channel 字段。
3. 在兼容主机采集 `package-consumer-runtime` proof。
4. 在真实 Linux x64 runner 采集 Linux proof。
5. 回填 Classification/YoloVision 的 `real-model-runtime` proof。
6. Owner 在脚本之外手工执行真实发布命令。
7. 发布后采集 clean consumer proof。
8. 刷新 release evidence bundle。
9. 最后回填并验证 release issue close record。

## 不能替代的材料

以下材料不能因为出现在 handoff 中就被当作 proof：

- local feed
- ProjectReference
- direct `.nupkg` reference
- template
- draft
- runbook
- collection package
- helper scan
- dependency-probe-only
- readiness snapshot
- preflight-only
- schema-only
- `release-issue-close-record-template.json`
- Windows handoff for Linux proof

## Release Gate

`owner-proof-execution-handoff` 会被 `release-evidence-bundle` 纳入 `sourceEvidence`、`sourceArtifacts` 和 `evidenceItems`，但这只是为了让 owner 交接材料可追溯。它不会改变 release gate：

- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `release-issue-close-record-validation=blocked-template-only`

最终关闭仍必须依赖真实 `release-issue-close-record.json` 和 validator：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady
```

下一层材料是 `owner-external-proof-input-preflight`。它读取 handoff，对 6 条 proof line 的候选输入做 missing/template-only/guidance-only/candidate-needs-owner-review/validator-passed-real-proof 分类，并继续保持 `canPublishPublicly=false` 与 `canCloseReleaseIssue=false`。
