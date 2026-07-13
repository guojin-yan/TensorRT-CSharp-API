# Owner External Proof Input Preflight

`owner-external-proof-input-preflight` 是真实 owner 外部输入候选预审。它不是 proof，不执行发布，不关闭 release issue，只在 owner 回填真实记录前，对候选材料做分类和风险标记。

生成命令：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerExternalProofInputPreflight.ps1
```

输出：

- `artifacts/final-release/owner-external-proof-input-preflight.json`
- `artifacts/final-release/owner-external-proof-input-preflight.md`

默认边界必须保持：

- `recordKind=owner-external-proof-input-preflight`
- `preflightState=blocked-real-proof-required`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `readyProofLineCount=0`

## 覆盖范围

预审固定覆盖 6 条 proof line：

- `owner-authorization`
- `package-consumer-runtime`
- `linux-runner-proof`
- `real-model-runtime`
- `post-publish-verification`
- `release-issue-close-record`

每条 line 都输出：

- `candidateClassification`
- `canPromoteProof`
- `requiredRealInputCount`
- `missingRealInputCount`
- `existingCandidateArtifactCount`
- `expectedArtifacts`
- `sourceArtifacts`
- `riskMarkers`
- `validatorCommand`
- `ownerNextAction`

## 候选分类

候选输入分类只用于预审，不是 release proof：

- `missing`：缺少相关验证记录或候选产物。
- `template-only`：只有模板、样例或 placeholder 记录。
- `guidance-only`：只有 runbook、handoff、collection、readiness 等指引材料。
- `candidate-needs-owner-review`：候选材料存在，但 validator 尚未提升为真实 proof。
- `validator-passed-real-proof`：validator 明确提升为真实 proof。

## 必须阻断的风险

以下风险出现时，不能进入 release close review：

- local feed
- ProjectReference
- direct `.nupkg` reference
- template-only record
- schema-only record
- preflight-only close record
- readiness snapshot
- dependency-probe-only
- `release-issue-close-record-template.json`
- missing log hash
- mismatched SHA256
- missing owner final close decision
- missing rollback plan

## Release Gate

即使所有 candidate artifacts 都存在，只要没有真实 validator-passing proof，仍必须保持：

- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `release-issue-close-record-validation=blocked-template-only`

最终关闭仍必须依赖真实 `release-issue-close-record.json` 和 validator：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady
```
