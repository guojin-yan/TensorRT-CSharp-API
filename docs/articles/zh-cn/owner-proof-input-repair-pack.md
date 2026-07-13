# Owner Proof Input Repair Pack

`owner-proof-input-repair-pack` 是真实 owner proof 输入的修复包。它读取 `owner-external-proof-input-preflight.json`，把 6 条仍为 `template-only` 或 blocked 的 proof line 拆成字段级动作：哪些 placeholder 必须替换、哪些路径必须指向真实文件、哪些字段必须带 SHA256、哪些字段必须来自 clean consumer、哪些字段需要 owner decision，以及哪些字段需要 rollback plan。

它不是 proof，不执行发布，不批准公开发布，也不关闭 release issue。默认必须保持：

- `recordKind=owner-proof-input-repair-pack`
- `repairPackState=blocked-real-input-repair-required`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerExternalProofInputPreflight.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerProofInputRepairPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
```

输出：

- `artifacts/final-release/owner-proof-input-repair-pack.json`
- `artifacts/final-release/owner-proof-input-repair-pack.md`

## Repair Item 字段

每条 repair item 至少包含：

- `currentCandidateClassification`
- `repairState`
- `requiredRealInputs`
- `placeholderFieldsToReplace`
- `fieldsRequiringExistingFiles`
- `fieldsRequiringSha256`
- `fieldsRequiringCleanConsumerEvidence`
- `fieldsRequiringOwnerDecision`
- `fieldsRequiringRollbackPlan`
- `inputDraftPath`
- `inputDraftIsProof=false`
- `firstRepairCommand`
- `validatorCommand`
- `cannotUseMarkers`
- `blockedReason`

这些字段只帮助 owner 填真实输入。即使生成了 `.input-draft.json` 或 `.repair.md`，这些 input draft 与 repair markdown 也只是修复过程材料，不是 release proof。

## 覆盖的 Proof Line

- `owner-authorization`
- `package-consumer-runtime`
- `linux-runner-proof`
- `real-model-runtime`
- `post-publish-verification`
- `release-issue-close-record`

## 不可替代边界

repair pack 不允许把以下材料晋级成 proof：

- template、draft、schema-only record
- runbook、collection package、handoff、preflight
- local feed、ProjectReference、direct `.nupkg`
- readiness snapshot、helper scan、dependency-probe-only
- `blocked-by-cuda-driver`
- Windows handoff for Linux proof
- `release-issue-close-record-template.json`
- missing log hash、mismatched SHA256
- missing owner final close decision
- missing rollback plan

真实晋级只能来自对应 validator 显式通过后的记录。例如 package-consumer-runtime 需要 clean external consumer、真实 package source、managed/runtime `.nupkg` SHA256、compatible host metadata、`--runtime-package-key` smoke command、真实 smoke log 和匹配 SHA256；release close 还需要 rollback plan 与 owner final close decision。

## 与 Evidence Bundle 的关系

`Export-ReleaseEvidenceBundle.ps1` 会把 repair pack 纳入 `evidenceItems`、`sourceEvidence` 和 `sourceArtifacts`，并显示 blocked item count 与 missing real input count。但这只是聚合状态，不改变 `canPublishPublicly=false` 或 `canCloseReleaseIssue=false`。

下一步应由 owner 逐条填充真实输入记录，运行对应 validator，再刷新：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerExternalProofInputPreflight.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerProofInputRepairPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
```
