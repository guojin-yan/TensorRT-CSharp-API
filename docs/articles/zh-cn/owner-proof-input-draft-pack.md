# Owner Proof Input Draft Pack

`owner-proof-input-draft-pack` 是真实 owner proof 输入的非 proof 填写面。它读取 `owner-proof-input-repair-pack.json`，为 6 条 proof line 输出 per-line draft spec、draft path、strict validator command 和 proof-substitute blocker。

它不执行发布，不批准公开发布，不关闭 release issue，也不把任何 input draft 标记为 proof。默认必须保持：

- `recordKind=owner-proof-input-draft-pack`
- `draftPackState=blocked-draft-non-proof`
- `inputDraftIsProof=false`
- `canPromoteProof=false`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

## 生成与校验

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerProofInputRepairPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerProofInputDraftPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerProofInputDraftPack.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
```

输出：

- `artifacts/final-release/owner-proof-input-draft-pack.json`
- `artifacts/final-release/owner-proof-input-draft-pack.md`
- `artifacts/final-release/owner-proof-input-draft-pack-validation.json`
- `artifacts/final-release/owner-proof-input-draft-pack-validation.md`

## 覆盖的 Proof Line

- `owner-authorization`
- `package-consumer-runtime`
- `linux-runner-proof`
- `real-model-runtime`
- `post-publish-verification`
- `release-issue-close-record`

## 关键字段

每条 draft spec 包含：

- `inputDraftPath`
- `inputDraftIsProof=false`
- `canPromoteProof=false`
- `requiredRealInputs`
- `requiredRealInputRules`
- `placeholderFieldsToReplace`
- `fieldsRequiringExistingFiles`
- `fieldsRequiringSha256`
- `fieldsRequiringCleanConsumerEvidence`
- `fieldsRequiringOwnerDecision`
- `fieldsRequiringRollbackPlan`
- `firstRepairCommand`
- `validatorCommand`
- `strictValidationCommand`
- `blockedByNonProofMarkers`

## 高优先级规则

`package-consumer-runtime` draft spec 必须要求：

- clean external consumer identity
- no ProjectReference
- no local feed as public proof
- managed/runtime nupkg SHA256
- runtime package key match
- compatible host metadata
- smoke command with `--runtime-package-key`
- smoke log path / SHA256

`release-issue-close-record` draft spec 必须要求：

- release evidence bundle SHA256
- release close preflight path/hash
- stale claims audit path/hash
- post-publish proof validation path/hash
- rollback plan
- owner final close decision
- strict close validator command

## 不可替代边界

Draft pack、input draft、repair pack、template、handoff、preflight、runbook、collection package 和 readiness snapshot 都不是 proof。`Test-OwnerProofInputDraftPack.ps1 -Strict` 只证明 draft pack 的阻断字段和 strict validator command 存在，不证明真实 package consumer runtime、Linux runner、真实模型 runtime、post-publish verification 或 release issue close 已完成。
