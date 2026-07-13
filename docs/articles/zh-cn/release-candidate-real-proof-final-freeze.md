# 发布候选真实 Proof 最终冻结

`release-candidate-real-proof-final-freeze` 是发布候选阶段的真实 proof 最终冻结面，用于记录 release evidence bundle、classification audit、公开发布 Owner 执行包以及外部真实 proof readiness gate 的路径、存在性和 SHA256。

它默认 `Passed=false`，只冻结本地证据状态和待 Owner 审阅字段，不会执行发布、不会下载公开包、不会关闭 release issue，也不会把本地 hash 一致性解释为 runtime proof 或 post-publish proof。

## 边界

- 不是 runtime proof。
- 不是 post-publish proof。
- 不是 publish approval。
- 不是 release close approval。
- 不是 package push。

## Owner 输入

- `releaseEvidenceBundleSha256`
- `classificationAuditSha256`
- `publicReleaseOwnerExecutionPackageSha256`
- `ownerExternalRealProofInputContractSha256`
- `ownerExternalRealProofImportValidatorSha256`
- `postPublishCleanConsumerRealProofGateSha256`
- `runtimeCompatibleHostRealProofGateSha256`
- `releaseCloseRealProofReadinessGateSha256`
- `ownerFreezeReviewDecision`

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateRealProofFinalFreeze.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCandidateRealProofFinalFreeze.ps1 -Strict
```
