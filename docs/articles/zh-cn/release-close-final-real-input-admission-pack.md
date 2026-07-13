# ReleaseClose 最终真实输入准入包

`release-close-final-real-input-admission-pack` 汇总发布候选真实 proof 最终冻结、Owner 真实输入导入预检、公开包 hash 交叉核对、clean consumer/runtime proof 交叉核对和发布后 rollback Owner 决策 gate。

它默认 `Passed=false`，用于明确 ReleaseClose 仍被哪些真实输入 blocker 阻塞；不会执行发布、不会下载包、不会关闭 release issue，也不会把任何本地候选材料晋级为真实 proof。

## 边界

- 不是 runtime proof。
- 不是 post-publish proof。
- 不是 publish approval。
- 不是 release close approval。
- 不是 package push。

## Owner 输入

- `releaseCandidateRealProofFinalFreeze`
- `ownerRealInputImportPreflight`
- `publicPackageHashCrossCheckGate`
- `cleanConsumerRuntimeProofCrossCheckGate`
- `postPublishRollbackOwnerDecisionGate`
- `releaseEvidenceClassificationAudit`
- `releaseIssueCloseStrictValidation`

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCloseFinalRealInputAdmissionPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCloseFinalRealInputAdmissionPack.ps1 -Strict
```
