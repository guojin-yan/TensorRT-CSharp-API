# Owner Real Proof Field Delta Pack

`owner-real-proof-field-delta-pack` 将 `real-proof-input-candidate-strict-record` 中仍未 ready 的 field contract 转换成 Owner 可执行的字段填报 delta。

它只提供下一步 Owner action、目标 report pack、目标 strict record 和验证命令，不是 runtime proof、不是 publish approval、不是 post-publish verification，也不是 release close approval。

## 命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerRealProofFieldDeltaPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerRealProofFieldDeltaPack.ps1 -Strict
```

## 输出

- `artifacts/final-release/owner-real-proof-field-delta-pack.json`
- `artifacts/final-release/owner-real-proof-field-delta-pack.md`
- `artifacts/final-release/owner-real-proof-field-delta-pack-validation.json`
- `artifacts/final-release/owner-real-proof-field-delta-pack-validation.md`

## 默认状态

- `recordKind=owner-real-proof-field-delta-pack`
- `deltaState=blocked-owner-real-proof-field-delta-required`
- `candidateCount=6`
- `fieldDeltaCount>=6`
- `blockedFieldContractCount>=6`
- `readyFieldContractCount=0`
- `failedBlockerCount=0`
- `failedActionRequiredCount>=1`
- `canPromoteRuntimeProof=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isReleaseCloseProof=false`

## 边界

delta pack 只是 Owner 填报差异清单。模板、candidate、report pack、hash-only audit、local feed、ProjectReference、direct nupkg、DependencyProbe 和 checklist-only 证据都不能满足真实 proof。
