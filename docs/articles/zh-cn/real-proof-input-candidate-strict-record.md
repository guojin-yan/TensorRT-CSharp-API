# Real Proof Input Candidate Strict Record

`real-proof-input-candidate-strict-record` 是 Owner real proof report pack 之后的严格候选记录合同。它把每条 report item 拆成 owner input、evidence file、hash、command、validator、forbidden substitute 和 owner review 七类必填字段，并判断这些字段是否足以进入候选 review。

它不是 runtime proof、不是 publish approval、不是 post-publish verification，也不是 release close approval。

## 命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealProofInputCandidateStrictRecord.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealProofInputCandidateStrictRecord.ps1 -Strict
```

## 输出

- `artifacts/final-release/real-proof-input-candidate-strict-record.json`
- `artifacts/final-release/real-proof-input-candidate-strict-record.md`
- `artifacts/final-release/real-proof-input-candidate-strict-record-validation.json`
- `artifacts/final-release/real-proof-input-candidate-strict-record-validation.md`

## 默认状态

- `recordKind=real-proof-input-candidate-strict-record`
- `candidateState=blocked-real-proof-input-candidate-required`
- `candidateCount=6`
- `blockedCandidateCount=6`
- `readyCandidateCount=0`
- `failedBlockerCount=0`
- `failedActionRequiredCount>=1`
- `canPromoteRuntimeProof=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isReleaseCloseProof=false`

## 边界

这个阶段只判断 Owner 填报包是否满足候选记录合同。即使候选记录 shape 合法，缺少真实 Owner evidence、log、hash、validator output、forbidden substitute 复查或 owner review 时，也必须保持 blocked/non-proof。
