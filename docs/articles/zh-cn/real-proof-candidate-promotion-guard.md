# Real Proof Candidate Promotion Guard

`real-proof-candidate-promotion-guard` 汇总 strict candidate 和 Owner field delta 的提升条件，用于防止 candidate、delta pack、hash field 或 checklist 被误判为真实 proof。

它最多只能说明候选 review 是否允许继续，不能自动晋级 runtime proof、不能发布 package、不能完成 post-publish verification，也不能关闭 release issue。

## 命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealProofCandidatePromotionGuard.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealProofCandidatePromotionGuard.ps1 -Strict
```

## 输出

- `artifacts/final-release/real-proof-candidate-promotion-guard.json`
- `artifacts/final-release/real-proof-candidate-promotion-guard.md`
- `artifacts/final-release/real-proof-candidate-promotion-guard-validation.json`
- `artifacts/final-release/real-proof-candidate-promotion-guard-validation.md`

## 默认状态

- `recordKind=real-proof-candidate-promotion-guard`
- `guardState=blocked-real-proof-candidate-promotion-not-allowed`
- `candidateCount=6`
- `promotionAllowedCandidateCount=0`
- `blockedCandidateCount=6`
- `failedBlockerCount=0`
- `failedActionRequiredCount>=1`
- `canPromoteRuntimeProof=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isReleaseCloseProof=false`

## 边界

promotion guard 必须同时满足 field contract、Owner delta、non-substitute 和后续 real proof validator 条件后，才允许进入候选 review。即使 guard shape 合法，也不能替代真实外部执行、真实 log、真实 SHA256、真实 validator output 和 Owner review。
