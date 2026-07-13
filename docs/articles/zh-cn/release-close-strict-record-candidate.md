# Release Close Strict Record Candidate

`release-close-strict-record-candidate` 是 release close 链路里更严格的一层最终候选输入面。它把 release evidence bundle、final evidence freeze、post-publish validation、release close candidate validation、final close decision validation、real external proof overlay、release issue close record overlay candidate、owner external execution result backfill kit 和 owner input cross-hash audit 绑定到同一份 strict candidate 中。

它仍然不是最终关闭记录，不会执行发布，不会批准公开发布，也不会关闭 release issue。当前状态必须保持：

- `candidateState=blocked-release-close-strict-record-owner-input-required`
- `validationState=blocked-release-close-strict-record-owner-input-required`
- `failedBlockerCount=0`
- `failedActionRequiredCount>=1`
- `mismatchedHashCount=0`
- `missingOwnerInputCount>=1`
- `missingRealProofCount>=1`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

## 生成与校验

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCloseStrictRecordCandidate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCloseStrictRecordCandidate.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
```

输出：

- `artifacts/final-release/release-close-strict-record-candidate.json`
- `artifacts/final-release/release-close-strict-record-candidate.md`
- `artifacts/final-release/release-close-strict-record-candidate-validation.json`
- `artifacts/final-release/release-close-strict-record-candidate-validation.md`

## 严格绑定范围

strict candidate 至少绑定以下本地 evidence path 与 SHA256：

- `release-evidence-bundle.json`
- `final-evidence-freeze.json`
- `post-publish-verification-validation.json`
- `release-issue-close-record-candidate-validation.json`
- `release-issue-final-close-decision-validation.json`
- `real-external-proof-overlay-pack-validation.json`
- `release-issue-close-record-overlay-candidate-validation.json`
- `owner-external-execution-result-backfill-kit-validation.json`
- `owner-input-cross-hash-audit-validation.json`

这些 hash 必须与当前本地文件一致。任何 mismatch 都是 blocker，但 `mismatchedHashCount=0` 也只能证明本地文件一致，不能证明 release close ready。

## 不能误读

- strict candidate 只是最终 close record 的候选输入面，不是最终 close record。
- hash consistency 不能替代真实 Owner approval、post-publish proof、clean consumer runtime proof 或 rollback approval。
- owner input placeholder 未替换时必须 blocked。
- final close decision 未真实回填时必须 blocked。
- post-publish verification 未真实通过时必须 blocked。
- 最终关闭仍必须通过 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`。
