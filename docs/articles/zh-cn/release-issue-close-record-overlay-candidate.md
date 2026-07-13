# Release Issue Close Record Overlay Candidate

`release-issue-close-record-overlay-candidate` 是 release issue close record 的候选输入映射层。它把 release evidence bundle、final evidence freeze、post-publish validation、release close candidate validation、final close decision validation 和 real external proof overlay pack validation 的路径与 SHA256 统一映射到 close record 的 Owner 回填面。

它不是关闭记录本身，不会关闭 release issue，也不会替代 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`。当前状态必须保持：

- `candidateState=blocked-release-close-real-proof-required`
- `validationState=blocked-release-close-overlay-owner-input-required`
- `failedBlockerCount=0`
- `failedActionRequiredCount>=1`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

## 生成与校验

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealExternalProofOverlayPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealExternalProofOverlayPack.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseRecordOverlayCandidate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecordOverlayCandidate.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
```

输出：

- `artifacts/final-release/release-issue-close-record-overlay-candidate.json`
- `artifacts/final-release/release-issue-close-record-overlay-candidate.md`
- `artifacts/final-release/release-issue-close-record-overlay-candidate-validation.json`
- `artifacts/final-release/release-issue-close-record-overlay-candidate-validation.md`

## 必填真实输入

该 candidate 保留 Owner 必填 placeholder，直到真实 post-publish proof、rollback review 和最终 close decision 全部回填：

- `rollbackPlan`
- `rollbackOwner`
- `rollbackTrigger`
- `ownerFinalCloseDecision`
- `releaseIssueId`
- `releaseIssueUrl`

## 不能误读

- candidate 中的 hash 只能证明本地 artifact 映射一致，不能证明真实外部执行完成。
- placeholder 字段未被真实 Owner 输入替换前，validation 必须保持 blocked。
- `canCloseReleaseIssue=false` 必须保持到真实 close record 通过 strict validator。
- 它不能替代 post-publish verification、clean external consumer smoke、rollback approval 或 final owner decision。
