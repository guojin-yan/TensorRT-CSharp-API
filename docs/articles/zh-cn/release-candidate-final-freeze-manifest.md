# Release Candidate Final Freeze Manifest

`release-candidate-final-freeze-manifest` 是公开发布 Owner 手动执行前的最终冻结清单。它记录 release-facing 文档、release evidence、最终审计包和目标测试结果的本地 SHA256，帮助 Owner 在复制任何发布命令前确认当前候选状态。

该 manifest 只做本地 hash inventory 和边界复核，不执行发布、不上传包、不生成 runtime proof、不生成 post-publish proof，也不能批准公开发布或关闭 release issue。

## 产物

- `artifacts/final-release/release-candidate-final-freeze-manifest.json`
- `artifacts/final-release/release-candidate-final-freeze-manifest.md`
- `artifacts/final-release/release-candidate-final-freeze-manifest-validation.json`
- `artifacts/final-release/release-candidate-final-freeze-manifest-validation.md`

## 当前边界

- `performsPublish=false`
- `canPromoteRuntimeProof=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isReleaseCloseProof=false`
- `isPostPublishProof=false`

## 覆盖范围

冻结清单会覆盖 README、docs 索引、release evidence bundle、classification audit、final owner decision audit、final post-publish audit pack，以及目标 TRX。所有 artifact 都必须存在并有 SHA256，但这些 hash 只能证明本地冻结状态，不能替代真实 public package proof 或 post-publish clean consumer proof。

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateFinalFreezeManifest.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCandidateFinalFreezeManifest.ps1 -Strict
```

当前预期状态是 `release-candidate-final-freeze-manifest-ready-for-owner-handoff`，但这只表示可交给 Owner 复核，不表示可以自动发布或关闭 release issue。
