# Public Publish Result Owner Input

`public-publish-result-owner-input` 定义 Owner 在真实公开发布完成后需要回填的 package URL、channel、timestamp、nupkg SHA256、publish transcript hash 和人工复核字段。

当前合同已经扩展为最终公开发布结果的 Owner 输入面，除旧字段外，还要求保留以下真实证据字段：

- `nugetPackageSource`：真实公开包源，例如 nuget.org 或 GitHub Packages 的公开源。
- `githubRelease.releaseUrl` / `githubRelease.tagName`：真实 GitHub Release 页面和 tag。
- `githubRelease.managedAssetPath` / `githubRelease.managedAssetSha256`：managed 包在 GitHub Release 中的 asset 路径和 SHA256。
- `githubRelease.runtimeAssetPath` / `githubRelease.runtimeAssetSha256`：runtime 包在 GitHub Release 中的 asset 路径和 SHA256。
- `managedPackage.packageUrl` / `managedPackage.publicDownloadUrl` / `managedPackage.publicDownloadSha256`：managed 包公开页面、公开下载地址和下载后 hash。
- `runtimePackage.packageUrl` / `runtimePackage.publicDownloadUrl` / `runtimePackage.publicDownloadSha256`：runtime 包公开页面、公开下载地址和下载后 hash。
- `ownerReview.reviewer` / `ownerReview.reviewedAtUtc` / `ownerReview.approvalState`：Owner 对公开包结果的复核记录。
- `rollbackReview.reviewedBy` / `rollbackReview.reviewedAtUtc` / `rollbackReview.rollbackPlanSha256` / `rollbackReview.decision`：rollback plan 复核记录。
- `finalCloseDecision.decision` / `finalCloseDecision.decidedAtUtc` / `finalCloseDecision.ownerReviewer` / `finalCloseDecision.releaseIssueUrl`：最终关闭决策输入，但只有全部真实 proof validator 通过后才允许后续 close readiness 变化。

该模板默认是 blocked / non-proof。它不执行发布、不上传包、不代表 publish approval、不代表 runtime proof、不代表 post-publish proof、不代表 release close approval，也不能替代 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`。

本地 feed、direct `.nupkg`、ProjectReference、build-only、dependency-probe-only、blocked-by-driver-only、dashboard、runbook、template 或 hash slot 均不能替代上述 Owner 真实输入。

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PublicPublishResultOwnerInputTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PublicPublishResultOwnerInput.ps1 -Strict
```
