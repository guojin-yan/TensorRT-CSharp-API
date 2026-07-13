# 真实公开发布 Owner 执行包

`public-release-owner-execution-package` 是发布候选最终 Owner 执行面的 blocked/non-proof artifact。它用于整理真实公开发布所需的命令、字段和证据路径，但不会执行公开发布、不会上传包、不会关闭 release issue，也不能替代 runtime proof 或 post-publish proof。

## 当前状态

- 状态：`blocked-public-release-owner-execution-package-owner-proof-required`
- 默认结果：`Passed=false`
- 边界：`not runtime proof; not post-publish proof; not publish approval; not release close approval; not package push`
- 适用阶段：真实公开发布前 Owner 手动执行与证据回填准备。

## 覆盖范围

该 artifact 汇总 NuGet.org、GitHub Packages、GitHub Release、包/hash/release notes 核对等最终人工发布命令面，但不执行任何上传。

## Owner 必填字段

- `ownerName`
- `packageId`
- `packageVersion`
- `nugetOrgPackageUrl`
- `githubPackagesUrl`
- `publishedNupkgSha256`
- `publishedSymbolsSha256`
- `releaseNotesUrl`
- `rollbackPlanReviewed`
- `ownerPublishDecision`

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PublicReleaseOwnerExecutionPackage.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PublicReleaseOwnerExecutionPackage.ps1 -Strict
```
