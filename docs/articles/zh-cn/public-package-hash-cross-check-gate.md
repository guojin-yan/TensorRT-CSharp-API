# 公开包 Hash 交叉核对 Gate

`public-package-hash-cross-check-gate` 用于记录公开包 hash 与本地 freeze/release package hash 的交叉核对要求。它现在显式承接 `public-package-proof-owner-input` 中的 NuGet package source、GitHub Release asset path/hash、managed/runtime public download URL/SHA256 和 ownerReview 字段。

它默认 `Passed=false`，只接收 Owner 回填的公开包 URL 和 hash 字段，不下载公开包、不访问外部源、不执行 package push，也不把 hash match 解释为 post-publish proof 或 runtime proof。

## 边界

- 不是 runtime proof。
- 不是 post-publish proof。
- 不是 publish approval。
- 不是 release close approval。
- 不是 package push。

## Owner 输入

- `publicPackageUrl`
- `publishedNupkgSha256`
- `publishedSymbolsSha256`
- `localFreezeNupkgSha256`
- `localFreezeSymbolsSha256`
- `hashComparisonResult`
- `ownerHashReviewDecision`

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PublicPackageHashCrossCheckGate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PublicPackageHashCrossCheckGate.ps1 -Strict
```

## Owner 输入交叉核对

- NuGet package source 必须与后续 clean consumer / post-publish owner input 使用的公开源一致。
- GitHub Release URL、tag、managed/runtime asset path/hash 只用于 Owner 回填和交叉核对，不会被脚本自动上传或下载。
- managed/runtime public download SHA256、published nupkg SHA256 和本地 freeze SHA256 mismatch 时必须保持 blocked。
- ownerReview reviewer/reviewedAtUtc 只是人工复核记录，不是 publish approval。
