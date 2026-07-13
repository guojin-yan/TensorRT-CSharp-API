# 发布后 Owner 验证采集包

`post-publish-owner-verification-kit` 定义真实发布后 Owner 必须回填的公开 URL、hash、clean restore、download proof、runtime proof 与 known limitations 链接。它不执行发布，也不允许关闭 release issue。

## 当前状态

- 状态：`blocked-post-publish-owner-verification-kit-owner-proof-required`
- 默认结果：`Passed=false`
- 边界：`not runtime proof; not post-publish proof; not publish approval; not release close approval; not package push`

## Owner 必填字段

- `nugetOrgPackageUrl`
- `publishedVersion`
- `publishedNupkgSha256`
- `cleanRestoreProofPath`
- `cleanConsumerProofPath`
- `runtimeProofPath`
- `knownLimitationsUrl`
- `ownerPostPublishDecision`

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishOwnerVerificationKit.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishOwnerVerificationKit.ps1 -Strict
```
