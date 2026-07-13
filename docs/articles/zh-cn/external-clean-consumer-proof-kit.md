# 外部干净 Consumer Proof 采集包

`external-clean-consumer-proof-kit` 定义仓库外 clean consumer 项目的真实验证步骤、日志字段、包源字段与不可替代边界。它只提供 Owner 执行与回填要求，不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

## 当前状态

- 状态：`blocked-external-clean-consumer-proof-kit-owner-proof-required`
- 默认结果：`Passed=false`
- 边界：`not runtime proof; not post-publish proof; not publish approval; not release close approval; not package push`

## Owner 必填字段

- `consumerProjectPath`
- `packageSource`
- `packageId`
- `packageVersion`
- `restoreLogSha256`
- `smokeLogSha256`
- `runtimePackageKey`
- `hostMetadata`
- `projectReferenceCount`
- `directNupkgReferenceCount`

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalCleanConsumerProofKit.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalCleanConsumerProofKit.ps1 -Strict
```
