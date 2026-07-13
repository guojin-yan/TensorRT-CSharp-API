# Final Release Close Record Real Validator

`final-release-close-record-real-validator` 是最终关闭记录的真实字段合同验证器。它把 release issue、公开发布结果、clean consumer proof、forbidden substitute、真实 proof import bridge、最终 readiness 和分类审计汇总为 13 个必填字段合同。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalReleaseCloseRecordRealValidator.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalReleaseCloseRecordRealValidator.ps1 -Strict
```

## 产物

- `artifacts/final-release/final-release-close-record-real-validator.json`
- `artifacts/final-release/final-release-close-record-real-validator.md`
- `artifacts/final-release/final-release-close-record-real-validator-validation.json`
- `artifacts/final-release/final-release-close-record-real-validator-validation.md`

## 边界

默认状态为 `blocked-final-release-close-record-real-proof-required`，`RequiredFields=13` 且 `Blocked=13`。它只是字段合同验证器，不发布包、不批准公开发布、不生成 runtime proof、post-publish proof 或 release close proof，也不关闭 release issue。

必须保持：

- `notExecutedByAutomation=true`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isPostPublishProof=false`
- `isReleaseCloseProof=false`
- `isReleaseCloseRecordProof=false`

## Owner 下一步

Owner 必须补齐真实 release issue id/url、真实公开包来源、下载包 hash、发布 transcript、仓库外 clean consumer smoke、forbidden substitute 清零、真实 proof import lane、最终 readiness 和分类审计。任一字段仍 blocked 时，release issue 必须保持打开。
