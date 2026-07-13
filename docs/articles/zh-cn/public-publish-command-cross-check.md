# Public Publish Command Cross-Check

`public-publish-command-cross-check` 是公开发布命令的最终人工核对表。它检查 managed package ID/version、runtime key、发布渠道、URL、SHA256、rollback 记录和凭据处理是否已经具备真实 Owner 输入。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PublicPublishCommandCrossCheck.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PublicPublishCommandCrossCheck.ps1 -Strict
```

## 产物

- `artifacts/final-release/public-publish-command-cross-check.json`
- `artifacts/final-release/public-publish-command-cross-check.md`
- `artifacts/final-release/public-publish-command-cross-check-validation.json`
- `artifacts/final-release/public-publish-command-cross-check-validation.md`

## 边界

默认状态为 `blocked-public-publish-command-cross-check-owner-action-required`。它只做发布前核对，不生成真实发布证明，也不会执行任何上传动作。

必须保持：

- `notExecutedByAutomation=true`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isPostPublishProof=false`
- `isReleaseCloseProof=false`

## 使用方式

Owner 应先核对命令中的 package ID、version、target source、API key 来源和 rollback plan，再执行真实发布。执行完成后，结果应进入 `public-publish-result-owner-input` 和后续 import/clean consumer 验证链路。该 cross-check 不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。
