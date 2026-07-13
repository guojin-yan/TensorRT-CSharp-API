# Public Publish Owner Manual Command Handoff

`public-publish-owner-manual-command-handoff` 是公开发布命令的 Owner 手动执行交接包。它可以列出 `dotnet nuget push` 等命令模板，但所有命令都必须保持 `notExecutedByAutomation=true`、placeholder-only 和 owner-execution-only。

该 handoff 不执行真实发布，不生成可直接运行的 materialized command，不把本地包、local feed、ProjectReference、dry-run、command placeholder 或 validator contract 当作 proof。

## 产物

- `artifacts/final-release/public-publish-owner-manual-command-handoff.json`
- `artifacts/final-release/public-publish-owner-manual-command-handoff.md`
- `artifacts/final-release/public-publish-owner-manual-command-handoff-validation.json`
- `artifacts/final-release/public-publish-owner-manual-command-handoff-validation.md`

## 当前边界

- `notExecutedByAutomation=true`
- `performsPublish=false`
- `canPromoteRuntimeProof=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isReleaseCloseProof=false`
- `isPostPublishProof=false`

## 命令范围

handoff 包含 publish 前 freeze review、NuGet public push placeholder、GitHub Packages push placeholder、post-publish clean consumer smoke placeholder 和 strict close validator placeholder。Owner 必须在外部终端手动替换占位符、执行命令并回填真实公开包与 post-publish proof 字段。

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PublicPublishOwnerManualCommandHandoff.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PublicPublishOwnerManualCommandHandoff.ps1 -Strict
```

validator 会要求出现 `dotnet nuget push` 时仍保持 `notExecutedByAutomation=true`，且所有 command item 都不能 `performsPublish=true`。
