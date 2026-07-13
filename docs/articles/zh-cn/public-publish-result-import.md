# Public Publish Result Import

`public-publish-result-import` 只导入 Owner 填写的真实公开发布结果，不执行发布命令。导入结果需要由 validator 检查 URL、hash、timestamp、package id/version、transcript path/hash 和人工确认字段。

默认状态保持 blocked / non-proof。local `.nupkg`、local feed、`file://` URL、ProjectReference、direct nupkg 和 command handoff 都不能替代真实公开包结果。

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-PublicPublishResultOwnerInput.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PublicPublishResultImport.ps1 -Strict
```
