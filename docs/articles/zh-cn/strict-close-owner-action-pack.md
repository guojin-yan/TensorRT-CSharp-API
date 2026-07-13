# StrictClose Owner 行动包

`strict-close-owner-action-pack` 给 Owner 一页式行动包，列出需要执行的外部命令、需要复制的 hash、需要保留的日志和需要填写的最终决策。

它是行动指引，不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-StrictCloseOwnerActionPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StrictCloseOwnerActionPack.ps1 -Strict
```
