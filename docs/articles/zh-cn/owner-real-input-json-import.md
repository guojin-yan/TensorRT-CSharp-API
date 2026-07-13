# Owner 真实输入 JSON 导入

`owner-real-input-json-import` 负责导入可选 Owner 输入 JSON。未提供真实输入时生成 blocked surface；提供路径时只读取本地 JSON 并记录存在性、SHA256、parse state、公开包字段、clean consumer 日志字段、host metadata、rollback review 和 final close decision，不访问网络。

导入成功仍不是 proof，不会设置 `canPublishPublicly=true` 或 `canCloseReleaseIssue=true`。

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerRealInputJsonImport.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerRealInputJsonImport.ps1 -Strict
```
