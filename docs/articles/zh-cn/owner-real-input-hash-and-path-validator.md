# Owner 真实输入 Hash 与路径校验器

`owner-real-input-hash-and-path-validator` 校验 Owner 输入中的本地路径、公开 URL、GitHub Release asset、public download SHA256、clean consumer 日志 SHA256 和 host metadata。

它只做字段级检查，不下载公开 URL、不访问公开包、不执行 package push。hash match 也不能替代 runtime proof、post-publish proof 或 release close approval。

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerRealInputHashAndPathValidator.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerRealInputHashAndPathValidator.ps1 -Strict
```
