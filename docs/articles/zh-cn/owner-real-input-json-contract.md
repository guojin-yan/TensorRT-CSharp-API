# Owner 真实输入 JSON 合同

`owner-real-input-json-contract` 定义真实 Owner 输入 JSON 的字段合同，覆盖 NuGet/GitHub Release 公开包、managed/runtime public download hash、仓库外 clean consumer 执行日志、runtime host metadata、post-publish verification、rollback review 和 Owner final close decision。

它默认 `Passed=false`，只定义输入字段，不执行发布、不下载包、不关闭 release issue，也不能替代 runtime proof 或 post-publish proof。

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerRealInputJsonContract.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerRealInputJsonContract.ps1 -Strict
```
