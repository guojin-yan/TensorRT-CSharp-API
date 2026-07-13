# Owner 真实输入导入预检

`owner-real-input-import-preflight` 定义 Owner 真实输入 JSON 导入前的预检面，覆盖输入路径、SHA256、字段合同版本、公开 URL 字段、日志 hash、host metadata 和禁止替代项计数。

它默认 `Passed=false`，只检查 Owner 输入形状和缺失字段，不会把 template、draft、local feed、ProjectReference、direct `.nupkg` 或 real proof readiness gate 晋级为真实 proof。

## 边界

- 不是 runtime proof。
- 不是 post-publish proof。
- 不是 publish approval。
- 不是 release close approval。
- 不是 package push。

## Owner 输入

- `ownerInputJsonPath`
- `ownerInputSha256`
- `contractVersion`
- `publicUrlFieldCount`
- `sha256FieldCount`
- `hostMetadataFieldCount`
- `forbiddenSubstituteCounts`
- `ownerImportDecision`

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerRealInputImportPreflight.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerRealInputImportPreflight.ps1 -Strict
```
