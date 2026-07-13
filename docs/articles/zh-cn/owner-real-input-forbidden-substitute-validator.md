# Owner 真实输入禁止替代项校验器

`owner-real-input-forbidden-substitute-validator` 检查 Owner 输入中是否出现 local feed、ProjectReference、direct `.nupkg`、pre-publish package、template、draft、dry-run、build-only、dependency probe、blocked-by-driver、local-only scan、manual handoff 或 real proof readiness gate 等禁止替代项。

发现替代项时必须保持 blocked，不能把本地或候选材料解释为真实 proof。

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerRealInputForbiddenSubstituteValidator.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerRealInputForbiddenSubstituteValidator.ps1 -Strict
```
