# Owner External Proof Execution Result Import

`owner-external-proof-execution-result-import` 将 Owner 外部执行命令包中的结果槽位投影为导入记录。它用于检查每条 proof lane 是否已经具备真实文件、SHA256、host/package metadata、validator output 和 reviewer 字段。

## 覆盖范围

- 6 条 proof lane 均生成 result import item。
- 默认所有 item 均为 blocked，直到 Owner 提供真实执行结果。
- `missingRealEvidenceCount` 汇总缺失字段和文件证据。
- 默认 `importState=blocked-owner-external-proof-execution-result-required`。

## 命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-OwnerExternalProofExecutionResult.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerExternalProofExecutionResultImport.ps1 -Strict
```

## 产物

- `artifacts/final-release/owner-external-proof-execution-result-import.json`
- `artifacts/final-release/owner-external-proof-execution-result-import.md`
- `artifacts/final-release/owner-external-proof-execution-result-import-validation.json`
- `artifacts/final-release/owner-external-proof-execution-result-import-validation.md`

## 边界

该 import 是 blocked owner input validation，不是 runtime proof。缺失真实文件、hash、metadata、validator output 或 reviewer 字段时，不能发布、不能晋级 proof，也不能关闭 release issue。
