# Real External Proof Record Import Validator

`real-external-proof-record-import-validator` 将 Owner result import 转成严格真实 proof 导入合同。它负责说明哪些字段、文件、hash、validator output 和 forbidden substitute 检查必须满足，才能进入后续 promotion guard 或 release close 记录。

## 覆盖范围

- 6 条 proof lane 均生成 import contract。
- 每个 contract 包含 lane identity、strict validator command、forbidden substitutes 和 blocked reasons。
- 默认 `validatorState=blocked-real-external-proof-record-import-required`。
- 默认 `canPromoteRuntimeProof=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`。

## 命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealExternalProofRecordImportValidator.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealExternalProofRecordImportValidator.ps1 -Strict
```

## 产物

- `artifacts/final-release/real-external-proof-record-import-validator.json`
- `artifacts/final-release/real-external-proof-record-import-validator.md`
- `artifacts/final-release/real-external-proof-record-import-validator-validation.json`
- `artifacts/final-release/real-external-proof-record-import-validator-validation.md`

## 边界

该 validator 是 blocked contract validation，不是 proof by itself。合同形状正确、blocked reasons 可读、命令齐全，都不能替代真实 runtime proof records、publish approval、post-publish proof 或 release close approval。
