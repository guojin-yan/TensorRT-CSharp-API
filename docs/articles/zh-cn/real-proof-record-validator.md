# Real Proof Record Validator

`real-proof-record-validator` 将 strict candidate、Owner field delta 与 promotion guard 串成真实 proof record 的严格字段契约。

它仍然是 blocked/non-proof surface：默认输出只说明未来真实 proof record 必须具备哪些 runtime evidence、host metadata、package identity、command capture、log hash、validator output 与 forbidden-substitute 检查。

## 生成产物

- `artifacts/final-release/real-proof-record-validator.json`
- `artifacts/final-release/real-proof-record-validator.md`
- `artifacts/final-release/real-proof-record-validator-validation.json`
- `artifacts/final-release/real-proof-record-validator-validation.md`

## 执行

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealProofRecordValidator.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealProofRecordValidator.ps1 -Strict
```

## 边界

- 不能替代真实 runtime smoke 日志。
- 不能替代 clean package-consumer proof。
- 不能替代 post-publish verification。
- 不能设置 `canPublishPublicly=true`。
- 不能设置 `canCloseReleaseIssue=true`。
- 不能设置 `isRuntimeExecutionProof=true` 或 `isReleaseCloseProof=true`。
