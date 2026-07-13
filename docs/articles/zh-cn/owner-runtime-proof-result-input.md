# Owner Runtime Proof Result Input

`owner-runtime-proof-result-input` 将 runtime proof execution input record 进一步展开成 Owner 可回填的真实执行结果模板，并用 strict validator 检查是否仍存在 placeholder、缺失文件、缺失 SHA256 或 proof substitute。

## 覆盖范围

- 6 条 proof lane 均生成 `resultInputId`，继承 `executionInputId`、`candidateId`、`proofLane` 和 `runtimePackageKey`。
- 每条记录要求 Owner 填写 host metadata、package identity、执行命令、working directory、stdout/stderr、merged transcript、validator output、exit code、时间戳、reviewer 和 review timestamp。
- SHA256 字段必须是 64 位十六进制，并且日志/validator 路径必须指向真实存在的文件。
- 默认 `templateState=blocked-owner-runtime-proof-result-input-required`，不提升 runtime proof，不允许 public publish，也不允许 close release issue。

## 命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerRuntimeProofResultInputTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerRuntimeProofResultInput.ps1 -Strict
```

## 产物

- `artifacts/final-release/owner-runtime-proof-result-input.template.json`
- `artifacts/final-release/owner-runtime-proof-result-input.template.md`
- `artifacts/final-release/owner-runtime-proof-result-input-validation.json`
- `artifacts/final-release/owner-runtime-proof-result-input-validation.md`

## 边界

该模板和 validator 只是 Owner 结果回填入口。placeholder、hash 槽位、日志路径、validator output、reviewer 字段和 non-substitute confirmations 都不能替代真实 runtime execution proof、post-publish proof、package publish、rollback approval、final close decision 或 release issue close approval。
