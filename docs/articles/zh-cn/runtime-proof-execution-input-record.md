# Runtime Proof Execution Input Record

`runtime-proof-execution-input-record` 把 `owner-real-proof-execution-closure-pack` 的 6 条 closure item 转成 Owner 可填写的真实执行输入记录。它只定义字段、命令、日志、hash 和 reviewer 槽位，不运行 proof。

## 覆盖范围

- 每条 proof lane 都有 runtime package key、host metadata、package identity、command line、working directory、stdout/stderr、merged transcript、validator output、owner reviewer 和 review timestamp。
- SHA256 字段采用 owner-fill 模式，默认 placeholder 会被 strict validation 拦截。
- `forbiddenSubstituteChecks` 明确拒绝 local feed、ProjectReference、direct nupkg、DependencyProbe-only、sidecar-only、build-only、precheck-only。
- 默认 `inputState=blocked-runtime-proof-execution-input-required`，并保持 `canPromoteRuntimeProof=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`。

## 命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RuntimeProofExecutionInputRecord.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RuntimeProofExecutionInputRecord.ps1 -Strict
```

## 产物

- `artifacts/final-release/runtime-proof-execution-input-record.json`
- `artifacts/final-release/runtime-proof-execution-input-record.md`
- `artifacts/final-release/runtime-proof-execution-input-record-validation.json`
- `artifacts/final-release/runtime-proof-execution-input-record-validation.md`

## 边界

该记录是 Owner 输入面，不是 runtime proof、不是 package publish、不是 post-publish verification，也不是 release-close approval。只有真实外部执行日志、SHA256、包身份、host metadata、validator output、reviewer 和 timestamp 全部通过 strict validation 后，后续链路才能继续推进。
