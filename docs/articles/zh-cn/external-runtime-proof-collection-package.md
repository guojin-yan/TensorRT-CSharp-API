# External Runtime Proof Collection Package

`external-runtime-proof-collection-package` 是给 release owner 在兼容 CUDA / TensorRT 主机上一次性复制执行的 proof 收集包。它比 backfill plan 更接近操作现场：包含 package consumer smoke、包哈希、smoke log 哈希、真实 record 回填、`-RequireExistingLog -FailOnNotProof` 校验，以及聚合证据刷新的顺序。

它仍然不是 proof，不发布包，也不批准公开发布。真实门禁只接受通过 validator 晋级的 `artifacts/final-release/external-runtime-proof-record.json`。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofCollectionPackage.ps1
```

输出：

- `artifacts/final-release/external-runtime-proof-collection-package.json`
- `artifacts/final-release/external-runtime-proof-collection-package.md`

默认边界必须保持：

- `recordKind=external-runtime-proof-collection-package`
- `packageState=owner-action-required`
- `compatibleHostRequired=true`
- `performsPublish=false`
- `approvesPublicRelease=false`
- `canPromoteRuntimeProof=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionEvidence=false`

## 执行顺序

collection package 的 `copyableExecutionOrder` 面向 owner 直接复制使用：

1. 刷新 final package review、compatible-host runbook 和 external proof input template。
2. 在兼容主机执行 `Test-PackageConsumer.ps1 -RunSmoke -KeepConsumerOutput`。
3. 计算 managed/runtime nupkg SHA256 与 smoke log SHA256。
4. 将 input template 复制为真实 `external-runtime-proof-record.json`。
5. 回填 clean consumer identity、host metadata、package hashes、commands、stdout/stderr summary 和 smoke results。
6. 执行 `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof`。
7. 真实 record 验证通过后刷新 release evidence、promotion issue、freeze summary 和 freeze validation。

## 证据边界

- collection package / runbook / handoff / template / draft / example 都不是 runtime proof。
- `blocked-by-cuda-driver` 不是 smoke passed。
- dependency-probe-only 不能提升 `package-consumer-runtime` proof。
- local package inventory 不能替代兼容主机 smoke。
- 只有真实 record 中 package hash、host metadata、runtime key、smoke command、stdout/stderr summary 和 log SHA256 全部通过 validator，才能提升 `canPromoteRuntimeProof`。

## 不可替代 Proof 清单

以下材料只能帮助 owner 执行或审阅，不能单独作为 release close proof：

- collection package、runbook、handoff、template、input-template、draft、example。
- local feed、local inventory、package hash 草稿、dependency-probe-only 输出。
- `blocked-by-cuda-driver`、`pending-compatible-host-execution`、`build-only`、`precheck`。
- 没有 `stdoutSummary` / `stderrSummary` 人工复核摘要的 logPath。
- 没有 `-RequireExistingLog` hash 校验的 smoke log。

owner 应在兼容 CUDA/TensorRT 主机上填写真实 `external-runtime-proof-record.json`，并运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 `
  -InputPath artifacts/final-release/external-runtime-proof-record.json `
  -RequireExistingLog `
  -FailOnNotProof
```

## 与聚合记录的关系

release evidence、promotion issue、freeze summary 和 freeze checklist 会展示本 collection package 的 state、step count 和 copyable execution order count。但这些聚合字段只是 owner action visibility，不会改变 `canCloseReleaseIssue=false`，也不会把 collection package 误升为 proof。
