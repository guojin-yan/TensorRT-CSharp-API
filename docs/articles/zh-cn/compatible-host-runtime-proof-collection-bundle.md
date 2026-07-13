# 兼容主机 Runtime Proof Collection Bundle

`compatible-host-runtime-proof-collection-bundle` 是给 release owner 或外部兼容 GPU 主机执行者使用的一次性收集包。它把 runbook、input template、package consumer smoke、log/package hash、真实 record 填写、`-FailOnNotProof` 验证，以及 owner-facing 证据刷新串成一个可复制流程。

它不是 runtime proof，不发布包，也不批准公开发布。真实门禁仍然只接受在兼容 NVIDIA driver / CUDA runtime / TensorRT runtime 主机上跑通的 `external-runtime-proof-record.json`，并且必须通过：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 `
  -InputPath artifacts/final-release/external-runtime-proof-record.json `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 `
  -RequireExistingLog `
  -FailOnNotProof
```

## 生成

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CompatibleHostRuntimeProofCollectionBundle.ps1 `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22
```

输出：

- `artifacts/final-release/compatible-host-runtime-proof-collection-bundle.json`
- `artifacts/final-release/compatible-host-runtime-proof-collection-bundle.md`

## 快速执行顺序

在兼容 CUDA / TensorRT 主机上，从仓库根目录按 bundle 的 `copyableExecutionOrder` 顺序执行。任何一步失败都停止，不允许把失败的 smoke 或 validator 输出改写成 proof：

1. 刷新 compatible-host runbook 和 external proof input template。
2. 运行 `Test-PackageConsumer.ps1 -RunSmoke -KeepConsumerOutput`，必须得到 package-consumer runtime smoke passed。
3. 计算真实 smoke log、managed nupkg、runtime nupkg 的 SHA256。
4. 复制 input template 为 `external-runtime-proof-record.json`，填入真实 host、package、command、hash 和 results。
5. 复核真实 smoke log，回填 `results.stdoutSummary` 和 `results.stderrSummary`；stderr 为空时也必须写入 `no-stderr-emitted`。
6. 运行 `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof`。
7. 导出 `package-consumer-runtime-proof-owner-input.template.json`，复制为 owner input，回填真实 public package source、clean external consumer、host、log、hash 和 stdout/stderr 字段。
8. 运行 `Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict`，通过后再运行 `Import-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict`。
9. 验证通过后再刷新 release evidence、owner approval input、owner decision、publish checklist 和 promotion issue。

## 执行前检查

| 项 | 必须具备 | 原因 |
| --- | --- | --- |
| 兼容 GPU 主机 | NVIDIA driver 可以运行目标 CUDA / TensorRT runtime | 当前 blocker 是 driver/runtime mismatch；`blocked-by-cuda-driver` 不是 smoke passed |
| 干净 consumer | smoke 使用 nupkg，不使用 `ProjectReference` | release gate 要求 package-consumer-runtime proof |
| 保留 smoke log | log path 存在且可计算 SHA256 | validator 需要 existing log 和匹配的 `command.logSha256` |
| 复核 stdout/stderr | `results.stdoutSummary` 与 `results.stderrSummary` 均非空 | log hash 只能证明文件一致，不能证明输出已审阅；无 stderr 时写明 `no-stderr-emitted` |
| 包 hash | managed/runtime nupkg SHA256 都已记录 | 证明 clean consumer 消耗的是可追踪包 |
| 真实 record | `recordKind=external-runtime-proof-record` 且 `templateOnly=false` | template、draft、example、collection bundle 都不是 proof |
| proof validator | `-RequireExistingLog -FailOnNotProof` 通过 | 只有该门禁通过才能提升为 package-consumer-runtime proof |
| owner input strict validator | `Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict` 在 owner-filled input 上通过 | Owner input 必须证明 clean external consumer path、public package source、no ProjectReference、package/log SHA256、host metadata 和 stdout/stderr summary |

## 边界

- `recordKind=compatible-host-runtime-proof-collection-bundle`
- `ownerRuntimeSmokeRunbookState=blocked-owner-compatible-host-runtime-smoke`
- `performsPublish=false`
- `approvesPublicRelease=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `canPromoteRuntimeProof=false`
- `isRuntimeExecutionEvidence=false`
- `blocked-by-cuda-driver` 不能写成 `smokeStatus=passed`
- `Smoke=not-requested` 不能写成 runtime smoke proof
- collection bundle / runbook / handoff / checklist / template / draft / example 都不能替代真实 proof

## 真实 Proof 必填项

真实 `external-runtime-proof-record.json` 至少需要：

| 字段 | 期望值 |
| --- | --- |
| `recordKind` | `external-runtime-proof-record` |
| `templateOnly` | `false` |
| `proofClassification` | `package-consumer-runtime` |
| `runtimePackageKey` | `win-x64-trt11.0-cuda13.2-cudnn9.22` |
| `packageSource.runtimePackageKey` | `win-x64-trt11.0-cuda13.2-cudnn9.22` |
| `packageSource.managedNupkgSha256` | 64 位 SHA256 |
| `packageSource.runtimeNupkgSha256` | 64 位 SHA256 |
| `command.exitCode` | `0` |
| `command.logSha256` | 与真实 smoke log 匹配的 64 位 SHA256 |
| `results.stdoutSummary` | 非空 stdout 摘要 |
| `results.stderrSummary` | 非空 stderr 摘要或 `no-stderr-emitted` |
| `results.smokeStatus` | `passed` |
| `results.nativeAssetsCopied` | `true` |
| `isRuntimeExecutionEvidence` | `true` |
| `canPromoteRuntimeProof` | `true` |

## Owner 使用方式

1. 在兼容 CUDA/NVIDIA driver/TensorRT runtime 主机上生成 collection bundle。
2. 按 bundle 里的命令刷新 input template 并运行 package consumer smoke。
3. 计算 smoke log、managed nupkg、runtime nupkg 的 SHA256。
4. 将 input template 复制为 `external-runtime-proof-record.json` 并填入真实主机、命令、hash、结果。
5. 复核 stdout/stderr 摘要；stderr 为空时也写明 `no-stderr-emitted`。
6. 运行 `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof`。
7. 生成并填写 `package-consumer-runtime-proof-owner-input.json`。
8. 运行 `Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict` 和 `Import-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict`。
9. 只有验证通过后，才刷新 release evidence / owner approval / owner decision / publish checklist / promotion issue。

如果仍然是 CUDA error 35 或 `blocked-by-cuda-driver`，保持 blocker，不允许把 collection bundle 写成 runtime proof。
