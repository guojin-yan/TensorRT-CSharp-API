# 兼容主机 Runtime Proof Runbook

`compatible-host-runtime-proof-runbook` 是给 release owner 或外部 GPU 主机执行者使用的零歧义运行手册。它不发布包、不批准公开发布，也不把当前本机的 `blocked-by-cuda-driver` 草稿改写成 proof。

当前 release 证据链已经能证明 package consumer 走到了 packaged runtime 边界，并且 managed/runtime nupkg SHA256、smoke log SHA256、no ProjectReference 等草稿信息已经可追踪。但真实发布门禁仍缺一件事：在兼容 NVIDIA driver / CUDA runtime / TensorRT runtime 的主机上跑通 clean package consumer smoke，并生成可通过 `-FailOnNotProof` 的 `external-runtime-proof-record.json`。

## 生成 Runbook

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CompatibleHostRuntimeProofRunbook.ps1 `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22
```

输出文件：

- `artifacts/final-release/compatible-host-runtime-proof-runbook.json`
- `artifacts/final-release/compatible-host-runtime-proof-runbook.md`

默认没有真实 proof 时应保持：

- `recordKind=compatible-host-runtime-proof-runbook`
- `runbookState=owner-action-required`
- `ownerRuntimeSmokeRunbookState=blocked-owner-compatible-host-runtime-smoke`
- `compatibleHostRequired=true`
- `performsPublish=false`
- `approvesPublicRelease=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionEvidence=false`
- `canPromoteRuntimeProof=false`
- `promotionBlockedReason=blocked-by-cuda-driver is not smoke passed`

## 兼容主机要求

执行主机必须满足以下条件，才有机会产出真实 runtime proof：

- 有可用 NVIDIA GPU。
- NVIDIA driver 支持目标 CUDA runtime，例如 `win-x64-trt11.0-cuda13.2-cudnn9.22` 对应 CUDA 13.2 runtime。
- 能加载 TensorRT 11.0、CUDA 13.2、cuDNN 9.22 的目标 runtime assets。
- 能执行 PowerShell、.NET SDK restore/build/test。
- 能保留完整 package consumer 输出目录与 smoke log。
- 能确认 clean consumer 没有 `ProjectReference`。

`nvidia-smi` 截图或 driver version 本身不是 proof。它只是 host metadata 的一部分，仍必须跑通 package consumer smoke。

## 执行命令

在 repo 根目录执行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofRecordInputTemplate.ps1 `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 `
  -SmokeRuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 `
  -RunSmoke `
  -KeepConsumerOutput

Get-FileHash -LiteralPath "artifacts/final-release/external-runtime-proof/win-x64-trt11.0-cuda13.2-cudnn9.22/package-consumer-smoke.log" `
  -Algorithm SHA256 | Select-Object -ExpandProperty Hash

Get-FileHash -LiteralPath "<downloaded-managed-nupkg>" -Algorithm SHA256 |
  Select-Object -ExpandProperty Hash

Get-FileHash -LiteralPath "<downloaded-runtime-nupkg>" -Algorithm SHA256 |
  Select-Object -ExpandProperty Hash

Copy-Item -LiteralPath artifacts/final-release/external-runtime-proof-record.input-template.json `
  -Destination artifacts/final-release/external-runtime-proof-record.json

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 `
  -InputPath artifacts/final-release/external-runtime-proof-record.json `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 `
  -RequireExistingLog `
  -FailOnNotProof

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1 `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22

Copy-Item -LiteralPath artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json `
  -Destination artifacts/final-release/package-consumer-runtime-proof-owner-input.json

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofOwnerInput.ps1 `
  -InputPath artifacts/final-release/package-consumer-runtime-proof-owner-input.json `
  -Strict

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-PackageConsumerRuntimeProofOwnerInput.ps1 `
  -InputPath artifacts/final-release/package-consumer-runtime-proof-owner-input.json `
  -Strict
```

在复制 input template 后，必须先复核真实 smoke log 并回填 `results.stdoutSummary` 与 `results.stderrSummary`。如果 stderr 为空，也必须在 `stderrSummary` 中写明 `no-stderr-emitted` 或等价复核说明，不能留空。

在复制 `package-consumer-runtime-proof-owner-input.template.json` 后，也必须回填真实 clean external consumer 路径、public package source、managed/runtime nupkg SHA256、host metadata、smoke log SHA256、stdout/stderr summary。`Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict` 不通过时，不能导入 owner input，更不能把 owner input 当成 runtime proof。

如果最后一条命令失败，不要手改 `canPromoteRuntimeProof=true`。应按 validation items 修补真实缺口，或保留 blocker。

## Record 回填字段

把 input-template 复制为 `external-runtime-proof-record.json` 后，必须回填或修改：

| 字段 | 目标值 | 说明 |
| --- | --- | --- |
| `recordKind` | `external-runtime-proof-record` | 必须是真实记录，不是 draft/template/example。 |
| `templateOnly` | `false` | 模板不能晋级。 |
| `proofClassification` | `package-consumer-runtime` | release runtime proof 只接受 clean package consumer runtime。 |
| `runtimePackageKey` | `win-x64-trt11.0-cuda13.2-cudnn9.22` | 必须匹配目标 runtime key。 |
| `packageSource.runtimePackageKey` | `win-x64-trt11.0-cuda13.2-cudnn9.22` | 必须证明消费的是同一个 runtime package。 |
| `packageSource.managedNupkgSha256` | 64 位 SHA256 | 对应实际被 clean consumer 消费的 managed 包。 |
| `packageSource.runtimeNupkgSha256` | 64 位 SHA256 | 对应实际被 clean consumer 消费的 runtime 包。 |
| `packageSource.noProjectReference` | `true` | 证明不是项目引用。 |
| `host.*` | 非空 | owner、machine、OS、GPU、driver、CUDA、TensorRT 信息必须可审查。 |
| `command.exitCode` | `0` | smoke 命令必须成功。 |
| `command.logPath` | 真实 smoke log 路径 | `-RequireExistingLog` 会读取该文件。 |
| `command.logSha256` | 与 log 匹配的 SHA256 | validator 会重新计算并比较。 |
| `results.stdoutSummary` | 非空 stdout 摘要 | 必须人工复核真实 smoke stdout，不能只依赖 log hash。 |
| `results.stderrSummary` | 非空 stderr 摘要或 `no-stderr-emitted` | stderr 为空也必须显式说明，不能留空。 |
| `results.smokeStatus` | `passed` | `blocked-by-cuda-driver` 不是 passed。 |
| `results.nativeAssetsCopied` | `true` | 证明 packaged native assets 已进入 clean consumer 输出目录。 |
| `isDependencyProbeOnly` | `false` | dependency probe 不是 runtime execution proof。 |
| `isRuntimeExecutionEvidence` | `true` | 真实记录必须显式声明。 |
| `canPromoteRuntimeProof` | `true` | 只有真实记录且校验通过后才允许。 |

## Owner Input 回填字段

`package-consumer-runtime-proof-owner-input.json` 至少要覆盖：

- `cleanExternalConsumerRoot`
- `consumerProjectPath`
- `publicPackageSource`
- `managedPackageId / managedPackageVersion / managedNupkgPath / managedNupkgSha256`
- `runtimePackageId / runtimePackageVersion / runtimePackageKey / runtimeNupkgPath / runtimeNupkgSha256`
- `ownerName / machineName / hostOs / hostArchitecture / gpuName`
- `cudaDriverVersion / cudaDriverSupportedRuntime / cudaRuntimeVersion / cudnnVersion`
- `tensorRtVersion / tensorRtLine`
- `restoreCommand / buildCommand / smokeCommand`
- `exitCode / startedAtUtc / finishedAtUtc`
- `dependencyProbeStatus / smokeStatus / nativeAssetsCopied`
- `smokeLogPath / smokeLogSha256`
- `stdoutSummary / stderrSummary`

禁止用 `local feed`、`ProjectReference`、`direct .nupkg`、`Smoke=not-requested`、`dependency-probe-only`、`blocked-by-cuda-driver`、`build-only`、`dry-run`、`template-only` 充当真实 runtime smoke。

## 通过标准

真实 proof 的最低通过标准是：

```text
ValidationState=runtime-proof-ready
ProofClassification=package-consumer-runtime
IsRuntimeExecutionEvidence=True
CanPromoteRuntimeProof=True
FailedProofItems=0
```

并且 `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof` 退出码必须为 `0`。

通过后，再刷新发布证据：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalReleaseDryRun.ps1 `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 `
  -AllowRuntimeSmokeBlocked `
  -WarnOnly
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofOwnerHandoff.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseOwnerApprovalInputTemplate.ps1 `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseOwnerApprovalInput.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseOwnerDecisionRecord.ps1 `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleasePublishExecutionChecklist.ps1 `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleasePromotionIssueRecord.ps1 `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22
```

## 常见失败状态

| 状态 | 含义 | 下一步 |
| --- | --- | --- |
| `draft-blocked-by-cuda-driver` | 本机证据是草稿，且 CUDA driver/runtime 不兼容。 | 换兼容主机或升级 driver。 |
| `dependency-probe-only` | 只跑到了 bridge/native dependency probe。 | 必须运行 package consumer smoke。 |
| `commandsReady=false` | restore/build/smoke 命令或 exit code/log path 不完整。 | 回填完整命令和结果。 |
| `hostReady=false` | owner/machine/GPU/driver/CUDA/TensorRT 元数据不完整。 | 补齐 host 字段。 |
| `logSha256Matches=false` | 记录中的 SHA256 与真实 log 不一致。 | 重新计算 log hash，不要手写猜测值。 |
| `stdoutSummaryReady=false` 或 `stderrSummaryReady=false` | stdout/stderr 复核摘要不完整。 | 复核真实 smoke log；stderr 为空时写明 `no-stderr-emitted`。 |
| `smokeStatus=blocked-by-cuda-driver` | 当前主机无法执行目标 CUDA runtime。 | 保留 blocker，不能宣称 passed。 |

## 不能误读

- runbook 不是 runtime proof。
- owner handoff 不是 runtime proof。
- draft record 不是 runtime proof。
- template/input-template/example 不是 runtime proof。
- `ready-needs-manual-approval` 不是 public release ready。
- `blocked-by-cuda-driver` 不是 smoke passed。
- `managedNupkgSha256Ready=true` 和 `runtimeNupkgSha256Ready=true` 只说明 draft/hash 信息较完整，不代表 runtime smoke passed。
- `stdoutSummary` 或 `stderrSummary` 任意一个为空，都不能作为可发布 external runtime proof。
- 没有 `package-consumer-runtime` + `smokeStatus=passed` + `exitCode=0` + `-FailOnNotProof` 通过，就不能关闭 release runtime proof blocker。
- `Smoke=not-requested` 不是 runtime smoke，也不能作为 package-consumer-runtime proof。
- `package-consumer-runtime-proof-owner-input` 是 owner 回填合同，不替代 `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof`。
