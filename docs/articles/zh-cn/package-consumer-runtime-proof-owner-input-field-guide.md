# Package Consumer Runtime Proof Owner Input 字段指南

`package-consumer-runtime-proof-owner-input.template.json` 是发布前 owner 在真实 clean external consumer 环境中回填 runtime proof candidate 的输入合同。它不是 proof，不发布包，不关闭 issue，也不会把本地 feed、ProjectReference、direct `.nupkg`、queued GitHub Actions run 或 missing self-hosted runner 当成 public package proof。

本文解释字段含义、采集方法和禁止替代项，方便后续真实 owner 在兼容 CUDA/TensorRT 主机上一次性回填完整数据。

## 生成模板

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1
```

模板输出：

```text
artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json
artifacts/final-release/package-consumer-runtime-proof-owner-input.template.md
```

验证模板或真实输入：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict
```

生成字段 schema 和禁止替代项扫描：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofOwnerInputSchema.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofForbiddenSubstituteScan.ps1
```

新增审计产物：

```text
artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.json
artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.md
artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.json
artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.md
```

`Import-PackageConsumerRuntimeProofOwnerInput.ps1` 会串联 owner input validator、schema 导出、forbidden substitute scan、record projection 和 record validator。schema/scan 只提供审计证据，不发布、不关闭 issue、不晋级 `package-consumer-runtime` proof。

## 本阶段执行清单

本阶段已经证明 bridge/full package consumer 的 restore、build、native asset copy 和 readiness 可以通过；但 `Smoke=not-requested` 仍不是 runtime proof。Owner 要把它提升为 `package-consumer-runtime` proof candidate，必须在兼容 CUDA/TensorRT 主机上执行真实 smoke，并保留以下输入。

| 顺序 | 输入 | 命令或字段 | 必须保留的证据 |
|---:|---|---|---|
| 1 | clean external consumer | 仓库外 `cleanExternalConsumerRoot` 与 `consumerProjectPath` | consumer `.csproj` hash，不能是源码目录 sample，不能有 `ProjectReference` |
| 2 | 公开 package source | `publicPackageSourceKind`、`publicPackageSource`、`publicPackageFeedUrl`、`managedPackageUrl`、`runtimePackageUrl`、`managedPackageId`、`runtimePackageId`、`runtimePackageKey` | restore log，公开 feed URL，包详情/下载 URL，包版本，managed/runtime `.nupkg` SHA256 |
| 3 | package consumer smoke | `pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -SmokeRuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -RunSmoke -KeepConsumerOutput` | restore/build/smoke log，native asset listing，`smokeStatus=passed` |
| 4 | stdout/stderr 复核 | `results.stdoutSummary`、`results.stderrSummary`、`results.failureDiagnostic` | owner 审阅后的 stdout 摘要；stderr 为空也要写 `no-stderr-emitted` |
| 5 | host metadata | `hostOs`、`hostArchitecture`、`gpuName`、`cudaDriverVersion`、`cudaRuntimeVersion`、`tensorRtVersion`、`cudnnVersion` | 与同一次 smoke 对应的 host/runtime 版本 |
| 6 | runner infrastructure | `sourceRunnerQueueStatus`、`sourceRunnerInfrastructureStatus`、`sourceRunnerOwnerAction` | 必须说明 queued/missing runner 是 owner-infra-action；只有 completed/not-queued 且 runner available/ready/not-required 才能作为上下文 |
| 7 | strict validator | `pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts\final-release\external-runtime-proof-record.json -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -RequireExistingLog -FailOnNotProof` | validator 退出码为 0、`validationState=real-runtime-proof`、failed proof item 为 0 |

如果当前机器只能得到 `blocked-by-cuda-driver`、`dependency-probe-only` 或 `Smoke=not-requested`，应记录为 owner-action-required，不能写成 `smokeStatus=passed`。

## 字段分组

### Owner 与主机

| 字段 | 示例 | 采集方式 | 要求 |
|---|---|---|---|
| `ownerName` | `Linda Johnson` | Owner 手工确认 | 必须是真实 owner，不允许占位符 |
| `machineName` | `TRT-BUILD-4090-01` | `$env:COMPUTERNAME` 或 host name | 必须能定位执行主机 |
| `hostOs` | `Windows 11 Pro 24H2` | `dotnet --info` / systeminfo | 必须真实 |
| `hostArchitecture` | `x64` | `dotnet --info` | 必须真实 |
| `gpuName` | `NVIDIA GeForce RTX 4090` | `nvidia-smi --query-gpu=name` | 必须真实 |

### CUDA / TensorRT / cuDNN

| 字段 | 示例 | 采集方式 | 要求 |
|---|---|---|---|
| `cudaDriverVersion` | `555.85` | `nvidia-smi` | 必须真实 |
| `cudaDriverSupportedRuntime` | `12.5` | `nvidia-smi` CUDA Version | 必须真实 |
| `cudaRuntimeVersion` | `13.2` | clean consumer dependency probe | 必须与 runtime package lane 对齐 |
| `cudnnVersion` | `9.22` | dependency probe 或 DLL version | 必须真实 |
| `tensorRtVersion` | `11.0.x` | TensorRT runtime probe | 必须真实 |
| `tensorRtLine` | `TRT11` | package/runtime lane | 必须真实 |

### 包与 clean consumer

| 字段 | 要求 |
|---|---|
| `cleanExternalConsumerRoot` | 必须是仓库外路径 |
| `consumerProjectPath` | 必须指向外部 clean consumer `.csproj` |
| `publicPackageSourceKind` | 必须是 `nuget` 或 `github-packages`；不能写 local-feed、direct-nupkg、dry-run 或 queued-run |
| `publicPackageSource` | 必须是公开 package source；不能是本地目录、本地 feed、artifacts、direct `.nupkg` |
| `publicPackageFeedUrl` | 必须是公开 NuGet/GitHub Packages feed URL |
| `managedPackageUrl` / `runtimePackageUrl` | 必须指向公开 managed/runtime package 详情页或下载页 |
| `managedPackageId` / `runtimePackageId` | 必须与发布包 ID 对齐 |
| `managedPackageVersion` / `runtimePackageVersion` | 必须与 owner 实际 restore 的版本一致 |
| `managedNupkgSha256` / `runtimeNupkgSha256` | 必须是 64 位 SHA256 |

### 命令与执行结果

| 字段 | 要求 |
|---|---|
| `restoreCommand` | 真实执行过的 restore 命令 |
| `buildCommand` | 真实执行过的 build 命令 |
| `smokeCommand` | 必须包含 `--runtime-package-key` 和目标 runtime key |
| `exitCode` | 必须可解析为 int 且等于 `0` |
| `startedAtUtc` / `finishedAtUtc` | 必须可解析为 DateTimeOffset |
| `dependencyProbeStatus` | 必须是 `passed` 或 `compatible-host-passed` |
| `smokeStatus` | 必须是 `passed` |
| `nativeAssetsCopied` | 必须可解析为 bool 且为 `true` |
| `smokeLogPath` / `smokeLogSha256` | log 必须存在并可 hash 对齐 |
| `stdoutSummary` / `stderrSummary` | 必须由 owner 审阅；无 stderr 时写明确说明 |
| `failureDiagnostic` | 成功时可写 `none`；失败时必须说明阻塞原因 |

### Source Runner 边界

| 字段 | 要求 |
|---|---|
| `sourceRunnerQueueStatus` | 必须是 `completed` 或 `not-queued`；`queued GitHub Actions run` 只能记录为 owner-infra-action |
| `sourceRunnerInfrastructureStatus` | 必须是 `available`、`ready` 或 `not-required`；`missing self-hosted runner` 不能晋级 proof |
| `sourceRunnerOwnerAction` | 必须明确包含 `owner-infra-action`，用于说明 queued/missing runner 只需要 owner 修复基础设施 |

## 禁止替代项

以下内容不能作为 package consumer runtime proof：

- template-only JSON。
- draft JSON。
- runbook。
- dashboard。
- local `.nupkg`。
- local feed。
- ProjectReference。
- direct `.nupkg`。
- repository path leakage。
- build-only。
- parse-only。
- dry-run。
- queued GitHub Actions run。
- missing self-hosted runner。
- dependency-probe-only。
- GUI 截图。
- TensorRtExec build report。
- sample README。
- sidecar-only。

这些材料可以辅助排查，但不能关闭 release issue。

## 晋级路径

1. Owner 在仓库外创建 clean external consumer。
2. 从公开 package source restore managed 包和 runtime 包。
3. 运行 dependency probe 和 smoke。
4. 记录 log、hash、stdout/stderr summary、host metadata。
5. 回填 owner input JSON。
6. 运行 owner input validator。
7. 生成 package consumer runtime proof record。
8. 运行 strict proof validators。

只有 validators 明确给出可晋级结果时，才允许进入 release close proof。
