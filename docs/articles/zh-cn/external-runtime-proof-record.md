# External Runtime Proof Record

`external-runtime-proof-record` 是给 CUDA 兼容主机回填真实 runtime smoke 的结构化输入层。当前本机仍是 `blocked-by-cuda-driver` 时，它只生成模板，不能被当作 runtime execution proof。

## 生成模板

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofRecordTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofRecordInputTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofRecordDraft.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofRecordExample.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofOwnerHandoff.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CompatibleHostRuntimeProofRunbook.ps1
```

输出：

- `artifacts/final-release/external-runtime-proof-record-template.json`
- `artifacts/final-release/external-runtime-proof-record-template.md`
- `artifacts/final-release/external-runtime-proof-record.input-template.json`
- `artifacts/final-release/external-runtime-proof-record.input-template.md`
- `artifacts/final-release/external-runtime-proof-record.draft.json`
- `artifacts/final-release/external-runtime-proof-record.draft.md`
- `artifacts/final-release/external-runtime-proof-record.example.json`
- `artifacts/final-release/external-runtime-proof-record.example.md`
- `artifacts/final-release/external-runtime-proof-owner-handoff.json`
- `artifacts/final-release/external-runtime-proof-owner-handoff.md`
- `artifacts/final-release/compatible-host-runtime-proof-runbook.json`
- `artifacts/final-release/compatible-host-runtime-proof-runbook.md`
- `artifacts/final-release/external-runtime-proof-backfill-plan.json`
- `artifacts/final-release/external-runtime-proof-backfill-plan.md`

默认状态：

- `recordKind=external-runtime-proof-record-template`
- `templateOnly=true`
- `proofState=template-only`
- `proofClassification=template-only`
- `isRuntimeExecutionEvidence=false`
- `isDependencyProbeOnly=true`
- `canPromoteRuntimeProof=false`

input-template 和 example 都不能当作 proof。example 固定为 `example-not-for-publication`，用于演示 validator 如何拒绝非真实证据。

`external-runtime-proof-owner-handoff` 是 release owner 交接材料。它会把当前 `runtimeProofStatus`、`runtimeProofRequiredForRelease`、external proof key/hash 状态、建议 smoke 命令、日志路径约定和 `-RequireExistingLog` 校验命令写到同一份 JSON/Markdown 中。它的状态通常是 `owner-action-required`，用于指导兼容 CUDA host 回填，不是 proof。

`compatible-host-runtime-proof-runbook` 是更细的外部执行手册。它把 handoff 的命令扩展为逐步执行、hash、record 字段回填和 `-FailOnNotProof` 校验清单。它必须保持 `performsPublish=false`、`approvesPublicRelease=false`，没有真实 compatible-host smoke passed 前也必须保持 `canPromoteRuntimeProof=false`。

`external-runtime-proof-backfill-plan` 是把 input-template、handoff、runbook、compatible-host smoke、hash 捕获、真实 record 回填、`-RequireExistingLog -FailOnNotProof` 校验和 release gate 刷新串起来的 owner-facing 阶段计划。它的默认状态是 `blocked-compatible-host-proof-required`，只能指导回填，不能替代真实 proof。

`external-runtime-proof-record.draft` 是从本地 package-consumer 与 runtime-readiness 产物自动生成的草稿。它会尽量收集 managed/runtime nupkg SHA256、smoke log SHA256、native asset copy 状态、smoke exit code 和 no ProjectReference 状态；但当当前机器仍是 `blocked-by-cuda-driver` 或字段不完整时，它必须保持 `recordKind=external-runtime-proof-record-draft`、`templateOnly=true` 或 `canPromoteRuntimeProof=false`。草稿只能减少兼容 CUDA 主机上的手工步骤，不能替代 `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof`。

## 证据分类

外部 runtime proof 现在必须显式填写 `proofClassification`。允许值如下：

| 分类 | 含义 | 能否提升为 external runtime proof |
| --- | --- | --- |
| `build-only` | 只证明 restore/build/engine build 或转换报告。 | 否 |
| `dependency-probe-only` | 只证明 bridge/native dependency probe 或加载诊断。 | 否 |
| `synthetic-input-runtime` | 使用 synthetic input 跑通 runtime 管线。 | 否 |
| `real-model-runtime` | 使用真实模型、真实输入和真实日志，但不一定来自干净 package consumer。 | 否 |
| `package-consumer-runtime` | 干净消费端通过 NuGet/runtime package 运行 smoke。 | 是，仍需满足全部校验项 |

bridge-only package consumer log、bridge-only wrapper surface、`Skipped=True`、`WrapperSurfaceEvidenceKind=compile-surface-proof`、`IsRuntimeExecutionProof=False`、Parser/ParserRefitter diagnostic snapshots 和 copied managed diagnostic snapshot 都只能作为诊断或 compile-surface evidence。它们必须在 `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof` 下保持 non-proof，不能被写成 `package-consumer-runtime`。

`real-model-runtime` 必须补模型名、模型 SHA256、许可证、输入资产名和输入资产 SHA256。它可以作为案例文章和模型教程证据，但不能替代 `package-consumer-runtime`。只有 `proofClassification=package-consumer-runtime`、`runtimePackageKey` 与 release target 匹配、`packageSource.runtimePackageKey` 与 release target 匹配、`packageSource.consumerProjectName` / `packageSource.consumerProjectPath` 指向干净消费端、host CUDA/TensorRT/cuDNN/driver/GPU/OS 元数据完整、`packageSource.managedNupkgSha256` 和 `packageSource.runtimeNupkgSha256` 均为 64 位 SHA256、`command.smokeCommand` 包含 `--runtime-package-key`、`smokeStatus=passed`、`exitCode=0`、`nativeAssetsCopied=true`、`noProjectReference=true`、`isDependencyProbeOnly=false`、`command.logSha256` 已回填且 `stdoutSummary` 与 `stderrSummary` 均已人工复核时，validator 才可能把记录提升为真实 runtime proof。若真实 smoke 没有 stderr 输出，`stderrSummary` 也必须明确写入 `no-stderr-emitted` 或等价复核说明，不能留空。

使用 `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog` 时，validator 会读取 `command.logPath` 指向的真实日志并重新计算 SHA256。只有 `logSha256Matches=true` 时，日志证据才算匹配。

## 真实回填需要什么

兼容主机回填真实 proof 时必须记录：

- owner name
- machine name
- OS description
- GPU name
- NVIDIA driver version
- CUDA driver supported runtime
- CUDA runtime version
- TensorRT runtime version
- cuDNN runtime version
- TensorRT line
- clean package consumer project name
- clean package consumer `.csproj` path
- smoke command with `--runtime-package-key <release target>`
- managed package source
- runtime package source
- no ProjectReference confirmation
- restore/build/smoke commands
- exit code
- log path
- log SHA256
- stdout 摘要
- stderr 摘要；如果没有 stderr，也必须写明 no-stderr-emitted
- proof classification
- model/hash/license/input asset 证据（真实模型 runtime 时必填）

只有真实 package consumer smoke 在兼容 CUDA driver/GPU host 上 `exitCode=0` 且 smoke output 可追溯，才能把 `isRuntimeExecutionEvidence` 改为 `true`。

## 校验模板或真实记录

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.example.json -OutputRoot artifacts/final-release/example-validation
```

默认校验模板时必须保持：

- `validationState=template-only`
- `runtimePackageKeyMatches=true`
- `packageSourceRuntimePackageKeyMatches=true`
- `managedNupkgSha256Ready=false`
- `runtimeNupkgSha256Ready=false`
- `logSha256FormatReady=false`
- `logSha256Matches=false`
- `isRuntimeExecutionEvidence=false`
- `canPromoteRuntimeProof=false`

真实 record 必须满足：

- `recordKind=external-runtime-proof-record`
- `runtimePackageKey` 与 release target 匹配
- `packageSource.runtimePackageKey` 与 release target 匹配
- `packageSource.managedNupkgSha256` 和 `packageSource.runtimeNupkgSha256` 为真实消费包的 64 位 SHA256
- `templateOnly=false`
- owner/machine/OS/GPU/driver/CUDA/TensorRT 信息完整
- no ProjectReference
- restore/build/smoke command 完整
- smoke `exitCode=0`
- `smokeStatus=passed`
- log path 非空
- log SHA256 为 64 位 SHA256；使用 `-RequireExistingLog` 时必须与日志文件匹配
- stdoutSummary 非空
- stderrSummary 非空；如果 smoke 没有 stderr，也必须写明 no-stderr-emitted
- native assets copied
- `isDependencyProbeOnly=false`
- `proofClassification=package-consumer-runtime`
- 真实记录可以加 `-RequireExistingLog` 要求 log path 文件存在

## Owner Handoff 回填流程

生成 handoff：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofOwnerHandoff.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22
```

在兼容 CUDA host 上按 handoff 中的命令执行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -SmokeRuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -RunSmoke -KeepConsumerOutput
Get-FileHash -LiteralPath "artifacts/final-release/external-runtime-proof/win-x64-trt11.0-cuda13.2-cudnn9.22/package-consumer-smoke.log" -Algorithm SHA256 | Select-Object -ExpandProperty Hash
Get-FileHash -LiteralPath "<downloaded-managed-nupkg>" -Algorithm SHA256 | Select-Object -ExpandProperty Hash
Get-FileHash -LiteralPath "<downloaded-runtime-nupkg>" -Algorithm SHA256 | Select-Object -ExpandProperty Hash
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -RequireExistingLog -FailOnNotProof
```

回填时必须把 input-template 复制为 `external-runtime-proof-record.json`，并修改：

- `recordKind=external-runtime-proof-record`
- `templateOnly=false`
- `proofClassification=package-consumer-runtime`
- `isDependencyProbeOnly=false`
- `isRuntimeExecutionEvidence=true`
- `canPromoteRuntimeProof=true`
- `packageSource.runtimePackageKey=win-x64-trt11.0-cuda13.2-cudnn9.22`
- `packageSource.consumerProjectName=PackageConsumerSmoke`
- `packageSource.consumerProjectPath=<clean consumer .csproj path>`
- `packageSource.managedNupkgSha256` / `packageSource.runtimeNupkgSha256` 为真实消费包 SHA256
- `packageSource.noProjectReference=true`
- `host.cudaDriverSupportedRuntime=<实际 driver supported runtime>`
- `host.cudnnVersion=<实际 cuDNN runtime>`
- `host.tensorRtLine=<实际 TensorRT line>`
- `command.smokeCommand` 包含 `--runtime-package-key <release target>`
- `command.exitCode=0`
- `command.logPath` 指向真实 smoke log
- `command.logSha256` 为 `Get-FileHash -Algorithm SHA256` 的结果
- `results.nativeAssetsCopied=true`
- `results.smokeStatus=passed`
- `results.stdoutSummary=<真实 stdout 摘要>`
- `results.stderrSummary=<真实 stderr 摘要或 no-stderr-emitted>`

只有 `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof` 通过后，external proof 才能晋级。`external-runtime-proof-owner-handoff.md` 自身不能晋级，也不能作为 release approval。

## 不能误读

- 模板不是 runtime proof。
- backfill plan 不是 runtime proof。
- input-template 和 example 不是 runtime proof。
- `DependencyProbe BridgeInitialized` 不是 runtime execution proof。
- `build-only` 不是 runtime proof。
- `synthetic-input-runtime` 不是真实模型 proof。
- `real-model-runtime` 不是干净 package consumer proof。
- `runtimePackageKeyMatches=false` 不是目标 runtime package proof。
- `packageSourceRuntimePackageKeyMatches=false` 不是目标 runtime package proof。
- `managedNupkgSha256Ready=false` 或 `runtimeNupkgSha256Ready=false` 不是可发布 runtime proof。
- `logSha256FormatReady=false` 或 `logSha256Matches=false` 不是可发布 runtime proof。
- `stdoutSummary` 或 `stderrSummary` 任意一个为空，都不是可发布 runtime proof。
- `blocked-by-cuda-driver` 不是 smoke passed。
- dry run 允许继续汇总证据，不代表 runtime proof complete。

## Owner-facing 输出

以下脚本会把 external proof 的 key/hash 状态同步给 release owner：

- `Export-ReleaseOwnerApprovalInputTemplate.ps1`
- `Export-ReleaseOwnerDecisionTemplate.ps1`
- `Export-ReleaseOwnerDecisionRecord.ps1`
- `Export-ReleasePublishExecutionChecklist.ps1`
- `Export-ReleasePromotionIssueRecord.ps1`
- `Test-FinalReleaseDryRun.ps1`

默认模板状态下应看到：

- `externalRuntimeProofState=template-only`
- `externalRuntimeProofClassification=template-only`
- `externalRuntimeProofRuntimePackageKeyMatches=true`
- `externalRuntimeProofPackageSourceRuntimePackageKeyMatches=true`
- `externalRuntimeProofManagedNupkgSha256Ready=false`
- `externalRuntimeProofRuntimeNupkgSha256Ready=false`
- `externalRuntimeProofLogSha256FormatReady=false`
- `externalRuntimeProofLogSha256Matches=false`
- `externalRuntimeProofOwnerActionStatus=owner-action-required`

这表示 release target key 与 package source key 已经对齐，但还没有真实 consumed nupkg hash 和 smoke log hash。它仍然不能晋级 `package-consumer-runtime`。
