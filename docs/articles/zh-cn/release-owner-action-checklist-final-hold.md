# Release Owner Action Checklist Final Hold

本文用于把 TensorRtSharp4.0 发布候选 final hold 状态下的 owner 待办动作整理成可执行清单。它不是 proof record，不代表公开发布已经完成；它只定义 owner 拿到真实外部条件后应如何补齐剩余 release close blockers。

Owner 最短执行面以 `owner-release-execution-package` 的 `oneScreenReleaseHoldChecklist` 为准。该一屏 Release Hold 清单只收敛 owner action，不是 proof；在 owner authorization、`package-consumer-runtime`、Linux runner proof、`real-model-runtime` 和 post-publish verification 缺失真实记录与 validator 之前，`canCloseReleaseIssue=false` 必须保持不变。

最终关闭还需要 `release-issue-close-record-validation` 从 `blocked-template-only` 晋级为真实 close-ready 记录。`release-issue-close-record-template.json` 不是 proof；只有上述真实 proof 和 release close preflight 全部通过后，owner 才能回填 release evidence bundle SHA256、rollback plan、final close decision，并运行 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`。

## 当前 Final Hold 状态

当前必须保持：

```text
FreezeState=blocked-real-proof-required
PerformsPublish=False
CanPublishPublicly=False
CanCloseReleaseIssue=False
```

仍未完成的真实 blockers：

1. owner authorization
2. package-consumer-runtime
3. Linux runner proof
4. real-model-runtime
5. post-publish verification

## Owner Authorization Proof Gate

Owner 授权不再只看“是否有 approval 文件”，还必须通过 `ownerAuthorizationProofGate`：

- `gateState=blocked-owner-authorization-required`
- `requiredFieldCount=14`
- `missingOwnerInputCount=14`
- `canMaterializeExecutableCommands=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

Owner 回填前必须逐项补齐 `ownerName`、`ownerDecisionId`、`approvalTimestampUtc`、`targetChannel`、`selectedRuntimePackageKey`、managed/runtime package ID 和 version、`approvedCommandPlanSha256`、`approvedProofBundleSha256`、`rollbackPlan`、`credentialHandlingAcknowledged`、`nvidiaRedistributionApproval`。`Test-ReleaseOwnerApprovalInput.ps1` 与 `Test-OwnerAuthorizedPublishCommandPlan.ps1` 都通过之前，任何 `dotnet nuget push`、GitHub Packages 上传或 GitHub Release asset 上传命令都只能保留为 placeholder。

命令物化还依赖 `manualMaterializationPrerequisites`：owner authorization、owner decision、freeze summary、package-consumer runtime proof、publish checklist 和 stale release claims audit。模板、draft、runbook、collection package、local feed、ProjectReference 或缺少 owner 字段的记录，不能把 `canPublishPublicly` 或 `canCloseReleaseIssue` 改成 true。

## Owner Proof Final Backfill Tracks

当前 owner final backfill 只允许沿四条固定 track 收口：

| Track | 目标 proof | Validator | 仍需 owner 输入 |
| --- | --- | --- | --- |
| `package-consumer-runtime` | 干净 package consumer runtime proof | `Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RequireExistingLog -FailOnNotProof` | 真实 nupkg/source、兼容 CUDA/TensorRT host、clean consumer、真实 smoke log 与 SHA256 |
| `linux-runner-proof` | 真实 Linux x64 runner proof | `Test-LinuxRunnerEvidenceRecord.ps1` | Linux runner、runtime package、CMake/native build、native asset copy 与 package consumer evidence |
| `real-model-runtime` | Classification / YoloVision 真实模型 proof | `Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog` | 真实模型、输入、labels、license、输出日志、日志 SHA256 |
| `post-publish verification` | 真实发布渠道 clean consumer proof | `Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json -RequireExistingLog -FailOnNotProof` | 真实发布 URL、下载后 nupkg SHA256、clean consumer restore/build/smoke log |

`ownerProofFinalBackfillTracks` 只能说明下一步要收集什么，不是 proof record。local feed、ProjectReference、bridge-only package consumer log、`Skipped=True`、mismatched log SHA256、build-only/precheck output、sidecar-only report、collection bundle、input package、runbook 和 Windows handoff for Linux proof 都不能替代上述四条 track 的 validator 通过结果。

## 真实记录字段置位要求

Owner 回填真实记录时，必须按 proof record 的真实语义置位字段；模板字段、draft 字段或 sidecar metadata 不能通过改名晋级。

### Package Consumer Runtime 真实记录

回填 `artifacts/final-release/external-runtime-proof-record.json` 前必须满足：

- `templateOnly=false`。
- `recordKind=external-runtime-proof-record`。
- `runtimePackageKey` 等于本次 release target key，且 `packageSource.runtimePackageKey` 必须与之完全一致。
- `proofClassification=package-consumer-runtime`。
- `isRuntimeExecutionEvidence=true`。
- `canPromoteRuntimeProof=true`。
- `currentRuntimeProofStatus=smoke-passed`。
- `packageSource.noProjectReference=true`，clean consumer 不得引用同仓库项目。
- `packageSource.managedNupkgSha256` 与 `packageSource.runtimeNupkgSha256` 必须是下载或安装所用真实 nupkg 的 64 字符 SHA256。
- `command.exitCode=0`，`command.logSha256` 必须对应真实 restore/build/run 日志。
- `results.stdoutSummary` 与 `results.stderrSummary` 必须由 owner 复核，且能对应 `command.logPath` 的真实输出。
- 必须通过 `Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RequireExistingLog -FailOnNotProof`。

### Post Publish Verification 真实记录

真实发布完成后，回填 `artifacts/final-release/post-publish-verification-record.json` 前必须满足：

- `templateOnly=false`。
- `recordKind=post-publish-verification-record`。
- `postPublishProofClassification=post-publish-package-consumer-runtime`。
- `isPostPublishVerificationProof=true`。
- `performsPublish=false`，该记录只证明发布后的 clean consumer 验证，不执行发布。
- `packageIdentity.managedPackageUrl` 与 `packageIdentity.runtimePackageUrl` 必须指向真实发布渠道。
- `packageIdentity.managedNupkgSha256` 与 `packageIdentity.runtimeNupkgSha256` 必须匹配发布后下载的真实包。
- `noProjectReference=true`，post-publish clean consumer 不得引用源码项目。
- `nativeAssetListingSha256`、`dependencyProbeLogSha256`、`smokeLogSha256` 必须分别对应真实 native asset listing、dependency probe log 与 smoke log。
- `dependencyProbePassed=true` 且 `runtimeSmokePassed=true`。
- `canCloseReleaseIssue` 必须保持 false，直到 `Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json -RequireExistingLog -FailOnNotProof` 通过后才允许 owner 手动置位。

### PostPublish Required Evidence 对齐

真实 post-publish 记录还必须覆盖 `postPublishRequiredEvidence`，不得只写 smoke passed：

- `selectedChannel`、`channelSourceUri`、`publishedPackageUrl`
- `managedPackageUrl`、`runtimePackageUrl`
- `managedNupkgSha256`、`runtimeNupkgSha256`
- `cleanConsumerRootOutsideRepository`、`consumerProjectPath`、`noProjectReference`
- `restoreLogPath`、`nativeAssetListingSha256`
- `dependencyProbeLogPath`、`dependencyProbeLogSha256`
- `runtimeSmokeLogPath`、`runtimeSmokeLogSha256`
- `runtimeSmokePassed`、`runtimeSmokeExitCode`
- `stdoutSummary`、`stderrSummary`、`hostMetadata`

## Owner 一次性执行顺序

建议 owner 按以下顺序执行，不要跳过授权或把准备材料写成 proof：

1. 确认 owner authorization。
2. 在 clean consumer 中完成 package-consumer-runtime。
3. 使用真实模型资产完成 real-model-runtime。
4. 在真实 Linux x64 runner 上完成 Linux runner proof。
5. 真实发布后完成 post-publish verification。

如果任何一步缺少真实日志、真实输入或 validator 结果，应保持 blocked 状态。

## 1. Owner Authorization

### 输入

- package ID：`JYPPX.TensorRT.CSharp.API`
- 目标版本。
- 目标 feed，例如 nuget.org 或 owner 明确指定的私有 feed。
- 授权人、授权时间、发布范围。
- 是否允许公开发布、是否允许关闭 release issue。

### 可接受证据

- owner 明确授权记录。
- 授权记录中包含 package ID、版本、feed、时间和授权范围。
- 授权结论与 `canPublishPublicly`、`canCloseReleaseIssue` 状态一致。
- 授权输入和手动发布命令计划必须通过 `Test-ReleaseOwnerApprovalInput.ps1` 与 `Test-OwnerAuthorizedPublishCommandPlan.ps1` 验证；这两个 validator 只确认 owner 授权材料和手动命令计划，不会执行真实发布。

### 不可接受替代

- “准备发布”“计划发布”“理论可发布”。
- 没有目标版本或 feed 的泛化授权。
- 由非 owner 推断出的发布许可。

## 2. Package Consumer Runtime

### 输入

- 真实 nupkg 或真实发布 feed。
- clean consumer 项目。
- 兼容 CUDA / TensorRT / driver host。
- 不依赖同仓库 ProjectReference。

### 执行

推荐执行并回填：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof
```

### 输出

- `external-runtime-proof-record.json`
- restore/build/run log
- package source、package version、runtime host、CUDA、TensorRT、driver 信息
- validator 通过日志

### 不可接受替代

- ProjectReference consumer。
- local feed。
- draft package。
- helper project。
- build-only。
- sidecar-only。
- blocked-by-cuda-driver。

## 3. Real Model Runtime

### 输入

- 真实模型文件。
- 真实 input 文件。
- labels 或类别映射。
- 模型 license / redistribution 说明。
- 模型 SHA256。
- 目标 sample，例如 `samples\ComputerVision\01.Classification` 或 `applications\YoloVision`。

### 执行

先验证资产 manifest，再回填真实运行日志：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleAssetManifest.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog
```

### 输出

- sample asset manifest。
- sample run evidence record。
- 模型来源、license、SHA256、input、labels、命令行、stdout/stderr。
- validator 通过日志。

### 不可接受替代

- parse-only。
- dry-run。
- build-only。
- 无模型 hash。
- 无 license。
- 只有 sidecar，没有真实 sample run log。

## 4. Linux Runner Proof

### 输入

- 真实 Linux x64 runner。
- runner OS 信息。
- NVIDIA driver、CUDA、TensorRT 版本。
- commit SHA。
- package/source 使用方式。
- 实际执行命令。

### 执行

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidenceRecord.ps1
```

### 输出

- Linux runner evidence record。
- OS / driver / CUDA / TensorRT / commit / commands / logs。
- validator 通过日志。

### 不可接受替代

- Windows 本机日志。
- 手写计划。
- 没有 runner 环境字段的说明。
- blocked-by-application-control。

## 5. Post Publish Verification

### 输入

- 真实发布后的 package feed。
- clean consumer。
- 目标 package version。
- restore/build/run 命令。
- 真实发布后可复验日志。

### 执行

只有真实发布完成后才能执行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof
```

### 输出

- post-publish verification record。
- feed URL 或 source。
- package version。
- clean consumer restore/build/run log。
- validator 通过日志。

### 不可接受替代

- pre-publish local feed。
- draft package。
- 本地 build。
- ProjectReference。
- runbook。
- 手写“验证通过”。

## 无真实条件时允许维护的工作

如果 owner 尚未提供真实外部条件，维护者可以继续做：

- README、README.zh-CN、docs index、toc、technical article roadmap 一致性维护。
- `eng/Test-StaleReleaseClaims.ps1` 审计。
- quality tests 增强。
- sample README、asset manifest 模板、license checklist、proof schema 维护。
- release hold 说明和 owner checklist 维护。
- `applications/TensorRtExec` 的 ONNX build/precheck report 维护，但它仍然只能作为 build/precheck 证据，不能替代真实 package-consumer-runtime 或 real-model-runtime proof。

但不能做：

- 执行 `dotnet nuget push`。
- 上传 GitHub Packages。
- 创建 GitHub Release。
- delist、delete、withdraw package。
- 不要把 local feed、ProjectReference、build-only、parse-only、dry-run、sidecar-only、collection bundle、input package、runbook 写成 proof。

## 最小验证组合

每次修改 release-facing 文档后，至少执行：

```powershell
Set-Location .
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StaleReleaseClaims.ps1
dotnet build .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-restore /m:1 /p:UseSharedCompilation=false /nr:false
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~TechnicalArticleRoadmap|FullyQualifiedName~ReleaseCandidateFinalEvidenceFreeze|FullyQualifiedName~CompatibleHostProofExecutionPack|FullyQualifiedName~ReleaseCloseGapDashboard"
```

如果 owner 提供真实 proof，则追加对应 validator，并只依据 validator 结果更新 blocker 状态。
