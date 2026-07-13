# Post-Publish Verification Record

`post-publish-verification-record` 是真实发布之后的 clean consumer 验证输入层。它不会执行发布，也不会替代 release owner approval。

## 生成模板

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationRecordTemplate.ps1
```

输出：

- `artifacts/final-release/post-publish-verification-record-template.json`
- `artifacts/final-release/post-publish-verification-record-template.md`
- `artifacts/final-release/post-publish-verification-backfill-plan.json`
- `artifacts/final-release/post-publish-verification-backfill-plan.md`

默认状态：

- `recordKind=post-publish-verification-record-template`
- `templateOnly=true`
- `verificationState=template-only`
- `performsPublish=false`
- `isPostPublishVerificationProof=false`
- `canCloseReleaseIssue=false`

`post-publish-verification-backfill-plan` 会把 owner authorization 确认、真实 channel package identity、clean consumer、restore/build/probe/smoke log、stdout/stderr 复核、`-RequireExistingLog -FailOnNotProof` 校验和 release close readiness 刷新串起来。它的默认状态是 `blocked-real-post-publish-proof-required`，只作为回填计划，不能关闭 release issue。

## 真实回填需要什么

真实发布后必须用全新的 consumer 目录验证：

- clean directory
- consumerProjectName / consumerProjectPath，且 `consumerProjectPath` 指向 clean consumer `.csproj`
- no ProjectReference
- managed package source 来自目标 channel
- runtime package source 来自目标 channel
- host metadata，包括 CUDA/TensorRT/cuDNN metadata、OS、GPU、driver、CUDA driver/runtime、TensorRT runtime/line、cuDNN version
- restore/build/smoke command，其中 `smokeCommand` 必须包含 `--runtime-package-key`
- native assets copied
- `DependencyProbe BridgeInitialized`
- compatible host smoke
- stdoutSummary / stderrSummary；stderr 为空时也必须写明 `no-stderr-emitted` 或等价复核说明
- owner/reviewer
- published version

该记录还要与 owner command plan 中的 `postPublishRequiredEvidence` 对齐：`selectedChannel`、`channelSourceUri`、`publishedPackageUrl`、`managedPackageUrl`、`runtimePackageUrl`、`managedNupkgSha256`、`runtimeNupkgSha256`、`cleanConsumerRootOutsideRepository`、`consumerProjectPath`、`noProjectReference`、`restoreLogPath`、`nativeAssetListingSha256`、`dependencyProbeLogPath`、`dependencyProbeLogSha256`、`runtimeSmokeLogPath`、`runtimeSmokeLogSha256`、`runtimeSmokePassed`、`runtimeSmokeExitCode`、`stdoutSummary`、`stderrSummary` 和 `hostMetadata` 都必须可追溯。缺任一类证据时，`canCloseReleaseIssue` 必须保持 false。

发布后建议先运行 `Test-PostPublishCleanConsumerProject.ps1` 生成 `post-publish-clean-consumer-project-scan.json`，再运行 `Export-PostPublishVerificationRecordInputDraft.ps1` 将 clean consumer 项目路径、无 `ProjectReference` 状态和日志 SHA256 带入 input draft。scan 和 input draft 都不是 proof；它们只降低真实 `post-publish-verification-record.json` 回填时的漏填风险。

## 推荐执行顺序

模板中的 `executionSteps` 给 release owner 一个固定顺序，避免把 dry-run、本地包或 dependency probe 误当成发布后 proof：

1. 确认发布 channel 已经包含 managed/runtime package。
2. 从该 channel 下载 managed/runtime nupkg，并记录 URL 与 SHA256。
3. 在源码仓库之外创建 clean consumer。
4. 记录 `consumerProjectName` 和指向 `.csproj` 的 `consumerProjectPath`。
5. 记录 compatible host 的 CUDA/TensorRT/cuDNN/driver/GPU/OS metadata。
6. 从 selected channel / package source restore，并保存 restore log SHA256。
7. 构建或列出输出目录，保存 native asset listing SHA256。
8. 运行 DependencyProbe 并保存 log SHA256。
9. 在兼容 CUDA/TensorRT host 上运行 runtime smoke，`smokeCommand` 必须包含 `--runtime-package-key <runtime package key>`，并且必须 `runtimeSmokeExitCode=0` 且 `smokeStatus=passed`。
10. 记录 restore/build/smoke 的 `stdoutSummary` 与 `stderrSummary`；stderr 为空时写 `no-stderr-emitted` 或等价复核说明。
11. 运行 `Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof`，重新计算 restore/native asset listing/dependency probe/smoke 日志 SHA256；该验证通过后只能进入 release close preflight 与 release issue close record，不得单独写成 `canCloseReleaseIssue=true`。
12. 刷新 `Export-ReleaseClosePreflight.ps1` 和 `Export-ReleaseEvidenceBundle.ps1`，确认 owner authorization、package-consumer runtime、Linux runner proof、real-model runtime 和 post-publish verification 全部关闭。
13. 回填 `release-issue-close-record.json`，记录 release evidence bundle SHA256、rollback plan 和 owner final close decision。
14. 运行 `Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady`；只有该最终关闭记录通过，owner 才能手工关闭 release issue。

模板现在提供可直接回填的证据字段：

- `selectedChannel`
- `channelSourceUri`
- `cleanConsumerRoot`
- `consumerProjectName` / `consumerProjectPath`
- `host.ownerName` / `host.machineName` / `host.osDescription` / `host.gpuName`
- `host.driverVersion` / `host.cudaDriverSupportedRuntime` / `host.cudaRuntimeVersion`
- `host.tensorRtRuntimeVersion` / `host.tensorRtLine` / `host.cudnnVersion`
- `restoreCommand` / `buildCommand` / `smokeCommand`
- `stdoutSummary` / `stderrSummary`（stderr 为空时写 `no-stderr-emitted`）
- `restoreLogPath` / `restoreLogSha256`
- `nativeAssetListingPath` / `nativeAssetListingSha256`
- `dependencyProbeLogPath` / `dependencyProbeLogSha256`
- `smokeLogPath` / `smokeLogSha256`
- `managedPackageSource`
- `runtimePackageSource`
- `packageIdentity.managedNupkgSha256` / `packageIdentity.runtimeNupkgSha256`
- `packageIdentity.managedPackageSha256Source` / `packageIdentity.runtimePackageSha256Source`
- `packageIdentity.managedPackageDownloadTimestampUtc` / `packageIdentity.runtimePackageDownloadTimestampUtc`
- `expectedRuntimePackageKey`
- `noProjectReference`
- `nativeAssetsCopied`
- `dependencyProbePassed`
- `runtimeSmokePassed`
- `runtimeSmokeExitCode`
- `smokeStatus`

`restoreLogSha256`、`nativeAssetListingSha256`、`dependencyProbeLogSha256` 和 `smokeLogSha256` 必须是 64 位 SHA256；真实记录必须用 `-RequireExistingLog` 重新读取引用文件并确认 SHA256 匹配。发布后 clean consumer 证据应来自目标 channel，不应来自当前仓库 `bin` 输出、`ProjectReference` 或未发布的 local feed。GitHub Release assets 只有在下载到明确的本地 package source 后，才能作为 restore source 的一部分记录。

`owner-authorized-publish-command-plan.json` 中的 `Owner Authorization Proof Gate` 通过前，post-publish 记录只能作为发布后的回填目标说明，不能反向授权发布。真实发布后也必须先通过 `release-close-preflight.json` 聚合检查 owner authorization、package-consumer runtime、Linux runner proof、real-model runtime 和 post-publish verification 全部关闭，再由 owner 手工判断是否关闭 release issue。

## 校验模板或真实记录

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof

# 验证真实发布后记录时必须要求引用日志存在并重新匹配 SHA256
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json -RequireExistingLog -FailOnNotProof

# 关闭 release issue 前聚合检查真实 proof 缺口
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseClosePreflight.ps1
```

默认校验模板时必须保持：

- `validationState=template-only`
- `isPostPublishVerificationProof=false`
- `canCloseReleaseIssue=false`

真实 record 必须满足：

- `recordKind=post-publish-verification-record`
- `templateOnly=false`
- selectedChannel 非空
- channelSourceUri 非空
- cleanConsumerRoot 指向全新消费端目录，不能是源码仓库根目录
- consumerProjectName 非空
- consumerProjectPath 非空并以 `.csproj` 结尾
- host CUDA/TensorRT/cuDNN/driver/GPU/OS metadata 齐全
- restoreCommand / buildCommand / smokeCommand 非空
- smokeCommand 包含 `--runtime-package-key` 和目标 runtime package key
- stdoutSummary / stderrSummary 非空；stderr 没有输出时也必须写 `no-stderr-emitted` 或等价说明
- ownerName / reviewerName 非空
- publishedVersion 非空
- managedPackageSource / runtimePackageSource 来自所选 channel
- expectedRuntimePackageKey 与 runtime package id、source 或 URL 匹配
- managed/runtime package id、version、URL、downloaded nupkg SHA256 均来自所选 channel
- managed/runtime package URL 是绝对 URL 或 `file:` URI，且 SHA256 source 与 download timestamp 能追溯到所选 channel 下载包
- noProjectReference=true
- nativeAssetsCopied=true
- dependencyProbePassed=true
- runtimeSmokePassed=true
- runtimeSmokeExitCode=0
- smokeStatus=passed
- restore/native asset/dependency probe/smoke 日志路径和 SHA256 齐全；使用 `-RequireExistingLog` 时必须与文件内容匹配
- 所有 verification items 均 passed
- 每个 required item 都有 evidence 或 evidence reference

## 不能误读

- 当前仓库 build 不是 post-publish clean consumer proof。
- post-publish backfill plan 不是 post-publish proof。
- local feed 验证不是 nuget.org 或 GitHub Packages 发布证明。
- runbook、collection package、input package、build-only、parse-only 和 sidecar-only 都不是 post-publish proof。
- bridge-only package consumer log、bridge-only wrapper surface、`Skipped=True`、`dependency-probe-only`、`WrapperSurfaceEvidenceKind=compile-surface-proof`、`IsRuntimeExecutionProof=False`、Parser/ParserRefitter diagnostic snapshots 和 copied managed diagnostic snapshot 都不是 post-publish proof。
- owner-authorized-publish-command-plan 不是 post-publish proof；它只给 owner 审阅命令和回填顺序。
- dependency probe 不是 runtime execution proof。
- 只有兼容 CUDA driver/GPU host 上的 smoke 成功，才能写 runtime smoke passed。
- `blocked-by-cuda-driver` 不能写成 `smokeStatus=passed`。
- 缺 `smokeLogSha256`、hash 格式不正确，出现 mismatched log SHA256，或 `-RequireExistingLog` 下 hash 与引用文件不匹配时，不能关闭 release issue。
- Windows handoff for Linux proof 只能作为 Linux owner 回填指引，不能替代真实 Linux runner proof。
- 缺 `consumerProjectName` / `consumerProjectPath`、host CUDA/TensorRT/cuDNN metadata、带 `--runtime-package-key` 的 `smokeCommand`、stdout/stderr 摘要（含 stderr 为空时的 `no-stderr-emitted` 说明）时，也不能关闭 release issue。
- post-publish verification 只证明发布后 clean consumer；它不能替代 external runtime proof record、Linux runner proof 或 callback runtime proof。
- post-publish verification validator 通过后仍不能跳过 `release-issue-close-record-validation`；`release-issue-close-record-template.json`、`blocked-template-only`、schema-only、preflight-only、缺 evidence bundle SHA256、缺 rollback plan 或缺 owner final close decision 的 close record 都不能关闭 release issue。
