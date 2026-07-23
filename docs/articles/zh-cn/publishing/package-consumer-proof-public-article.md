# Package Consumer Runtime Proof：为什么本地包不能证明发布可用

发布一个 TensorRT C# 项目，最危险的误区是“仓库里能 build，所以 NuGet 包一定可用”。事实上，本地源码、ProjectReference、local feed 和 direct `.nupkg` 都会绕过真实用户会遇到的包解析、runtime asset 复制和依赖加载路径。对于 TensorRtSharp4.0 这种同时涉及 managed assembly、native bridge、TensorRT、CUDA、cuDNN、GPU driver 和 runtime package key 的项目，发布前必须单独证明 package consumer 路径。

Package Consumer Runtime Proof 的目标就是补上这段证据：一个完全仓库外的 clean consumer，能否从 public package source restore 指定 managed/runtime 包，build，并在兼容 CUDA/TensorRT host 上执行 runtime smoke，留下可复核的 log、SHA256、host metadata 和 strict validator 结果。

这篇文章是 proof 说明文章，不是普通安装教程，也不是发布授权。它不会执行 `dotnet nuget push`，不会触发 GitHub Actions，也不会关闭 release issue。

## 适合谁阅读

- 准备公开发布 NuGet 包或 GitHub runtime 包的 owner。
- 需要判断 local feed、ProjectReference、direct `.nupkg` 是否能作为 proof 的评审者。
- 负责外部 clean consumer 验证、runtime smoke、post-publish verification 和 release issue close 的维护者。
- 写公众号/博客发布文章时，需要把“项目可用”和“公开包已证明可用”分清楚的作者。

## 它证明什么

package-consumer-runtime proof 只回答一个问题：真实外部用户能否在仓库外项目中消费公开包并跑通 runtime smoke。

它至少证明：

1. clean consumer project 位于仓库外。
2. consumer 没有 ProjectReference。
3. managed package 来自 public package source。
4. runtime package 来自 public package source。
5. package id、version、runtime package key 和 release target 匹配。
6. restore/build 成功。
7. native runtime assets 被复制到输出目录。
8. dependency probe 不是唯一证据，runtime smoke 真的执行。
9. smoke `exitCode=0` 且 `smokeStatus=passed`。
10. stdout/stderr、log path、log SHA256、host metadata 和 owner review 可复核。
11. strict validator 通过。

它不证明模型精度，不证明 post-publish verification，也不证明 release issue 可以关闭。真实发布还需要 owner authorization、Linux runner proof、real-model-runtime、post-publish verification 和 release close gate。

## 证据链总览

推荐把 release 证据分成五层：

```text
source quality / build-only
  -> local package consumer / bridge-only diagnostics
  -> real-model-runtime sample evidence
  -> package-consumer-runtime clean external consumer
  -> post-publish verification
```

前两层能帮助开发和打包，但不能证明公开包。real-model-runtime 能证明某个模型资产在某个 host 上可运行，但不证明 NuGet/GitHub package consumer。package-consumer-runtime 证明公开包消费路径。post-publish verification 则要求真实发布后，从真实渠道重新下载包并验证。

## 关键文件与脚本

当前仓库里的 owner input、record、validator、execution pack 和 forbidden substitute scan 分布在这些路径：

```text
artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json
artifacts/final-release/package-consumer-runtime-proof-owner-input.template.md
artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.json
artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json
artifacts/final-release/package-consumer-runtime-proof-record.json
artifacts/final-release/package-consumer-runtime-proof-record-validation.json
artifacts/final-release/package-consumer-runtime-proof-record-validation.md
artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.json
artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.md
artifacts/final-release/package-consumer-runtime-proof-execution-pack.md
artifacts/final-release/clean-consumer-proof-owner-execution-pack.md
artifacts/final-release/clean-consumer-runtime-proof-execution-checklist.md
artifacts/final-release/external-runtime-proof-record.json
artifacts/final-release/external-runtime-proof-validation.json
artifacts/final-release/external-clean-consumer-proof-kit.md
artifacts/final-release/post-publish-verification-record.json
artifacts/final-release/release-close-preflight.json
artifacts/final-release/final-release-close-blocker-dashboard.md
```

常用脚本：

```text
eng/Export-CleanConsumerProofOwnerExecutionPack.ps1
eng/Test-CleanConsumerProofOwnerExecutionPack.ps1
eng/Import-PackageConsumerRuntimeProofOwnerInput.ps1
eng/Test-PackageConsumerRuntimeProofRecord.ps1
eng/Test-ExternalRuntimeProofRecord.ps1
eng/Test-PostPublishVerificationRecord.ps1
eng/Test-ReleaseIssueCloseRecord.ps1
eng/Test-PackageConsumerRuntimeProofCandidate.ps1
eng/Export-PackageConsumerRuntimeProofCandidate.ps1
```

这些脚本可以导出模板、校验 owner input、投影 record、验证 strict proof 或生成候选材料；它们本身不上传包、不发布、不关闭 release issue。

## Clean consumer 必须满足什么

真实 clean consumer 至少需要：

```text
cleanExternalConsumerRoot
consumerProjectPath
publicPackageSourceKind
publicPackageSource
publicPackageFeedUrl
managedPackageUrl
managedPackageId
managedPackageVersion
managedNupkgSha256
runtimePackageUrl
runtimePackageId
runtimePackageVersion
runtimePackageKey
runtimeNupkgSha256
ownerName
machineName
hostOs
hostArchitecture
gpuName
cudaDriverVersion
cudaDriverSupportedRuntime
cudaRuntimeVersion
cudnnVersion
tensorRtVersion
tensorRtLine
restoreCommand
buildCommand
smokeCommand
exitCode
startedAtUtc
finishedAtUtc
dependencyProbeStatus
smokeStatus
nativeAssetsCopied
smokeLogPath
smokeLogSha256
stdoutSummary
stderrSummary
failureDiagnostic
```

如果 smoke 没有 stderr，也必须明确写入 `no-stderr-emitted` 或等价复核说明，不能留空。使用 `-RequireExistingLog` 时，validator 会读取真实日志并重新计算 SHA256；只有 `logSha256Matches=true`，日志证据才匹配。

## 推荐执行顺序

先生成 owner 执行包和字段说明：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CleanConsumerProofOwnerExecutionPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-CleanConsumerProofOwnerExecutionPack.ps1 -Strict
```

owner 在兼容 CUDA/TensorRT host 上创建仓库外 consumer，使用 public package source 安装包：

```powershell
dotnet new console -n TensorRtSharpConsumerProof
cd TensorRtSharpConsumerProof
dotnet add package JYPPX.TensorRT.CSharp.API --version <public-version> --source <public-package-source>
dotnet add package <runtime-package-id> --version <public-version> --source <public-package-source>
dotnet restore
dotnet build -c Release --no-restore
dotnet run -c Release -- --runtime-package-key <runtime-package-key>
```

回填 owner input 后导入和验证：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-PackageConsumerRuntimeProofOwnerInput.ps1 `
  -InputPath .\artifacts\final-release\package-consumer-runtime-proof-owner-input.template.json `
  -Strict

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 `
  -Strict `
  -RequireExistingLog `
  -FailOnNotProof
```

如果走 `external-runtime-proof-record` 路线，还要执行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 `
  -InputPath .\artifacts\final-release\external-runtime-proof-record.json `
  -RequireExistingLog `
  -FailOnNotProof
```

post-publish verification 是另一条更靠后的真实发布后验证：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 `
  -InputPath .\artifacts\final-release\post-publish-verification-record.json `
  -RequireExistingLog `
  -FailOnNotProof
```

## 导入与校验脚手架

Owner 拿到真实 clean consumer 结果后，不应该直接改 release proof record，而应该先导入 owner input。导入脚手架会生成：

```text
artifacts/final-release/package-consumer-runtime-proof-owner-input.imported.json
artifacts/final-release/package-consumer-runtime-proof-owner-input-import.json
artifacts/final-release/package-consumer-runtime-proof-owner-input-import.md
artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json
artifacts/final-release/package-consumer-runtime-proof-record.json
artifacts/final-release/package-consumer-runtime-proof-record-validation.json
artifacts/final-release/package-consumer-runtime-proof-record-validation.md
```

这一步只复制、校验和投影 owner 输入；它不会执行 `dotnet nuget push`，不会关闭 release issue，也不会把 template、dry-run、build-only、local feed、ProjectReference 或 direct `.nupkg` 晋级为 proof。

## Strict validator 会拒绝什么

validator 必须拒绝以下情况：

- `cleanExternalConsumerRoot` 在仓库内部。
- consumer 使用 `ProjectReference`。
- package source 是 local feed。
- 使用 direct `.nupkg` 路径安装。
- 只有 build/restore，没有 runtime smoke。
- 只有 dependency-probe-only。
- `smokeStatus` 不是 passed。
- `exitCode` 不是 0。
- `nativeAssetsCopied` 不是 true。
- `managedNupkgSha256` 或 `runtimeNupkgSha256` 不是 64 位 SHA256。
- log path 缺失或 `smokeLogSha256` 不匹配。
- host metadata 缺失。
- ownerName、machineName、review timestamp 缺失。
- proofClassification 不是 `package-consumer-runtime`。
- `isPackageConsumerRuntimeProof=false` 或 `canPromoteProof=false`。
- dry-run、queued GitHub Actions run、missing self-hosted runner、dashboard、template、skipped run 被当成 proof。

这些拒绝不是“流程太严格”，而是为了防止用户将来按公开包安装时踩到未验证路径。

## 禁止替代项

以下内容不能作为 package-consumer-runtime proof：

- local feed package consumer。
- ProjectReference consumer。
- direct `.nupkg` install。
- template。
- draft。
- input draft。
- owner execution package。
- collection package。
- dry-run。
- GitHub Actions dry-run。
- queued GitHub Actions run。
- missing self-hosted runner。
- build-only report。
- dependency-probe-only log。
- bridge-only diagnostics。
- compile-surface proof。
- TensorRtExec report。
- TensorRtExec GUI screenshot。
- command preview。
- OnnxToEngine report。
- YoloVision candidate template。
- YoloVision matrix。
- sample-run evidence。
- real-model-runtime。
- post-publish input draft。
- release close preflight。
- stale claim audit。

这些材料有价值，但它们属于开发证据、教程证据、准备材料或其他 proof lane，不是 package consumer release proof。

## 与 YoloVision / TensorRtExec 的关系

YoloVision 可以产生 real-model-runtime proof 的候选材料：模型、labels、输入图、preprocessed tensor、output JSON、SVG visualization、`YoloVision Passed=True` run log、sample-run-evidence record 和 owner review。它证明的是某个真实模型样例，不证明公开包可被外部 consumer 消费。

TensorRtExec 可以产生 build-only report、conversion diagnostics、engine readback、normalized command SHA256、`OptionImplementationStatus` 和 evidence sidecar。它证明的是工具命令和构建/诊断路径，不证明 package-consumer-runtime。

OnnxToEngine 可以提供教程型 ONNX -> engine build path。它同样不能替代 clean external consumer。

## 与 release close 的关系

package-consumer-runtime 是 release close 的关键 blocker 之一，但不是唯一 blocker。最终发布和关闭 release issue 还需要：

```text
owner authorization
package-consumer-runtime
Linux runner proof
real-model-runtime
post-publish verification
release close preflight
release issue close record
```

在真实 proof 缺失时，状态必须保持：

```text
blocked-real-proof-required
canPublishPublicly=false
canCloseReleaseIssue=false
performsPublish=false
```

最终 close 仍需 `eng/Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady` 通过，并由 owner 手动关闭。自动化脚本可以辅助验证，但不应替 owner 做公开发布决定。

## 配图建议

- 内部 build、real-model-runtime、package-consumer-runtime、post-publish verification 四层证据对比图。
- clean consumer 的 restore -> build -> native asset listing -> dependency probe -> runtime smoke -> strict validator 流程图。
- owner input JSON 字段标注图，突出 package source、runtime key、SHA256、host metadata 和 log hash。
- forbidden substitutes 表格：local feed、ProjectReference、direct `.nupkg`、dry-run、template、build-only、dependency-probe-only 为什么都不能替代。

## 下一步

发布前应冻结公开包版本、hash、runtime package key 和 clean consumer 记录，再由 owner 手动执行发布或关闭 release issue。当前用户已明确本月 GitHub Actions 额度用完，所以不要 workflow dispatch、不要 push、不要 NuGet/GitHub Packages 发布、不要上传 GitHub Release；本阶段只能继续完善文档、门禁和 owner proof 输入质量。等 owner 提供真实 public package source、clean consumer log、hash、host metadata 和 strict validator 结果后，再进入发布动作。
