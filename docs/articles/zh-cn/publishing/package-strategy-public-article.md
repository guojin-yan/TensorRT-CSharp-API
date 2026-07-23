# 包策略：GitHub Full Runtime 与 NuGet 小包双路线

TensorRtSharp4.0 的包策略不是“一个 NuGet 包装下所有东西”，而是把公开消费拆成两条路线：GitHub full runtime 包负责大体积 native runtime 的整包分发，NuGet small bridge/core 包负责托管 API 和小型 bridge 的常规 .NET 引用体验。这样做可以同时照顾开箱即用、包大小、NVIDIA runtime 再分发边界和发布证据可审计性。

## 适合

- 准备用 NuGet、GitHub Packages 或 GitHub Release 分发 TensorRtSharp4.0 的维护者。
- 需要理解 managed API、C++ bridge、CUDA、TensorRT、cuDNN 和 runtime split package 关系的用户。
- 负责 owner proof、clean external consumer、post-publish verification 和 package-consumer-runtime proof 的发布负责人。
- 想判断 local feed、ProjectReference、direct `.nupkg` 和 build-only 记录能否进入 release close 的审核者。

## 双发布路线

| 路线 | 交付内容 | 适合用户 | 证据边界 |
| --- | --- | --- | --- |
| GitHub full runtime 包 | managed API、C++ bridge DLL、CUDA/TensorRT/cuDNN runtime assets、版本矩阵、SHA256 和 release asset metadata | 想开箱即用、能接受大包下载的用户 | 必须有公开 GitHub asset、下载元数据、hash、clean external consumer 日志和 owner 授权 |
| NuGet small bridge/core 包 | `JYPPX.TensorRT.CSharp.API` 托管 API，以及按平台和 SDK 组合拆分的小型 runtime 包 | 已在机器上安装或能自行管理 NVIDIA runtime 的 .NET 用户 | 必须说明用户负责 CUDA/TensorRT/cuDNN 安装与版本匹配；NuGet restore/build 本身不是 runtime proof |

这两条路线可以同时存在。GitHub full runtime 包解决大依赖分发问题；NuGet 小包降低引用门槛，便于普通业务项目先建立 managed API 依赖，再按环境选择 runtime package key。

## 包结构

托管侧核心包面向 C# API 和上层使用者：

```text
JYPPX.TensorRT.CSharp.API
JYPPX.CudaSharp
JYPPX.TensorRtSharp
```

runtime 侧按平台、TensorRT、CUDA、cuDNN 和角色拆分。当前公开文档和 metadata audit 使用的核心角色是：

```text
Bridge
CudaCudnn
TensorRt
```

典型 split runtime package id 形如：

```text
JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge
JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.CudaCudnn
JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.TensorRt
```

full runtime 包则更接近单个大包，例如 release candidate inventory 中的本地候选：

```text
JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda13.2.cudnn9.22
```

这些本地候选可以记录 size、SHA256 和路径，但在公开渠道下载、外部 consumer 运行、owner 回填和 strict validator 通过前，仍然只是 candidate inventory，不是 post-publish proof。

## Runtime Package Key

runtime package key 用来表达一个可审计的 native runtime 组合，推荐保持可读且可机械解析：

```text
win-x64-trt10.11-cuda12.9-cudnn9.22
win-x64-trt11.0-cuda13.2-cudnn9.22
win-x64-trt8.6-cuda11.8-cudnn8.9
```

选择 key 时要同时检查：

- 操作系统和 RID，例如 `win-x64`。
- TensorRT 主版本和小版本，例如 `trt10.11` 或 `trt11.0`。
- CUDA 版本线，例如 `cuda12.9` 或 `cuda13.2`。
- cuDNN 版本线，例如 `cudnn9.22`。
- bridge DLL 是否由同一组 header/lib 生成并验证。
- consumer 机器的 GPU driver 是否满足 CUDA runtime 要求。

不要只因为 `dotnet restore` 成功就判定 runtime key 可用。restore 只能说明包解析成功，不能说明 native DLL 已加载、TensorRT engine 可反序列化、CUDA context 可创建，或 smoke 已在真实 host 上通过。

## 本地脚本的用途

本地脚本用于收集资产、生成候选包、导出矩阵和检查文档元数据。它们对维护者很重要，但它们的产物默认都是 pre-publish evidence。

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Collect-SplitRuntimeAssets.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-DualPackagePublishPreflightMatrix.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalOwnerExecutionChecklist.ps1
```

相关审计文件可以说明候选包和文档是否齐备：

```text
artifacts/final-release/release-docs-and-nuget-metadata-audit.json
artifacts/final-release/release-candidate-package-inventory.md
artifacts/final-release/release-candidate-package-inventory.json
```

它们不能替代公开发布后的下载证据，也不能替代真实 package-consumer-runtime proof。`failedBlockerCount=0` 只能说明该审计门没有 blocker，不等于可以发布、不等于公开包已存在，也不等于 runtime smoke 已通过。

## 仓库中的包入口

如果读者要从源码理解双路线，建议先看这些固定入口，而不是从临时 `bin`、`obj` 或本地 `.nupkg` 反推：

```text
pack/JYPPX.TensorRT.CSharp.API/JYPPX.TensorRT.CSharp.API.csproj
pack/JYPPX.TensorRT.CSharp.API/README.md
pack/runtime/Directory.Build.props
pack/runtime/README.md
pack/runtime/runtime-packages.manifest.json
pack/runtime/linux-runtime-targets.manifest.json
pack/runtime/runtime-package-smoke-command-template.json
pack/runtime/runtime-packages.local.example.json
pack/runtime-split/Directory.Build.props
pack/runtime-split/README.md
pack/runtime-split/split-runtime-packages.manifest.json
```

`pack/JYPPX.TensorRT.CSharp.API/JYPPX.TensorRT.CSharp.API.csproj` 是 managed/core 小包的打包入口，
它收拢 `JYPPX.Shared`、`JYPPX.TensorRtSharp` 和 `JYPPX.CudaSharp` 的托管产物；README 则说明
consumer 需要引用的托管 API。`pack/runtime/runtime-packages.manifest.json` 描述 full runtime 候选，
它按 RID、TensorRT、CUDA、cuDNN、bridge 和 native asset 组合组织。`pack/runtime-split/split-runtime-packages.manifest.json`
描述 split runtime 组件包，负责把 Bridge、CudaCudnn、TensorRt 以及 TRT11 builder/runtime 分片拆开，
让包大小和复用关系更可控。

`runtime-packages.local.example.json` 是 owner 本地路径模板；真正机器上的
`runtime-packages.local.json` 只说明“这台机器如何找到 NVIDIA 资产”，不能写入公开文章作为下载来源，
也不能当成 public package source。`runtime-package-smoke-command-template.json` 是 smoke 命令模板，
它能帮助 owner 统一 restore/build/probe/smoke 命令形状，但模板本身不是执行日志。

## Full Runtime 与 Split Runtime 的取舍

full runtime 包把一个 runtime key 下的 bridge、CUDA、cuDNN 和 TensorRT 资产合进单个包。它适合
GitHub Release asset 或 GitHub Packages 中的“整包下载”路线，优点是 consumer 只需要选中一个 runtime
key；缺点是包体大、更新粒度粗，并且每条 TRT/CUDA/cuDNN 组合都要重新产生完整证据。

split runtime 包把同一 runtime key 拆成组件：

```text
JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda12.9.cudnn9.22.Bridge
JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda12.9.cudnn9.22.CudaCudnn
JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda12.9.cudnn9.22.TensorRtRuntime
JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda12.9.cudnn9.22.TensorRtBuilder.Sm75Sm86
JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda12.9.cudnn9.22.TensorRtBuilder.Sm89Sm90
JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda12.9.cudnn9.22.TensorRtBuilder.Sm100Sm120Ptx
```

split meta package 通过 `PackageReference` pin 住同一 runtime key 下的组件版本。这样做的意义是：
Bridge 包可以跟 managed API 一起快速验证，TensorRT builder kernel 按 SM 分片，runtime-only consumer
不用下载不需要的 builder 资产。缺点是 release owner 必须记录每个组件 nupkg 的 SHA256、source URL、
package id、version 和 meta package pin，不能只记录 meta package 名称。

TRT8、TRT10 和 TRT11 的包策略不要混写。TRT8 常见于 CUDA 11.8/12.1 和 cuDNN 8.9；TRT10/11
常见于 CUDA 12.9/13.2 和 cuDNN 9.22。文章可以解释兼容矩阵，但不能暗示一个 runtime key 能覆盖所有
driver、GPU 架构或 TensorRT ABI。

## 发布前矩阵与候选审计

维护者可以用这些脚本生成包策略相关的候选证据：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Invoke-LocalRuntimePackage.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Resolve-SplitPackagePins.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Validate-SplitRuntimePackages.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RuntimePackageReadiness.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidatePackageInventory.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PreReleasePackageProofReadinessMatrix.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerDualRouteProofPlan.ps1
```

这些脚本解决的问题不同：

- `Invoke-LocalRuntimePackage.ps1` 和 `Invoke-LocalSplitRuntimePackage.ps1` 生成本地候选包。
- `Resolve-SplitPackagePins.ps1` 检查 split meta package 的组件 pin。
- `Validate-SplitRuntimePackages.ps1` 检查 split package manifest、项目和资产布局是否一致。
- `Test-RuntimePackageReadiness.ps1` 检查 runtime package readiness，不等于 clean consumer proof。
- `Export-ReleaseCandidatePackageInventory.ps1` 记录候选 `.nupkg` 的路径、大小和 hash。
- `Export-PreReleasePackageProofReadinessMatrix.ps1` 和 `Export-PackageConsumerDualRouteProofPlan.ps1`
  把 full runtime、split runtime、managed package、clean consumer、post-publish verification 和 blocker
  状态放进同一张 release owner 视图。

候选审计输出通常落在 `artifacts/final-release`，例如：

```text
dual-package-publish-preflight-matrix.json
dual-package-publish-preflight-matrix.md
pre-release-package-proof-readiness-matrix.json
pre-release-package-proof-readiness-matrix.md
package-consumer-dual-route-proof-plan.json
package-consumer-dual-route-proof-plan.md
release-package-proof-bundle.json
release-package-proof-bundle.md
public-package-url-hash-verification-candidate.json
public-package-url-hash-verification-candidate.md
```

这些文件可以帮助 owner 发现 package id、runtime key、hash、source URL 和 smoke 字段缺口；它们不能
替代 owner 在公开渠道下载包、计算 hash、创建仓库外 clean consumer 并执行 smoke 的结果。

## Public Package Source 与 Hash

公开渠道 proof 至少要绑定以下字段：

```text
publicPackageSource
publicPackageUrl
packageId
packageVersion
runtimePackageKey
downloadedNupkgSha256
expectedNupkgSha256
packageHashMatch
cleanConsumerRoot
packageReferenceOnly
restoreCommand
buildCommand
smokeCommand
smokeExitCode
nativeAssetsCopied
mergedTranscriptSha256
ownerReviewer
ownerReviewTimestampUtc
```

`publicPackageUrlHashVerification` 只能证明下载 URL 和 hash 的匹配关系；`publicPackageDownloadProof`
只能证明公开包可下载。只有当这些下载证据和 clean external consumer restore/build/runtime smoke
记录合并，并由 strict validator 接受后，才可以讨论 package-consumer-runtime proof。若 `packageHashMatch=false`、
`packageReferenceOnly=false`、`cleanConsumerRoot` 位于仓库内、`smokeExitCode` 非 0，或 log hash 缺失，
这条证据必须留在 blocker 状态。

## Release Close 边界

包策略文章能帮助读者理解发布路线，但它本身不能关闭 release。release close 至少还要同时满足：

- public package download proof 已通过，且 managed/runtime nupkg SHA256 与 owner 回填一致。
- clean external consumer 使用公开包源和 PackageReference-only 项目，restore/build/runtime smoke 均成功。
- package-consumer-runtime proof record 通过 strict validator。
- post-publish verification 记录真实 channel URL、下载 hash、consumer log、stdout/stderr summary 和 host metadata。
- Linux runner proof、real-model-runtime proof、owner authorization 和 release issue close record 按 release gate 要求齐备。

缺少上述任一项时，`canPublishPublicly=false` 或 `canCloseReleaseIssue=false` 不能被文章、README、
dashboard、matrix ready、candidate inventory ready、`failedBlockerCount=0` 或 dry-run output 覆盖。

## Clean Consumer Proof

package-consumer-runtime proof 必须来自仓库外部的 clean external consumer，并且使用公开包源。最小闭环应包含：

```powershell
dotnet new console -n TensorRtSharpConsumerProof
cd TensorRtSharpConsumerProof
dotnet add package JYPPX.TensorRT.CSharp.API --version <public-version>
dotnet add package <runtime-package-id> --version <public-version>
dotnet restore
dotnet build -c Release
dotnet run -c Release
```

proof 记录至少要能回答这些问题：

- 包是否来自公开 NuGet、GitHub Packages 或 GitHub Release URL。
- 下载到的 `.nupkg` 或 release asset SHA256 是否与 owner 回填一致。
- consumer 是否在仓库外部创建，且没有 ProjectReference。
- restore、build、runtime smoke 的 stdout、stderr、merged transcript 和 validator output 是否都有 SHA256。
- host runtime metadata 是否记录 GPU、driver、CUDA、TensorRT、cuDNN、RID 和 .NET SDK。
- native assets 是否被复制到 consumer 输出目录并由 smoke 实际加载。
- 退出码是否为 `0`，且 strict validator 是否接受。

## 真实 Proof 输入

发布闭环以 owner 回填结果为准。关键输入文件是：

```text
artifacts/final-release/owner-external-proof-execution-result.input.template.json
artifacts/final-release/owner-external-proof-execution-result.input.json
```

每个 `resultInputs[]` 项至少要包含：

```text
resultInputId
packageIdentity.nupkgPath
packageIdentity.nupkgSha256
stdoutPath / stdoutSha256
stderrPath / stderrSha256
mergedTranscriptPath / mergedTranscriptSha256
validatorOutputPath / validatorOutputSha256
exitCode
passed
ownerReviewer
ownerReviewTimestampUtc
nonSubstituteConfirmations
```

导入和验证命令：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-OwnerExternalProofExecutionResult.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerExternalProofExecutionResultImport.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealProofRecordCandidateFromOwnerResultImport.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealProofRecordCandidateFromOwnerResultImport.ps1 -Strict
```

## 不能替代 Proof 的内容

以下内容可以用于开发、排障或候选审计，但不能作为 package-consumer-runtime proof：

- local feed consumer。
- ProjectReference consumer。
- direct `.nupkg` install。
- build-only report。
- dependency-probe-only report。
- README、截图、dashboard 或文章。
- candidate record、draft release、dry-run、template JSON。
- `failedBlockerCount=0`、package inventory ready 或 metadata audit ready。
- blocked-by-cuda-driver 结论。

公开文章可以说“项目提供 GitHub full runtime 包和 NuGet small bridge/core 包两条路线”，但不能说“公开发布已经完成”或“runtime proof 已完成”，除非 owner proof、clean external consumer、post-publish verification 和 strict validator 已经全部通过。

## 配图建议

- 一张 managed API、Bridge、CudaCudnn、TensorRt split package 与 full runtime 包的关系图。
- 一张 GitHub full runtime 包和 NuGet small bridge/core 包的双路线流程图。
- 一张 owner result input JSON 截图，突出 `resultInputs[]`、SHA256 和 `nonSubstituteConfirmations` 字段。
- 一张 release evidence ladder，标出 local build、candidate inventory、public package download、clean external consumer、strict validator 和 release close 的分层。

## 下一步

包策略收口后，继续执行 clean external consumer 与 post-publish verification。只有 strict validator 接受真实 owner result 后，候选记录才能进入 release close bridge；本地 pack、local feed、direct `.nupkg`、ProjectReference、dependency-probe-only、build-only、截图、dashboard 和文章都只能作为说明或候选证据，不能单独推动发布。
