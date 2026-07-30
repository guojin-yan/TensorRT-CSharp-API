# 包策略：GitHub Release 与 NuGet 的 Bridge-only 双路线

TensorRtSharp4.0 不再打包或发布 CUDA、cuDNN、TensorRT、NVRTC 及其 builtins 等 NVIDIA 原厂运行库。公开消费仍保留两条路线，但两条路线都只交付项目源码编译得到的 managed API 与 C++ bridge DLL/.so：GitHub Release 提供带不可变 URL 和 SHA256 digest 的 `.nupkg` 资产，NuGet-compatible source 提供常规 `PackageReference` 体验。用户必须自行安装与 runtime key 匹配的 NVIDIA 依赖。

## 适合

- 准备用 NuGet、GitHub Packages 或 GitHub Release 分发 TensorRtSharp4.0 的维护者。
- 需要理解 managed API、C++ bridge、CUDA、TensorRT、cuDNN 和 runtime split package 关系的用户。
- 负责 owner proof、clean external consumer、post-publish verification 和 package-consumer-runtime proof 的发布负责人。
- 想判断 local feed、ProjectReference、direct `.nupkg` 和 build-only 记录能否进入 release close 的审核者。

## 双发布路线

| 路线 | 交付内容 | 适合用户 | 证据边界 |
| --- | --- | --- | --- |
| GitHub Release managed + bridge assets | `JYPPX.TensorRT.CSharp.API` 与匹配的 `.Bridge` 包、Release URL、GitHub digest 和源码归档 | 需要从 GitHub Release 获取不可变资产的用户 | 下载后必须核对 URL、digest、package id/version 和 bridge-only 内容；隔离 restore staging 不能混入本地构建包 |
| NuGet managed + bridge packages | `JYPPX.TensorRT.CSharp.API` 与按 RID/TRT/CUDA 组合编译的 `.Bridge` 包 | 已自行安装 NVIDIA runtime、希望使用标准 `PackageReference` 的用户 | 必须记录公开 NuGet source、解析版本和下载 hash；restore/build 本身不是 runtime proof |

这两条路线可以同时存在，区别只在公开获取通道，不在打包范围。两者都不得携带 NVIDIA 原厂运行库，也不能用旧的 full-runtime、`CudaCudnn`、`TensorRt`、`CudaRtc`、collection 或 meta 包作为正式发布资产。

GitHub Release 资产路线使用仓库提供的执行器验证远端 URL、GitHub digest、package id/version、包内容和仓库外 runtime smoke：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Invoke-PublicReleaseBridgePackageConsumer.ps1 `
  -ManagedReleaseTag <managed-tag> `
  -BridgeReleaseTag <bridge-tag> `
  -SourceRuntimeKey win-x64-trt10.11-cuda12.9-cudnn9.22
```

执行器把公开下载资产放入隔离 NuGet restore staging，但项目仍只使用 `PackageReference`，不会直接引用 `.nupkg` 或 DLL。只有远端 URL、GitHub SHA256、下载 SHA256、包身份、nuspec `repository commit` 和 bridge-only 内容全部一致时才继续；managed 与 bridge 必须来自同一源码提交，staging 中也不能混入本地构建包。

执行器会在清理临时下载之前自动调用独立验证器，重新计算两个 `.nupkg`、runtime JSON、stdout 和 stderr 的 SHA256，并重新读取 nuspec 与 bridge native entries。保存证据后也可以显式复核：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PublicReleaseBridgePackageConsumer.ps1 `
  -InputPath artifacts\public-release-consumer\<runtime-key>\public-release-bridge-package-consumer.json `
  -RequireReferencedFiles `
  -Strict `
  -FailOnNotEvidence
```

`-AllowCrossCommitPair` 只用于调查历史资产不齐的情况。它会强制进入 diagnostic-only 路径并关闭已安装 vendor asset 的 hash 晋级条件；即使进程完成了 CUDA/TensorRT 调用，也必须保持 `isPublicReleaseAssetConsumerEvidence=false`、`isPackageConsumerRuntimeProof=false` 和 `isPostPublishProof=false`。同提交但误加该参数同样会 fail closed。

## 包结构

托管侧核心包面向 C# API 和上层使用者：

```text
JYPPX.TensorRT.CSharp.API
JYPPX.CudaSharp
JYPPX.TensorRtSharp
```

native 侧按平台、TensorRT ABI 和 CUDA toolchain 组合编译 bridge。唯一允许公开发布的 native 包角色是：

```text
Bridge
```

典型 split runtime package id 形如：

```text
JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge
```

历史 manifest 中仍可能保留 `CudaCudnn`、`TensorRt`、full-runtime、collection 和 meta identity，用于远端清理、兼容审计和历史证据解释。它们的项目必须保持 `IsPackable=false`，不能重新进入 pack、push 或 Release upload。

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

`pack/JYPPX.TensorRT.CSharp.API/JYPPX.TensorRT.CSharp.API.csproj` 是 managed API 包的打包入口，
它收拢 `JYPPX.Shared`、`JYPPX.TensorRtSharp` 和 `JYPPX.CudaSharp` 的托管产物；README 则说明
consumer 需要引用的托管 API。`pack/runtime/runtime-packages.manifest.json` 只用于描述 runtime key、用户安装路径和编译输入，
不是 full-runtime 发布清单。`pack/runtime-split/split-runtime-packages.manifest.json` 的 `publicationPolicy.state=bridge-only`，
只有 `role=bridge` 的条目可打包；其余条目保留为历史 identity 且不可 pack。

`runtime-packages.local.example.json` 是 owner 本地路径模板；真正机器上的
`runtime-packages.local.json` 只说明“这台机器如何找到 NVIDIA 资产”，不能写入公开文章作为下载来源，
也不能当成 public package source。`runtime-package-smoke-command-template.json` 是 smoke 命令模板，
它能帮助 owner 统一 restore/build/probe/smoke 命令形状，但模板本身不是执行日志。

## Bridge-only 与用户自装依赖

每个 runtime key 只发布一个项目自有 bridge 包：

```text
JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda12.9.cudnn9.22.Bridge
```

bridge 包只包含 `jyppxtrtbridge.dll` 或 `libjyppxtrtbridge.so`。CUDA、cuDNN、TensorRT、parser、plugin、
builder resource、NVRTC 和 NVRTC builtins 必须来自用户机器上的 NVIDIA 安装。release owner 仍需记录
managed/bridge nupkg 的 SHA256、source URL、package id 和 version，同时 clean consumer 需要记录实际加载的
外部依赖路径与版本，不能把 dependency probe 当作 runtime execution proof。

TRT8、TRT10 和 TRT11 的包策略不要混写。TRT8 常见于 CUDA 11.8/12.1 和 cuDNN 8.9；TRT10/11
常见于 CUDA 12.9/13.2 和 cuDNN 9.22。文章可以解释兼容矩阵，但不能暗示一个 runtime key 能覆盖所有
driver、GPU 架构或 TensorRT ABI。

## 发布前矩阵与候选审计

维护者可以用这些脚本生成包策略相关的候选证据：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalVendorRuntimePackagePolicy.ps1 -StaticOnly
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RuntimePackageReadiness.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidatePackageInventory.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PreReleasePackageProofReadinessMatrix.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerDualRouteProofPlan.ps1
```

这些脚本解决的问题不同：

- `Invoke-LocalRuntimePackage.ps1` 已 fail closed；`Invoke-LocalSplitRuntimePackage.ps1` 只允许 `bridge`。
- `Test-ExternalVendorRuntimePackagePolicy.ps1` 拒绝 vendor runtime binary 和非 managed/bridge package id。
- `Test-RuntimePackageReadiness.ps1` 检查 runtime package readiness，不等于 clean consumer proof。
- `Export-ReleaseCandidatePackageInventory.ps1` 记录候选 `.nupkg` 的路径、大小和 hash。
- `Export-PreReleasePackageProofReadinessMatrix.ps1` 和 `Export-PackageConsumerDualRouteProofPlan.ps1`
  把 managed/bridge、GitHub Release/NuGet 获取通道、clean consumer、post-publish verification 和 blocker
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

公开文章可以说“项目提供 GitHub Release 与 NuGet-compatible source 两种 managed + bridge-only 获取通道”，但不能说“公开发布已经完成”或“runtime proof 已完成”，除非 owner proof、clean external consumer、post-publish verification 和 strict validator 已经全部通过。

## 配图建议

- 一张 managed API、Bridge 与用户自装 CUDA/TensorRT/cuDNN 的依赖关系图。
- 一张 GitHub Release assets 和 NuGet managed + bridge packages 的双路线流程图。
- 一张 owner result input JSON 截图，突出 `resultInputs[]`、SHA256 和 `nonSubstituteConfirmations` 字段。
- 一张 release evidence ladder，标出 local build、candidate inventory、public package download、clean external consumer、strict validator 和 release close 的分层。

## 下一步

包策略收口后，继续执行 clean external consumer 与 post-publish verification。只有 strict validator 接受真实 owner result 后，候选记录才能进入 release close bridge；本地 pack、local feed、direct `.nupkg`、ProjectReference、dependency-probe-only、build-only、截图、dashboard 和文章都只能作为说明或候选证据，不能单独推动发布。
