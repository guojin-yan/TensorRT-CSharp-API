# TensorRtSharp4.0 NuGet 安装与 Runtime 包选择

TensorRtSharp4.0 的安装体验不能只看一个 managed NuGet 包。真正能跑起来，还要匹配 RID、CUDA、TensorRT、cuDNN、native bridge、Windows/Linux runtime asset、GPU driver 和目标 .NET 版本。本文面向第一次集成项目的 C# 开发者，说明如何选择包、如何验证依赖、以及哪些内容不能被当成 package-consumer-runtime proof。

## 适合谁阅读

- 准备在自己的 .NET 项目中引用 TensorRtSharp4.0 的用户。
- 需要区分 managed package、runtime package、split runtime package 的发布维护者。
- 遇到 DLL 加载、CUDA runtime、TensorRT 版本组合问题的工程师。
- 需要把 NuGet 安装教程写成公开文章，但又不能越界承诺发布 proof 的文章作者。

## 安装思路

推荐把安装拆成四层看：

1. managed API：`JYPPX.TensorRT.CSharp.API`，也就是 C# wrapper、interop、工具类和高层对象。
2. native bridge：`jyppxtrtbridge.dll` 或 Linux shared object，负责稳定 ABI 边界。
3. vendor runtime assets：TensorRT、CUDA runtime、cuDNN 对应的 DLL/shared object。
4. owner proof：外部 clean consumer 使用公开包执行 restore/build/runtime smoke 后留下的记录。

前三层解决“项目能不能引用、复制和加载 native 文件”；第四层才解决“公开包是否在干净外部 consumer 中真实可用”。因此 NuGet 安装文章可以帮助用户安装，但不是 package-consumer-runtime proof。

## 两条 runtime 包路线

仓库当前把 runtime 包信息放在两个 manifest 中：

```text
pack/runtime/runtime-packages.manifest.json
pack/runtime-split/split-runtime-packages.manifest.json
pack/runtime/runtime-package-smoke-command-template.json
pack/runtime/README.md
pack/runtime-split/README.md
docs/articles/zh-cn/tensorrtsharp-nuget-runtime-package-guide.md
docs/articles/zh-cn/runtime-package-selection.md
docs/articles/zh-cn/runtime-package-matrix-reading-guide.md
docs/articles/zh-cn/runtime-package-installation-deep-dive.md
docs/articles/zh-cn/package-consumer-runtime-proof-clean-consumer-guide.md
```

`pack/runtime/runtime-packages.manifest.json` 描述整包路线：一个 runtime package 同时携带 bridge、TensorRT、CUDA runtime 和 cuDNN。典型 key 形如：

```text
win-x64-trt8.6-cuda11.8-cudnn8.9
win-x64-trt8.6-cuda12.1-cudnn8.9
win-x64-trt10.11-cuda11.8-cudnn8.9
win-x64-trt10.11-cuda12.9-cudnn9.22
win-x64-trt11.0-cuda12.9-cudnn9.22
win-x64-trt11.0-cuda13.2-cudnn9.22
linux-x64-trt10.11-cuda12.9-cudnn9.22
```

整包路线更容易理解，但包体积更大。文章中可以说它适合第一次验证、内网分发或 release candidate 复核，不能说所有组合都已经公开发布。

`pack/runtime-split/split-runtime-packages.manifest.json` 描述拆分路线：同一个 source runtime key 拆成三类角色。

```text
role = bridge
role = cuda-cudnn
role = tensorrt
JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge
JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.CudaCudnn
JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.TensorRt
```

拆分路线让 bridge、CUDA/cuDNN 和 TensorRT 可以独立更新，也能减少重复下载。它更适合 GitHub full runtime 包、私有 feed 或后续公开拆包策略。无论整包还是拆分包，runtime key 都必须和实际 host metadata 对齐。

## 如何选择 runtime key

选择顺序建议固定：

1. 先看 RID：`win-x64` 或 `linux-x64`。
2. 再看 TensorRT line：TRT8、TRT10、TRT11。
3. 再看 CUDA line：CUDA 11、CUDA 12、CUDA 13。
4. 再看 cuDNN major：cuDNN 8 或 cuDNN 9。
5. 最后看 GPU driver 是否支持对应 CUDA runtime。

不要只因为机器上有 `CUDA_PATH`、`PATH` 中能找到某个 DLL、或者本地装过 TensorRT，就随便选包。NuGet consumer 依赖的是包内 native assets 是否被复制到输出目录、loader 是否能解析它们、以及主机 driver/runtime 是否兼容。

关键字段应从 manifest 或 owner proof input 中读取：

```text
key
packageId
rid
platform
tensorRtLine
tensorRtVersion
cudaLine
cudaVersion
cudnnMajor
cudnnVersion
distributionTier
validationState
buildPreset
bridgeFile
tensorRtFiles
cudaFiles
cudnnFiles
```

如果文章、README 或教程需要写具体包名，推荐写成“以发布清单为准”的显式占位，而不是从本机 `.nupkg` 文件名、Downloads 截图或 local feed 推断公开可用性。

## 基础安装命令

在业务项目中引用 managed 包：

```powershell
dotnet new console -n TensorRtSharpApp
cd TensorRtSharpApp
dotnet add package JYPPX.TensorRT.CSharp.API --version <public-version>
```

再按目标环境选择 runtime 包。整包路线示例：

```powershell
dotnet add package JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22 --version <public-version>
dotnet restore
dotnet build -c Release
```

拆分路线示例：

```powershell
dotnet add package JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge --version <public-version>
dotnet add package JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.CudaCudnn --version <public-version>
dotnet add package JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.TensorRt --version <public-version>
dotnet restore
dotnet build -c Release
```

实际 package id、版本、渠道和 runtime key 必须以 owner 发布清单为准。本文不会执行 `dotnet nuget push`，也不会暗示 GitHub Actions dry-run 已经产生公开包。

## Package source 与缓存边界

真实用户安装时，首先要确认包源，而不是从本机 `.nupkg` 文件名倒推：

```powershell
dotnet nuget list source
dotnet nuget add source <public-or-owner-approved-source> --name TensorRtSharpPublic
dotnet nuget locals all --list
```

公开文章应把 NuGet.org、GitHub Packages、GitHub Release asset 和企业内网 feed 分开写。`public package source` 必须是 owner 明确发布的渠道；local folder source 只能用于开发诊断，不能出现在 clean consumer proof 的 package source 字段里。

NuGet 默认全局缓存可能在用户目录的 C 盘，例如 `%UserProfile%\\.nuget\\packages`。如果机器空间紧张，可以在普通用户项目中设置：

```powershell
$env:NUGET_PACKAGES = "E:\\NuGetPackages"
dotnet restore --force-evaluate
```

但这只是缓存位置调整，不改变 proof 语义。不要把 runtime `.nupkg`、TensorRT/CUDA/cuDNN 大文件、ONNX、engine 或 plan 放进 C 盘 Temp/Downloads 作为教程默认路径。文章示例继续使用 `E:\\TensorRtSharpAssets` 或仓库外固定 workspace。

## PackageReference-only consumer

一个干净 consumer 的项目文件应只包含 package 引用，不应出现仓库内 ProjectReference：

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <RuntimeIdentifier>win-x64</RuntimeIdentifier>
    <PlatformTarget>x64</PlatformTarget>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="JYPPX.TensorRT.CSharp.API" Version="<public-version>" />
    <PackageReference Include="JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge" Version="<public-version>" />
    <PackageReference Include="JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.CudaCudnn" Version="<public-version>" />
    <PackageReference Include="JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.TensorRt" Version="<public-version>" />
  </ItemGroup>
</Project>
```

如果使用 full runtime collection package，consumer 可能只引用 managed 包和 collection/runtime 包；如果使用 split components，Bridge、CudaCudnn、TensorRt 组件必须版本一致。collection package 只是引用集合，不能替代每个组件的 nupkg SHA256、native asset listing 和 runtime smoke 结果。

## 安装后检查什么

最低限度的用户自检可以分三步：

```powershell
dotnet --info
dotnet restore --force-evaluate
dotnet build -c Release
```

然后检查输出目录是否出现 managed assembly 与 native assets：

```text
JYPPX.TensorRtSharp.dll
JYPPX.CudaSharp.dll
jyppxtrtbridge.dll
nvinfer_10.dll / nvinfer.dll
nvonnxparser_10.dll / nvonnxparser.dll
cudart64_12.dll / cudart64_110.dll
cudnn64_9.dll / cudnn64_8.dll
```

这些检查能定位“包选错”“native assets 没复制”“PATH 污染”“driver/runtime 不兼容”等问题。它们仍只是 user troubleshooting evidence，不是 release proof。真正的 release proof 还需要 clean consumer、public package source、真实日志、SHA256、host metadata 和 strict validator。

建议安装后生成一份最小 native asset manifest：

```text
NativeAssetsCopied=true/false
BridgeAssetPresent=true/false
CudaCudnnAssetsPresent=true/false
TensorRtAssetsPresent=true/false
RuntimePackageKey=win-x64-trt10.11-cuda12.9-cudnn9.22
RestoreSourceMode=public-package-source
PackageReferenceOnly=true
UsesProjectReference=false
UsesLocalFeed=false
UsesDirectNupkg=false
DependencyProbeOnly=false
RuntimeSmokeAttempted=false
PackageConsumerRuntimeProof=false
```

这些字段适合进入 issue 或 owner input 草稿；其中 `RuntimeSmokeAttempted=false` 和 `PackageConsumerRuntimeProof=false` 要明确保留，直到真实 smoke 和 validator 通过。

## Clean consumer proof 的最低字段

如果要做 package-consumer-runtime proof，必须在仓库外干净目录执行，并记录：

```text
public package source URL
managed package id/version
runtime package id/version/runtime key
runtime package role 或整包 key
managed nupkg SHA256
runtime nupkg SHA256
clean consumer project path
restore log path + SHA256
build log path + SHA256
native asset listing
dependency probe log
runtime smoke log
exitCode = 0
OS / architecture / GPU / driver
CUDA / TensorRT / cuDNN metadata
owner name / machine name / review timestamp
strict validator result
```

常见 validator 路径包括：

```text
eng/Test-ExternalRuntimeProofRecord.ps1
eng/Test-PackageConsumerRuntimeProofRecord.ps1
eng/Test-PostPublishVerificationRecord.ps1
artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json
artifacts/final-release/package-consumer-runtime-proof-record-validation.md
artifacts/final-release/post-publish-verification-record.json
```

只有这些真实输入存在并通过 validator，才能把状态从 blocked-real-proof-required 推进。删除 forbidden substitutes、修改 dashboard、改测试断言或把模板填成占位值，都不能授权发布。

## 不能作为 proof 的情况

以下内容可以用于开发调试或文章说明，但不能作为 package-consumer-runtime proof：

- local feed package consumer。
- ProjectReference consumer。
- direct `.nupkg` install。
- GitHub Actions dry-run。
- GitHub full runtime collection package。
- owner execution package。
- package id/version template。
- release issue close record template。
- post-publish verification input draft。
- NuGet global packages cache 命中。
- `NUGET_PACKAGES` 改到 E 盘。
- PackageReference-only 但没有 runtime smoke。
- NativeAssetsCopied=true 但 dependency probe 失败。
- dependency probe passed 但没有 enqueue/output validation。
- build-only report。
- dependency-probe-only log。
- README、截图、GUI screenshot 或 command preview。

一句话边界：local feed、ProjectReference、direct `.nupkg`、dry-run、template、input draft、collection package、build-only、sidecar-only、precheck-only 和 blocked-by-cuda-driver 都不是 package-consumer-runtime proof。

## 常见错误与处理

`DllNotFoundException` 通常表示 bridge 或 vendor DLL 没被复制、RID/runtime key 选错，或输出目录缺少依赖链。先检查 package id，再检查 output directory，不要直接把系统目录塞进 PATH 来掩盖包问题。

`BadImageFormatException` 通常表示 x86/x64、Windows/Linux、TRT line 或 CUDA line 不匹配。确认项目目标是 `x64`，runtime key 是 `win-x64` 或 `linux-x64`，并重新 restore/build。

CUDA initialization failure 可能是 GPU driver 与 CUDA runtime 不兼容。它可以形成 blocked-by-cuda-driver 诊断，但不能被写成“包已通过 runtime proof”。

TensorRT engine 反序列化失败可能来自 TensorRT major/minor、plugin、lean runtime、version-compatible 或 engine 构建配置差异。不要把 runtime deserialization ownership、plugin lifecycle、allocator、callback、borrowed pointer 或 external resource 伪装成低风险安装问题。

## 配图建议

- managed package、bridge、CUDA/cuDNN、TensorRT 三层 native assets 依赖图。
- 整包 runtime package 与 split runtime package 的对照表。
- runtime key 选择流程图：RID -> TensorRT line -> CUDA line -> cuDNN major -> driver compatibility。
- clean external consumer 的 restore/build/native listing/dependency probe/runtime smoke/strict validator 流程图。

## 下一步

后续应把每个 runtime package key 的 CUDA/TensorRT/cuDNN 组合列成公开表格，并在 owner proof 输入中记录实际 GPU、driver、runtime package key、native asset listing、smoke 输出和 validator 结果。发布 owner 若要进入真实发布，还必须补齐 package-consumer-runtime、Linux runner proof、real-model-runtime、owner authorization 和 post-publish verification；在这些 proof 通过前，不能发布、不能关闭 release issue，也不能把安装教程当成发布完成声明。
