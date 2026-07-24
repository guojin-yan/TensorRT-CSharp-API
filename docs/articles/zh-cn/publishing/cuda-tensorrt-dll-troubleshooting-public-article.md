# CUDA / TensorRT DLL 加载问题排查

在 Windows 上使用 TensorRT C# binding，最常见的问题不是 C# 语法，而是 native 依赖加载：CUDA driver、CUDA runtime、TensorRT DLL、cuDNN、Visual C++ runtime、PATH、当前工作目录、进程位数、RID 和 runtime package 都必须匹配。本文给出面向用户的排查路径，帮助把“DLL 找不到”变成可定位、可复现、可反馈的问题。

这篇文章是 troubleshooting guide，不是 package-consumer-runtime proof。它可以帮助用户缩小问题范围，但不能授权发布、不能关闭 release issue，也不能替代 clean external consumer 的 runtime smoke 和 strict validator。

## 适合谁阅读

- 遇到 `DllNotFoundException`、`BadImageFormatException`、CUDA error 35 或 TensorRT native load 失败的新用户。
- 需要维护 runtime package、native assets copied 和 dependency probe 的发布负责人。
- 正在排查 `jyppxtrtbridge.dll`、`nvinfer_10.dll`、`nvonnxparser_10.dll`、`cudart64_12.dll`、`cudnn64_9.dll` 等依赖链的工程师。
- 准备采集 clean external consumer proof，但还没有真实 owner log/hash/host metadata 的 owner。

## 先确认版本组合

排查前记录这些字段：

```text
dotnet --info
nvidia-smi
OS / architecture
GPU 型号
NVIDIA driver 版本
CUDA driver/runtime 版本
TensorRT line/version
cuDNN major/version
目标 RID，例如 win-x64 或 linux-x64
managed package id/version
runtime package id/version/runtime key
native asset listing
stdout/stderr log SHA256
```

这些字段后续也会进入 owner proof 输入，不能只靠截图或口头描述。若 driver 支持的 CUDA runtime 低于 runtime package 需求，常见表现是 CUDA error 35、CUDA driver/runtime mismatch 或 native initialization failed；这只能形成 blocked-by-cuda-driver 诊断，不能写成 runtime proof。

## 代码与文档入口

排查时优先看这些入口：

```text
src/JYPPX.Shared/Interop/NativeBridgePathResolver.cs
src/JYPPX.Shared/Interop/NativeBridgeLibraryLoader.cs
src/JYPPX.Shared/BridgeConstants.cs
src/JYPPX.CudaSharp/CudaEnvironmentProbe.cs
src/JYPPX.TensorRtSharp/TensorRtEnvironmentProbe.cs
src/JYPPX.TensorRtSharp.Tools/TensorRtToolSupport.cs
src/JYPPX.TensorRtSharp.Tools/OnnxEngineBuildService.cs
pack/runtime/runtime-packages.manifest.json
pack/runtime-split/split-runtime-packages.manifest.json
docs/articles/zh-cn/runtime-package-native-load-troubleshooting.md
docs/articles/zh-cn/windows-installation-and-troubleshooting-guide.md
docs/articles/zh-cn/cuda-error-35-troubleshooting.md
docs/articles/zh-cn/runtime-package-minimal-smoke-commands.md
docs/articles/zh-cn/package-consumer-runtime-proof-clean-consumer-guide.md
```

`NativeBridgePathResolver` 决定 bridge 候选路径和进程搜索路径，`NativeBridgeLibraryLoader` 通过 `NativeLibrary.SetDllImportResolver` 与 `NativeLibrary.TryLoad` 加载 bridge。`BridgeConstants.NativeBridgeLibraryName` 固定为 `jyppxtrtbridge`。如果 bridge 本身能找到，但它依赖的 TensorRT/CUDA/cuDNN DLL 找不到，错误仍可能表现为 bridge load failure。

## 决策树

第一步：确认 .NET 与进程位数。

```powershell
dotnet --info
[Environment]::Is64BitProcess
[Runtime.InteropServices.RuntimeInformation]::RuntimeIdentifier
```

如果进程不是 x64，先修正项目平台和 RID。TensorRT/CUDA Windows runtime 基本按 x64 交付，x86 进程会导致 `BadImageFormatException` 或无法加载 native dependency。

第二步：确认 driver 与 CUDA。

```powershell
nvidia-smi
Get-Command nvcc -ErrorAction SilentlyContinue
```

`nvcc` 是否存在不是 runtime package proof；很多用户只装 driver 不装 Toolkit。真正要确认的是 driver 是否支持 runtime package 需要的 CUDA runtime。CUDA error 35 代表 driver/runtime 组合不兼容时，应该换兼容 host 或 runtime key，而不是修改测试或删除 blocker。

第三步：确认 NuGet runtime package。

```powershell
dotnet list package
dotnet restore --force-evaluate
dotnet build -c Release
```

核对 `pack/runtime/runtime-packages.manifest.json` 或 `pack/runtime-split/split-runtime-packages.manifest.json` 中的：

```text
key
packageId
rid
tensorRtLine
tensorRtVersion
cudaLine
cudaVersion
cudnnMajor
cudnnVersion
bridgeFile
tensorRtFiles
cudaFiles
cudnnFiles
role = bridge
role = cuda-cudnn
role = tensorrt
```

第四步：确认输出目录。

```powershell
Get-ChildItem .\bin\Release\net8.0 -Filter *.dll | Sort-Object Name
Get-ChildItem .\bin\Debug\net8.0 -Filter *.dll | Sort-Object Name
```

常见文件包括：

```text
JYPPX.TensorRtSharp.dll
JYPPX.CudaSharp.dll
JYPPX.TensorRtSharp.Tools.dll
jyppxtrtbridge.dll
nvinfer.dll / nvinfer_10.dll
nvinfer_plugin.dll / nvinfer_plugin_10.dll
nvonnxparser.dll / nvonnxparser_10.dll
cudart64_110.dll / cudart64_12.dll
cudnn64_8.dll / cudnn64_9.dll
```

缺少 bridge 时先看 Bridge package；缺少 TensorRT DLL 时看 TensorRt package；缺少 CUDA/cuDNN DLL 时看 CudaCudnn package。不要把系统目录复制到输出目录来掩盖包内容问题，除非只是临时诊断且记录清楚。

第五步：确认搜索路径。

```powershell
$env:PATH -split ';'
[System.IO.Directory]::GetCurrentDirectory()
```

Windows 搜索路径容易被旧 TensorRT/CUDA 目录污染。若 PATH 中有多个 TensorRT/CUDA/cuDNN 版本，优先移除无关项或启动干净 shell 复测。文章和 issue 中应记录实际 PATH 摘要，但不要上传包含密钥或私人目录的完整日志。

## Windows native loader 检查表

如果前五步仍不能定位问题，继续用 Windows 原生命令把 bridge 和 vendor DLL 分开看：

```powershell
where jyppxtrtbridge.dll
where nvinfer.dll
where nvinfer_10.dll
where nvinfer_plugin.dll
where nvonnxparser.dll
where cudart64_12.dll
where cudnn64_9.dll
Get-Command jyppxtrtbridge.dll -ErrorAction SilentlyContinue
dumpbin /dependents .\\bin\\Release\\net8.0\\jyppxtrtbridge.dll
```

`where` 和 `Get-Command` 只能说明 PATH 上能看到什么；`dumpbin /dependents` 只能说明 bridge import table 需要什么。真正的加载结果还取决于应用输出目录、runtime package `runtimes/<rid>/native`、当前进程 PATH、Visual C++ runtime 和 Windows loader 缓存。因此 issue 中建议同时记录：

```text
NativeBridgePathResolver candidate paths
NativeBridgeLibraryLoader load result
NativeBridgeLoadException message
NativeDependencyProbeStatus
ResolvedBridgePath
ResolvedVendorDllDirectory
ProcessArchitecture
RuntimeIdentifier
VC++ runtime installed
```

`NativeBridgePathResolver` 和 `NativeBridgeLibraryLoader` 的日志应该区分两种失败：

- bridge DLL 自身找不到：通常是 Bridge package/RID/output copy 问题。
- bridge DLL 找到了但 vendor dependency 找不到：通常是 TensorRT/CUDA/cuDNN runtime package、PATH 或 driver/toolkit 组合问题。

如果只是为了临时确认缺哪一个 DLL，可以把 vendor DLL 放进应用输出目录复测；但这种动作必须标记为 `temporary-local-diagnostic-copy`。它不能作为 package content proof，也不能写成 clean external consumer proof。发布前仍要回到 package restore 后的 native asset copy 结果。

## PATH 污染与版本漂移

很多“本机可以、用户不行”的问题来自 PATH 污染。典型情况是：

```text
PATH 里同时有 TensorRT-8、TensorRT-10、TensorRT-11
CUDA_PATH 指向 12.9，但 PATH 先命中 CUDA 11.8 bin
输出目录里是 trt10 runtime package，但 PATH 里先加载 trt11 DLL
cuDNN 8 和 cuDNN 9 DLL 混在同一目录
Visual C++ runtime 缺失或版本过旧
```

建议启动一个干净 shell，只保留目标 runtime key 需要的路径，再复测 dependency probe。若干净 shell 通过、日常 shell 失败，结论应写成 `path-contamination`，不是 package bug，也不是 runtime proof。

第六步：跑最小 probe，再跑模型。

先跑 help、environment probe、dependency probe 或最小 smoke，确认 native load 能完成，再进入 OnnxToEngine、TensorRtExec 或 YoloVision。复杂模型失败可能是 shape、plugin、engine serialization 或 postprocess 问题，不一定是 DLL load 问题。

## 常见错误

### 找不到 `jyppxtrtbridge.dll`

通常是 runtime package 没安装、RID 不匹配、输出目录缺少 Bridge assets，或项目使用了 ProjectReference/local build 但没有复制 native bridge。先看 `dotnet list package` 和输出目录，再看 `NativeBridgePathResolver` 的候选路径。

### 找不到 `nvinfer_10.dll` / `nvonnxparser_10.dll`

通常是 TensorRT runtime assets 没复制、TensorRt split package 未引用、整包 runtime key 选错，或 PATH 中先加载了不兼容的 TensorRT 主版本。TRT8、TRT10、TRT11 的 ABI 和 DLL 命名不同，不要混在同一输出目录。

### 找不到 `cudart64_12.dll` / `cudnn64_9.dll`

通常是 CUDA/cuDNN runtime assets 缺失、CudaCudnn split package 未引用，或 cuDNN 8/9 选错。CUDA Toolkit 安装目录不能替代 runtime package；它只能帮助本机开发和诊断。

### `BadImageFormatException`

通常表示 x86/x64、Windows/Linux、TRT line 或 CUDA line 不匹配。确认项目目标是 x64，runtime key 是 `win-x64` 或 `linux-x64`，并重新 restore/build。

### CUDA error 35

通常表示 NVIDIA driver 太旧，不能支持目标 CUDA runtime。它应该记录为 blocked-by-cuda-driver，并换兼容 host 复测。不能把这个 blocker 改写成 package-consumer-runtime proof。

### 本地可跑但外部 consumer 不可跑

这类问题最容易被 local feed、ProjectReference 或 direct `.nupkg` 掩盖。真实 package-consumer-runtime proof 必须使用公开包、干净外部项目、真实 package source、restore/build/runtime smoke log、hash、host metadata 和 strict validator。

## 推荐采集包

给 issue 或 owner proof input 提供材料时，建议收集：

```text
dotnet --info 输出
nvidia-smi 输出
dotnet list package 输出
restore/build log
输出目录 native asset listing
dependency probe log
runtime smoke log
stdout/stderr SHA256
runtime package key
managed/runtime package SHA256
OS / architecture / GPU / driver / CUDA / TensorRT / cuDNN metadata
```

若用于 release proof，还必须通过：

```text
eng/Test-ExternalRuntimeProofRecord.ps1
eng/Test-PackageConsumerRuntimeProofRecord.ps1
eng/Test-PostPublishVerificationRecord.ps1
```

对普通 issue，推荐同时提供一份“最小可复现记录”：

```text
MinimalReproProjectOutsideRepository = true/false
PackageReferenceOnly = true/false
UsesLocalFeed = true/false
UsesProjectReference = true/false
UsesDirectNupkg = true/false
NativeAssetsCopied = true/false
DependencyProbeOnly = true/false
RuntimeSmokeAttempted = true/false
RuntimeSmokePassed = true/false
DriverBlocked = true/false
PathContaminationSuspected = true/false
TemporaryLocalDiagnosticCopyUsed = true/false
```

这些字段能让 maintainer 快速判断：这是安装问题、PATH 问题、runtime package 内容问题，还是 owner proof 仍缺真实 clean consumer smoke。

## 不能作为 proof 的材料

以下材料可以帮助排查，但不是 package-consumer-runtime proof：

- “我机器上能 build”。
- 本地 `.nupkg`。
- local feed。
- ProjectReference。
- direct `.nupkg` install。
- GitHub Actions dry-run。
- dependency-probe-only log。
- build-only 的 TensorRtExec report。
- OnnxToEngine report。
- YoloVision matrix。
- sidecar-only metadata。
- GUI screenshot 或 command preview。
- 没有 hash 的日志片段。
- blocked-by-cuda-driver 诊断。
- `temporary-local-diagnostic-copy`。
- `path-contamination` 清理后通过。
- `NativeBridgePathResolver` 候选路径日志。
- `dumpbin /dependents` 输出。
- `where nvinfer.dll` / `where cudnn64_9.dll` 输出。
- dependency probe passed 但 runtime smoke 未执行。

也不要把 runtime deserialization ownership、plugin lifecycle、callback、allocator、borrowed pointer 或 external resource 伪装成 DLL 加载问题；这些属于更高风险的 owner/lifecycle 设计边界。

## 配图建议

- DLL 搜索路径示意图：application output -> NuGet runtime assets -> process PATH -> system directory。
- managed package / bridge / TensorRT / CUDA / cuDNN 的分层图。
- 常见错误信息到排查动作的表格。
- clean external consumer 的 restore/build/native listing/dependency probe/runtime smoke/strict validator 流程图。

## 下一步

后续应把每个 runtime package key 的依赖 DLL 列成机器可读清单，并让 clean consumer validator 检查 native assets copied、dependency probe status 和 smoke status。若目标是发布，owner 还需要补齐 package-consumer-runtime、Linux runner proof、real-model-runtime、owner authorization 和 post-publish verification；在这些 proof 通过前，排查文章不能授权发布，也不能关闭 release issue。
