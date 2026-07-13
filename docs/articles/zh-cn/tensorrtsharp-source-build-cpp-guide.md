# TensorRtSharp C++ 原生桥接源码编译总教程

本文面向希望自己从源码编译 TensorRtSharp4.0 原生 C++ bridge 的用户。它把环境准备、CMake preset、绑定生成、包验证和发布边界放在同一条路径里，适合作为公众号、博客和仓库文档的“从源码构建”入口。

> 证据边界：本文是源码编译教程，不是 release proof、post-publish proof 或 package-consumer-runtime proof。local build、local feed、ProjectReference、direct `.nupkg`、dependency probe、build-only report 和截图都不能替代真实 Owner proof。

## 你会编译出什么

TensorRtSharp4.0 的运行时由三层组成：

| 层级 | 产物 | 说明 |
| --- | --- | --- |
| C# managed API | `JYPPX.TensorRtSharp.dll`、`JYPPX.CudaSharp.dll`、`JYPPX.Shared.dll` | 面向 .NET 用户的高层 wrapper |
| C++ bridge | `jyppxtrtbridge.dll` 或对应平台共享库 | no-throw C ABI，隔离 TensorRT/CUDA C++ ABI 与托管 P/Invoke |
| NVIDIA runtime | TensorRT、CUDA、cuDNN DLL/shared objects | 由用户本机安装，或由 GitHub full runtime 包承载 |

源码编译主要验证第二层：C++ bridge 是否能按目标 TensorRT/CUDA/cuDNN 组合构建，并和托管绑定保持一致。

## 环境需求

Windows x64 开发机建议准备：

- Windows 10/11 x64。
- Visual Studio 2022，安装 “Desktop development with C++”。
- CMake 3.27 或更高版本。
- .NET SDK，版本以 `global.json` 为准。
- Git。
- NVIDIA CUDA Toolkit，与目标 runtime key 匹配。
- NVIDIA TensorRT SDK，与目标 TensorRT line 匹配。
- NVIDIA cuDNN，与目标 runtime key 匹配。

常见本机路径示例：

```powershell
$env:JYPPX_TENSORRT_ROOT = "C:\nvidia\TensorRT-10.x"
$env:JYPPX_CUDA_ROOT = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"
$env:JYPPX_CUDNN_ROOT = "C:\nvidia\cudnn-windows-x86_64-9.x"
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"
```

如果使用仓库内 `third_party\nvidia` 或 runtime manifest 解析路径，可以先不设置 override 变量，让脚本按默认规则探测。只有当你明确要指定某一套本机 SDK 时，再设置这些变量。

## 先确认 runtime key

源码编译前先明确目标组合。Windows 典型 key 包括：

- `win-x64-trt8.6-cuda11.8-cudnn8.9`
- `win-x64-trt8.6-cuda12.1-cudnn8.9`
- `win-x64-trt10.11-cuda12.9-cudnn9.22`
- `win-x64-trt11.0-cuda12.9-cudnn9.22`
- `win-x64-trt11.0-cuda13.2-cudnn9.22`

查看当前仓库建模的 runtime 包：

```powershell
Get-Content .\pack\runtime\runtime-packages.manifest.json
Get-Content .\CMakePresets.json
```

不要用“机器上有 CUDA 目录”代替 runtime key。TensorRT、CUDA、cuDNN、Visual Studio toolset 和 CMake preset 必须一起匹配。

## 生成绑定

修改 manifest、native header、native source 或 generated interop 前后，都应重新生成绑定并验证输出：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
```

如果你只改文档、sample README 或 C# wrapper，不一定需要重新生成绑定；但涉及 C ABI shape、manifest 参数、entrypoint 名称、version guard 时必须执行。

## 编译 C++ bridge

以 TensorRT 11 + CUDA 13.2 为例：

```powershell
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
```

常用 preset 还包括：

```powershell
cmake --preset win-x64-trt10-cuda12-release
cmake --build --preset win-x64-trt10-cuda12-release --parallel

cmake --preset win-x64-trt8-cuda12-release
cmake --build --preset win-x64-trt8-cuda12-release --parallel
```

实际 preset 名称以 `CMakePresets.json` 为准。构建输出通常位于：

```text
build-out/<preset>/bin/<Configuration>/
build-out/<preset>/lib/<Configuration>/
```

## 编译托管层与质量测试

C++ bridge 成功后，继续验证托管层：

```powershell
dotnet build .\TensorRtSharp.sln -c Debug --no-restore /p:UseSharedCompilation=false
dotnet build .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-restore /p:UseSharedCompilation=false
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build
```

如果只是做某个小批次，可先跑 focused tests，但合入前仍建议跑完整质量门。

## 本地打包与消费验证

源码编译不是发布。要验证本地包消费路径，可以先生成 managed 包：

```powershell
dotnet pack .\pack\JYPPX.TensorRT.CSharp.API\JYPPX.TensorRT.CSharp.API.csproj `
  -c Debug `
  -o .\artifacts\managed `
  -p:JYPPXPackageVersion=4.0.0 `
  /p:UseSharedCompilation=false
```

然后运行 bridge package consumer：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-BridgePackageConsumer.ps1
```

这个验证证明 package layout、managed API surface、bridge asset copy 和 dependency diagnostics 可用；它默认仍是 `compile-surface-proof` 或 bridge-only consumer proof，不是 clean public package runtime proof。

## 两条发布路线如何衔接

源码编译完成后，发布路线分两类：

| 路线 | 包含内容 | 适合用户 | 边界 |
| --- | --- | --- | --- |
| GitHub full runtime 包 | C# API、C++ bridge、CUDA/TensorRT/cuDNN 大依赖、runtime assets | 希望下载完整组合包的用户 | GitHub Release asset 不是 NuGet feed；仍需 hash、license 和 owner proof |
| NuGet small core/bridge 包 | C# core API、小体积 C++ bridge 包 | 已自行安装 TensorRT/CUDA/cuDNN 的用户 | 用户负责 NVIDIA runtime 安装与版本匹配 |

具体策略见 `docs/articles/zh-cn/nuget-github-dual-package-strategy.md` 和 `docs/articles/zh-cn/tensorrtsharp-nuget-runtime-package-guide.md`。

## 常见错误

### 找不到 TensorRT / CUDA / cuDNN

先确认路径变量和 SDK 版本：

```powershell
$env:JYPPX_TENSORRT_ROOT
$env:JYPPX_CUDA_ROOT
$env:JYPPX_CUDNN_ROOT
```

再确认 runtime key 对应的 SDK 是否真的存在。不要混用 TensorRT 8 的 lib 与 TensorRT 11 的 headers。

### CMake configure 通过但 build 失败

通常是 include/lib 组合不一致、Visual Studio C++ workload 缺失、CUDA toolkit 版本不匹配，或 preset 指向了错误 runtime line。先用一个明确 preset 复现，再检查 `CMakePresets.json` 和 `pack/runtime/runtime-packages.manifest.json`。

### 托管测试通过但 native 运行失败

托管 build/test 只能证明 C# 编译和质量门。native runtime 还需要 bridge DLL、NVIDIA DLL、PATH/probing、GPU driver 与 runtime 兼容。`blocked-by-cuda-driver` 是环境兼容阻塞，不是 API 完成证明。

### 想把本地包当作发布证明

不可以。local feed、direct `.nupkg`、ProjectReference、build-only report、dependency probe、GUI screenshot 和 readonly diagnostics 都不是 public release proof。public proof 必须来自真实公开包源、仓库外 clean consumer、runtime smoke、hash、stdout/stderr、host metadata 和 Owner review。

## 推荐阅读顺序

1. `docs/articles/zh-cn/source-build-windows-cpp-bridge.md`
2. `docs/articles/zh-cn/source-build-cmake-presets-and-bindings.md`
3. `docs/articles/zh-cn/tensorrtsharp-source-build-cpp-guide.md`
4. `docs/articles/zh-cn/nuget-github-dual-package-strategy.md`
5. `docs/articles/zh-cn/tensorrtsharp-nuget-runtime-package-guide.md`
6. `docs/articles/zh-cn/runtime-package-installation-deep-dive.md`

这条路径适合写成一组连续技术文章：先让用户理解为什么需要 C++ bridge，再让用户按 preset 编译，最后解释如何在 NuGet 小包和 GitHub full runtime 包之间选择。
