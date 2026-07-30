# TensorRtSharp NuGet 包结构：Managed 与 Bridge 如何协作

TensorRtSharp4.0 当前只发布 managed C# 包和项目自有 `.Bridge` 包。CUDA、cuDNN、TensorRT 以及可选 NVRTC 由用户自行安装；任何 NVIDIA DLL/`.so` 都不会进入 TensorRtSharp nupkg 或 GitHub Release asset。

## 适用读者

本文适合准备安装 TensorRtSharp4.0 的 .NET 用户、维护 NuGet 发布流程的 owner，以及需要理解 managed/bridge/host dependency 边界的部署工程师。

## 解决问题

managed 包提供 C# API，但不能单独运行 native TensorRT；bridge 包提供稳定的项目 C ABI 边界，但不负责分发第三方 runtime。将这两个包与用户安装的 NVIDIA stack 分开，可避免重复下载、许可证混淆和旧 vendor binary 长期滞留。

## 包结构如何协作

- Managed API：`JYPPX.TensorRT.CSharp.API`，包含 `JYPPX.TensorRtSharp`、`JYPPX.CudaSharp` 等托管程序集。
- Native bridge：`JYPPX.TensorRT.CSharp.API.Runtime.<runtime-key>.Bridge`，只包含 `jyppxtrtbridge.dll` 或 `libjyppxtrtbridge.so`。
- Host dependencies：用户安装匹配的 TensorRT、CUDA、cuDNN、parser/plugin/builder resource 与可选 NVRTC。

runtime key 记录 bridge 编译 ABI 和期望主机版本，例如 `win-x64-trt11.0-cuda12.9-cudnn9.22`。它不是 vendor bundle identity。

## 安装示例

```powershell
dotnet new console -n TensorRtSharpConsumer
cd TensorRtSharpConsumer
dotnet add package JYPPX.TensorRT.CSharp.API --version <version> --source <approved-source>
dotnet add package JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda12.9.cudnn9.22.Bridge --version <version> --source <approved-source>
dotnet restore --force-evaluate
dotnet build -c Release
```

运行前先按 NVIDIA 官方方式安装匹配版本，并确认系统 loader 能解析它们。不要从历史 GitHub Release 或旧 package 恢复 vendor DLL。

## 用户应该如何选择 Bridge package

先看 OS/RID，再看 TensorRT line、CUDA line、cuDNN major 和 GPU driver。package id/version/source 以 owner 的公开发布清单为准。managed 与 bridge 必须来自同一源码提交；跨提交组合只能用于 diagnostic-only 回放。

## 公开来源

GitHub Release 通道会校验 managed/bridge asset 的 immutable URL、digest、SHA256、nuspec repository commit，再建立 verified staging。NuGet-compatible source 通道直接按 PackageReference restore 两个包。两条通道都必须经过仓库外 clean consumer。

## 边界说明

NuGet 安装说明不是 runtime proof。`build-only`、`dry-run`、`template`、local feed、ProjectReference、direct `.nupkg`、GUI screenshot、TensorRtExec report、YoloVision matrix、OnnxToEngine report、sidecar、readonly diagnostics 都不能替代 package-consumer-runtime proof。

只有公开来源的 managed + bridge 包被仓库外 clean consumer restore/build/smoke，实际加载用户安装的 NVIDIA runtime，并通过 strict validator，结果才可能晋级。

## 下一步

Windows 用户继续阅读 `windows-installation-and-troubleshooting-guide.md`；Linux/CI 用户继续阅读 `linux-installation-runner-boundary-guide.md`；发布 owner 参考 `package-consumer-runtime-proof-clean-consumer-guide.md`。
