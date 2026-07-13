# TensorRtSharp NuGet 包结构：Managed 包与 Native Runtime 包如何协作

## 适用读者

这篇文章适合准备安装 TensorRtSharp 4.0 的 .NET 用户、维护 NuGet 发布流程的 owner、以及需要理解 managed/native 包拆分策略的部署工程师。

## 解决问题

TensorRT/CUDA/cuDNN 的 native 组件体积大、版本多、平台差异明显。如果把所有 native 文件都塞进一个 managed 包，用户会下载大量无用资产，发布者也很难按 CUDA/TensorRT 组合维护。TensorRtSharp 4.0 采用 managed 包和 runtime package 分工：managed 包提供 C# API，runtime package 提供对应 RID 和 runtime key 的 native bridge/vendor assets。

## 包结构如何协作

使用时，项目通常需要两个维度：

- Managed API：`JYPPX.TensorRtSharp`、`JYPPX.CudaSharp`、`JYPPX.TensorRtSharp.Tools`。
- Native runtime：按 `win-x64-trt...-cuda...-cudnn...` 或 Linux runtime key 选择。

构建时，managed 包解析 API；运行时，native runtime package 负责把 DLL/shared object 放到可加载位置。项目中的 `eng/Test-PackageConsumer.ps1`、`artifacts/package-consumer` 和 `artifacts/final-release` 记录的是 package consumer 验证路径，但只有仓库外 clean consumer、public package source 和 runtime smoke 才能作为 release proof。

## 安装示例

```powershell
dotnet new console -n TensorRtSharpConsumer
cd TensorRtSharpConsumer
dotnet add package JYPPX.TensorRtSharp --version <version>
dotnet add package JYPPX.TensorRtSharp.Native.<runtime-key> --version <version>
dotnet restore
dotnet build -c Release
```

真实发布前，owner 还需要把 public package source、package id/version、runtime key、nupkg SHA256、restore/build/smoke 日志、host metadata 和 owner review 写入 proof input，并运行 strict validator。

## 用户应该如何选择 runtime package

先看 OS/RID，再看 TensorRT line，再看 CUDA/cuDNN，再看本机 driver。不要用“机器上有某个 CUDA 目录”替代 runtime package 选择，也不要用 local feed 或 direct `.nupkg` 代替 public source。如果需要写教程或公众号文章，建议把 runtime key 写成显式占位，避免读者误以为所有组合都已经验证。

## 边界说明

NuGet 安装说明不是 runtime proof。`build-only`、`dry-run`、`template`、local feed、ProjectReference、direct `.nupkg`、GUI screenshot、TensorRtExec report、YoloVision matrix、OnnxToEngine report、sidecar、readonly diagnostics 都不能替代 package-consumer-runtime proof。只有 public package source 上的真实 package 被仓库外 clean consumer restore/build/smoke，并通过 strict validator，才能提升 proof。

## 下一步

Windows 用户继续阅读 `windows-installation-and-troubleshooting-guide.md`；Linux/CI 用户继续阅读 `linux-installation-runner-boundary-guide.md`。发布 owner 则应参考 `package-consumer-runtime-proof-clean-consumer-guide.md`。
