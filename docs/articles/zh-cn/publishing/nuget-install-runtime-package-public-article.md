# TensorRtSharp4.0 NuGet 安装与 Runtime 包选择

TensorRtSharp4.0 的安装体验不能只看一个 managed NuGet 包。真正能跑起来，还要匹配 CUDA、TensorRT、cuDNN、Windows/Linux runtime asset、GPU driver 和目标 .NET 版本。本文面向第一次集成项目的 C# 开发者，说明如何选择包、如何验证依赖、以及哪些内容不能被当成发布 proof。

## 适合谁阅读

- 准备在自己的 .NET 项目中引用 TensorRtSharp4.0 的用户。
- 需要区分 managed package 与 runtime package 的发布维护者。
- 遇到 DLL 加载、CUDA runtime、TensorRT 版本组合问题的工程师。

## 安装思路

推荐把安装拆成三层看：

1. managed API：C# wrapper、interop、工具类和高层对象。
2. runtime assets：按平台、CUDA、TensorRT、cuDNN 组合拆分的 native 文件。
3. owner proof：外部 consumer 使用公开包执行 smoke 后留下的记录。

前两层解决“项目能不能引用和加载”；第三层才解决“公开包是否在干净外部 consumer 中真实可用”。

## 基础命令

在业务项目中引用 managed 包：

```powershell
dotnet add package JYPPX.TensorRT.CSharp.API --prerelease
```

再按目标环境选择 runtime 包。实际包名和版本以发布清单为准，不要从本地 `nupkg` 或 local feed 截图推断公开可用性。

```powershell
dotnet restore
dotnet build -c Release
```

如果要做发布 proof，必须在干净外部 consumer 中执行：

```powershell
dotnet new console -n TensorRtSharpConsumerProof
cd TensorRtSharpConsumerProof
dotnet add package JYPPX.TensorRT.CSharp.API --version <public-version>
dotnet restore
dotnet build -c Release
```

## 不能作为 proof 的情况

以下内容可以用于开发调试，但不能作为 package-consumer-runtime proof：

- local feed。
- ProjectReference。
- direct `.nupkg`。
- template JSON。
- dry-run。
- build-only report。
- README 或截图。

package-consumer-runtime proof 必须来自公开包、干净外部 consumer、真实 host metadata、exitCode=0、native assets copied 和 strict validator。

## 配图建议

- managed package 与 runtime package 的依赖结构图。
- clean external consumer 的 restore/build/smoke 流程图。
- 常见 DLL 加载错误与对应 runtime 包选择表。

## 下一步

后续应把每个 runtime package key 的 CUDA/TensorRT/cuDNN 组合列成公开表格，并在 owner proof 输入中记录实际 GPU、driver、runtime package key 和 smoke 输出。
