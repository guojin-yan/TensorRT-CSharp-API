# Windows 本地开发环境准备

本文面向需要在 Windows x64 上构建、验证或打包 TensorRtSharp4.0 的开发者。普通 C# 消费端只需要 NuGet 包和匹配 runtime package；维护者则需要 .NET、CMake、CUDA、TensorRT、cuDNN 以及本地 runtime root 配置。

## 基础工具

建议准备：

| 工具 | 用途 |
| --- | --- |
| .NET SDK `10.0.300` 或更高 | 构建 solution、运行 tests、生成 DocFX。 |
| PowerShell 7 或 Windows PowerShell | 执行 `eng/*.ps1` 脚本。 |
| CMake `3.27` 或更高 | 配置和构建 native bridge。 |
| Visual Studio Build Tools | 提供 MSVC、Windows SDK 和 CMake generator。 |
| NVIDIA driver | 运行 CUDA/TensorRT smoke。 |
| CUDA Toolkit | 构建和运行 CUDA 相关 bridge/runtime 包。 |
| TensorRT SDK | 提供 TensorRT headers、libs、DLLs。 |
| cuDNN | runtime package 收集依赖。 |

只写文档或只改托管层时，不一定需要完整 NVIDIA runtime root。但只要要跑 native build、runtime packaging 或 package consumer smoke，就必须准备匹配依赖。

## Runtime root 配置

公开 manifest 不记录本机绝对路径。本地 root 应写入：

```text
pack/runtime/runtime-packages.local.json
```

可从示例文件复制：

```text
pack/runtime/runtime-packages.local.example.json
```

本地文件已被 Git 忽略，适合填写当前机器上的 TensorRT、CUDA、cuDNN 根目录。

## 常用质量门

修改文档或托管代码后，常用验证为：

```powershell
dotnet build .\TensorRtSharp.sln -c Debug --no-restore /p:UseSharedCompilation=false
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build /p:UseSharedCompilation=false
dotnet docfx .\docs\docfx.json
```

修改 manifest、native 或 generated 文件后，再追加：

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
powershell -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
```

修改 runtime package 相关脚本后，应运行对应 readiness 或 package consumer 脚本，而不是只看 solution build。

## Runtime 输入校验

打包前可以先解析并校验 root：

```powershell
$roots = powershell -ExecutionPolicy Bypass -File .\eng\Resolve-RuntimeRoots.ps1 `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 | ConvertFrom-Json

powershell -ExecutionPolicy Bypass -File .\eng\Validate-WindowsRuntimeInputs.ps1 `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 `
  -TensorRtRoot $roots.tensorRtRoot `
  -CudaRoot $roots.cudaRoot `
  -CudnnRoot $roots.cudnnRoot
```

这一步能提前发现缺 DLL、只有 import library、目录不匹配等问题。

## CMake preset

原生桥接库按 preset 隔离输出。常用 TRT11/CUDA13 release preset：

```powershell
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
```

runtime asset collection 会按 manifest 中配置的 build preset 收集 bridge 输出，因此不要手工从其它目录复制 DLL 来替代 preset 构建结果。

## 当前 CUDA 13.2 注意事项

`win-x64-trt11.0-cuda13.2-cudnn9.22` 当前已经完成 split/full package readiness，full package consumer restore/build/native-copy 通过，readiness blockers 为 `0`。

但本机 full package consumer runtime smoke 在 `cudaRuntimeGetVersion` 处返回 CUDA error 35，记录为：

```text
SmokeResult=blocked-by-cuda-driver
RealCallbackRuntimeEvidence.Status=blocked-by-cuda-driver
IsRealCallbackRuntimeProof=False
```

这表示当前机器 driver/runtime 组合不能完成 CUDA 13.2 runtime smoke。它不是 package layout failure，也不是 API 缺失，更不是 callback proof。

## 常见准备问题

| 现象 | 优先检查 |
| --- | --- |
| CMake 找不到 headers 或 libs | runtime local root 是否指向正确 TensorRT/CUDA/cuDNN 版本。 |
| native-copy 数量不足 | `eng/Collect-RuntimeAssets.ps1` 和 runtime manifest 的 expected assets。 |
| smoke 提示 DLL missing | 输出目录是否复制 native assets，PATH 是否被污染。 |
| `0x800711C7` | Windows Defender Application Control，可尝试 package consumer 的 `-SignConsumerOutput`。 |
| CUDA error 35 | NVIDIA driver 是否支持当前 CUDA runtime。 |

## 下一步阅读

- [NuGet 消费端验证全流程](nuget-package-consumer-validation-flow.md)
- [Package Readiness 当前状态](package-readiness-current-state.md)
- [常见问题排查总表](troubleshooting-index.md)
