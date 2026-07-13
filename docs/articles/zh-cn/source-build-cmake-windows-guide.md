# TensorRtSharp4.0 Windows 源码编译教程：C++ Bridge、CUDA、TensorRT 与 cuDNN

这篇文章面向想自己从源码编译 TensorRtSharp4.0 的用户，重点覆盖 C++ bridge 的本地编译环境、CMake preset、依赖版本、验证命令和常见排障。它不是只给维护者看的内部清单，而是可以发布到公众号、博客或项目官网的完整教程。

## 适用读者

- 想从源码编译 `JYPPX.TensorRT.CSharp.API` 的 .NET 开发者。
- 需要自定义 TensorRT、CUDA、cuDNN 组合的部署工程师。
- 需要验证 GitHub 全依赖包和 NuGet 小包路线差异的发布负责人。
- 需要调试 native bridge、P/Invoke、DLL 加载或 CUDA driver 兼容问题的维护者。

## 编译产物

源码编译主要产生三类产物：

1. managed C# assemblies：`JYPPX.TensorRtSharp`、`JYPPX.CudaSharp`、`JYPPX.Shared`。
2. native C ABI bridge：项目 C++ 层生成的 TensorRT/CUDA bridge DLL。
3. runtime/package 验证资产：按 TensorRT/CUDA/cuDNN 组合收集、拆分或打包的依赖文件。

当前项目采用双发布路线：

- GitHub 全依赖包：包含 managed API、C++ bridge、TensorRT/CUDA/cuDNN runtime assets，适合开箱部署和完整示例。
- NuGet 小包：只发布 C# 核心 API 与中间 C++ bridge 小包，用户自行安装 CUDA、TensorRT、cuDNN，适合长期维护和公开分发。

## 推荐环境

最低 CMake 版本要求是 CMake 3.27；如果本机安装了更老版本，请先升级再执行 `cmake --preset`。

Windows 本地编译推荐准备：

| 组件 | 推荐版本 | 说明 |
|---|---:|---|
| Windows | Windows 10/11 x64 | 推荐启用长路径。 |
| .NET SDK | .NET 8 SDK 或当前仓库 `global.json` 指定版本 | 用于 managed build、tests、samples。 |
| Visual Studio | Visual Studio 2022 | 需安装 Desktop development with C++。 |
| MSVC | VS 2022 x64 toolchain | CMake Windows preset 默认使用 `Visual Studio 17 2022`。 |
| CMake | 3.27+ | `CMakePresets.json` 要求 3.27。 |
| CUDA Toolkit | 11.8、12.1/12.9、13.2 | 与 preset 中 `JYPPX_CUDA_VERSION` 对齐。 |
| TensorRT | 8.x、10.11、11.x | 与 `JYPPX_TENSORRT_LINE` 和 CUDA 版本对齐。 |
| cuDNN | 8.9 或 9.x | 与 CUDA/TensorRT 组合对齐。 |
| PowerShell | PowerShell 7 或 Windows PowerShell 5.1 | 运行 `eng/*.ps1` 脚本。 |

## 版本组合

仓库当前 Windows CMake presets 覆盖以下组合：

| Preset | TensorRT Line | CUDA Line | CUDA Version | cuDNN Major |
|---|---:|---:|---:|---:|
| `win-x64-trt8-cuda11-release` | 8 | 11 | 11.8 | 8 |
| `win-x64-trt8-cuda12-release` | 8 | 12 | 12.1 | 8 |
| `win-x64-trt10-cuda11-release` | 10 | 11 | 11.8 | 8 |
| `win-x64-trt10-cuda12-release` | 10 | 12 | 12.9 | 9 |
| `win-x64-trt11-cuda12-release` | 11 | 12 | 12.9 | 9 |
| `win-x64-trt11-cuda13-release` | 11 | 13 | 13.2 | 9 |

如果只是验证 managed 项目和 wrapper API，可以先跑 .NET build/tests；如果修改 native manifest、generated binding 或 C++ 实现，再执行 CMake preset。

## 目录准备

建议保持 NVIDIA 官方安装目录，减少环境变量配置成本：

```powershell
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
```

TensorRT 和 cuDNN 可以放在固定工具目录，例如：

```powershell
E:\NVIDIA\TensorRT-11.x
E:\NVIDIA\TensorRT-10.11
E:\NVIDIA\TensorRT-8.6
E:\NVIDIA\cuDNN-9.x
E:\NVIDIA\cuDNN-8.9
```

如果你的目录不同，请优先检查项目 `build/`、`cmake/` 和 `eng/` 下的依赖探测脚本，或在 CMake configure 时显式传入对应根目录变量。

## 第一步：恢复和生成托管绑定

在仓库源码目录执行：

```powershell
cd E:\GitSpace\TensorRT-CSharp-API-4.0\TensorRtSharp4.0
dotnet restore .\TensorRtSharp.sln
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
```

这一步验证 manifest、entry point、P/Invoke 生成物和 managed interop 是否一致。只改 C# wrapper 时通常不需要重新生成；改 native manifest 或 generated source 时必须运行。

## 第二步：构建 managed 项目

```powershell
dotnet build .\TensorRtSharp.sln -c Debug --no-restore
dotnet build .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-restore /m:1 /nodeReuse:false /p:UseSharedCompilation=false /p:BuildInParallel=false /v:minimal
```

如果这里失败，先不要进入 CMake。常见原因包括：

- 新增 public API 没有同步 XML 注释或测试断言。
- generated P/Invoke 签名与 C# wrapper 期望不一致。
- 使用了错误属性名或旧类型名，例如 interface metadata 应读取 `TensorRtInterfaceInfo.Kind`。
- samples 或 applications 引用了旧项目名；当前 live sample path 应使用 `samples/YoloVision`。

## 第三步：配置 CMake

以 TensorRT 11 + CUDA 13.2 为例：

```powershell
cmake --preset win-x64-trt11-cuda13-release
```

CMake 会在 `build-out/win-x64-trt11-cuda13-release` 下生成构建目录。该 preset 使用：

- `Visual Studio 17 2022`
- x64 architecture
- `JYPPX_ENABLE_TENSORRT_BINDINGS=ON`
- `JYPPX_ENABLE_CUDA_BINDINGS=ON`
- `JYPPX_TENSORRT_LINE=11`
- `JYPPX_CUDA_LINE=13`
- `JYPPX_CUDA_VERSION=13.2`
- `JYPPX_CUDNN_MAJOR=9`

如果要验证 TensorRT 10 + CUDA 12.9：

```powershell
cmake --preset win-x64-trt10-cuda12-release
```

## 第四步：构建 C++ bridge

```powershell
cmake --build --preset win-x64-trt11-cuda13-release --parallel
```

构建成功后，检查输出目录中是否存在 native bridge DLL。随后再运行 .NET smoke 或 package-consumer 验证。

## 第五步：运行质量门

推荐至少运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build
```

如果只改某个 batch，可以先跑 targeted tests，例如：

```powershell
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~Cuda|FullyQualifiedName~Onnx|FullyQualifiedName~ErrorRecorder" --logger "trx;LogFileName=targeted-native-wrapper-batch.trx" --results-directory .\artifacts\test-results\targeted /p:UseSharedCompilation=false /nr:false /v:minimal
```

## 第六步：运行 smoke

Smoke 不是替代 package-consumer proof 的最终证据，但能快速验证 wrapper 可调用面：

```powershell
dotnet run --project .\smoke\OnnxToEngineSmokeRunner\OnnxToEngineSmokeRunner.csproj -c Debug
dotnet run --project .\smoke\CallbackAllocatorSafeControlsSmokeRunner\CallbackAllocatorSafeControlsSmokeRunner.csproj -c Debug
dotnet run --project .\smoke\CudaSmokeRunner\CudaSmokeRunner.csproj -c Debug
```

关注输出中的关键 marker：

- `ParserDiagnosticSnapshot=`
- `ParserDiagnosticSummary=`
- `ParserUsedVCPluginLibraries Count=`
- `LayerOutputIdentity=`
- `RuntimeDiagnosticSnapshot=`
- `RuntimeDiagnosticSummary=`
- `ErrorRecorderSummary=`
- `AtomicCapabilities`

如果出现 `Skipped=True Reason=...`，它只说明当前机器缺少某项运行条件，不能直接晋级为 runtime proof。

## 常见问题排查

### CMake 找不到 CUDA

检查：

```powershell
where nvcc
nvcc --version
$env:CUDA_PATH
```

如果安装了多个 CUDA 版本，确认 `CUDA_PATH` 与 CMake preset 的 `JYPPX_CUDA_VERSION` 一致。TensorRT 11 + CUDA 13 preset 需要 CUDA 13.2 headers/libraries，否则 CUDA 13-only API 会被 version guard 屏蔽或 configure 失败。

### 找不到 TensorRT 头文件或库

确认 TensorRT 解压目录中存在：

```text
include\NvInfer.h
include\NvOnnxParser.h
lib\nvinfer*.lib
lib\nvonnxparser*.lib
```

如果使用自定义目录，请在 configure 前设置项目支持的 TensorRT root 变量，或检查 `build/`、`cmake/` 脚本里的 root 探测逻辑。

### 找不到 cuDNN

确认 cuDNN 与 CUDA major 版本匹配。CUDA 12.9 / 13.2 组合通常使用 cuDNN 9 系；CUDA 11.8 / TensorRT 8 组合通常使用 cuDNN 8.9。

### DLL 加载失败

优先检查：

```powershell
$env:PATH
dumpbin /dependents path\to\your\bridge.dll
```

常见原因：

- CUDA runtime DLL 不在 PATH。
- TensorRT DLL 不在 PATH。
- cuDNN DLL 不在 PATH。
- x86/x64 架构不一致。
- NuGet 小包路线下用户没有本机安装 CUDA/TensorRT/cuDNN。

### CUDA error 35

CUDA error 35 通常表示 driver/runtime 不兼容。处理顺序：

1. 用 `nvidia-smi` 查看 driver。
2. 用 `nvcc --version` 查看 toolkit。
3. 确认运行的 runtime package 与 driver 支持的 CUDA major 对齐。
4. CUDA 13 smoke 需要 CUDA 13-capable driver；否则只能证明 build/package，不要声明 runtime proof。

### 生成绑定后测试失败

如果 `Generate-Bindings.ps1` 后出现测试失败，重点查：

- manifest 中 `managedType` 是否正确。
- native source 是否真的实现对应 entry point。
- generated P/Invoke 是否使用 `SafeTensorRtObjectHandle`、caller buffer、`out` 或 pinned array。
- public wrapper 是否仍暴露 `IntPtr` / `nint`。
- deferred 记录是否被错误删除。

## 推荐开发顺序

单个 deferred batch 建议按这个顺序推进：

1. 用覆盖矩阵筛选只读/查询型候选。
2. 查 manifest 是否已有 non-deferred safe alternative。
3. 查 native source 是否真实调用 vendor API 或明确返回 dependency missing。
4. 查 generated P/Invoke 是否与 native 参数一致。
5. 补 managed interop 和 high-level wrapper。
6. 补 smoke marker。
7. 补 project-quality tests。
8. 跑 build、targeted tests、必要时跑 CMake。
9. 写入 diary 和下一阶段提示词。

## 发布前检查

发布前至少确认：

- managed build 通过。
- project-quality tests 通过或有明确 owner 输入项。
- native bridge 使用目标 preset 成功 configure/build。
- GitHub 全依赖包与 NuGet 小包路线都能解释清楚。
- `samples/YoloVision`、`samples/OnnxToEngine`、`applications/TensorRtExec` 的文档不再回流旧命名。
- 技术文章不是碎片式 API 文档，而是有背景、环境、步骤、代码、验证和排障的完整文章。

## 下一步

完成源码编译后，建议继续阅读：

- `onnx-to-engine-trtexec-conversion-guide.md`
- `tensorrtexec-tool-getting-started.md`
- `yolo-vision-model-matrix.md`
- `cuda-tensorrt-cudnn-version-matrix-guide.md`
- `tensorrtsharp-nuget-runtime-package-guide.md`
