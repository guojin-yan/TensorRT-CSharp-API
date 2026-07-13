# Windows 源码编译：C++ Bridge 与 Native Runtime

本文面向希望自己从源码编译 TensorRtSharp4.0 native bridge 的用户。它不是 release proof，也不代表公开包已经发布；它的目标是让 Windows 用户能按同一套环境、命令和排错路径复现 C++/C# 之间的桥接构建。

## 适用场景

- 你需要调试 `native/` 下 TensorRT 或 CUDA C ABI bridge。
- 你希望重新生成 C# interop bindings，并确认 generated files 与 manifests 一致。
- 你选择 NuGet 小包路线：NuGet 只安装 C# core API 和 C++ bridge，CUDA/TensorRT/cuDNN 由本机自行安装。
- 你准备为 GitHub full runtime 包做本地构建验证，但尚未进行公开发布。

## 推荐环境

| 组件 | 建议 | 说明 |
| --- | --- | --- |
| Windows | Windows 10/11 x64 | 以 PowerShell 7 为主 |
| Visual Studio | 2022 + Desktop development with C++ | 需要 MSVC、Windows SDK、CMake tools |
| .NET | .NET 8 SDK | 用于 solution、samples、tests |
| CMake | 3.27+ | 以 repo preset 为准 |
| CUDA | 与目标 runtime key 匹配 | 例如 CUDA 12/13 lane |
| TensorRT | 与目标 TRT8/TRT10/TRT11 lane 匹配 | 不同 TensorRT line 不能混用头文件和 DLL |
| cuDNN | 目标 sample/runtime 需要时安装 | GitHub full runtime 包可携带，NuGet 小包路线要求用户自装 |

## 环境变量

优先让项目自己的 probing 逻辑寻找 `build-out`、`third_party/nvidia` 和标准安装路径：

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"
```

只有当你要固定到某套 SDK 时再显式覆盖：

```powershell
$env:JYPPX_TENSORRT_ROOT = "C:\nvidia\TensorRT-10.x"
$env:JYPPX_CUDA_ROOT = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"
$env:JYPPX_CUDNN_ROOT = "C:\nvidia\cudnn"
$env:JYPPX_NATIVE_BRIDGE_PATH = "E:\GitSpace\TensorRT-CSharp-API-4.0\TensorRtSharp4.0\build-out\..."
```

不要把不同 TensorRT major 的 include/lib/bin 混在同一个 shell 会话里。TRT8、TRT10、TRT11 的 manifests、native 实现和托管路由必须保持 version guard 一致。

## 构建顺序

从仓库根目录执行：

```powershell
Set-Location E:\GitSpace\TensorRT-CSharp-API-4.0\TensorRtSharp4.0

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
dotnet build .\TensorRtSharp.sln -c Debug --no-restore /p:UseSharedCompilation=false
```

如果要编译 TRT11/CUDA13 native bridge：

```powershell
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
```

其他 preset 以 `CMakePresets.json` 为准。不要手写临时 include/lib 路径绕过 preset；如果 preset 不覆盖你的环境，先补 preset 或环境探测，再构建。

## 生成与验证

构建完成后至少运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseEvidenceClassificationAudit.ps1 -Strict
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build
```

如果修改了 native manifest 或 bridge entrypoint，必须重新运行 bindings 生成与输出测试。只改 manifest 不改 native/source，或者只改 generated C# 不更新 generator，都属于不完整修改。

## 常见问题

### 找不到 CUDA/TensorRT DLL

先确认 `JYPPX_ENABLE_DEVELOPMENT_PROBING=1`，再检查目标 DLL 是否在预期 `bin` 目录。NuGet 小包路线不会携带大型 CUDA/TensorRT/cuDNN runtime，用户必须自行安装。

### CUDA error 35

通常是 driver/runtime 不兼容。先记录 GPU、driver、CUDA runtime、TensorRT line 和 Windows build，再按 `cuda-tensorrt-cudnn-version-matrix-guide.md` 排查。`blocked-by-cuda-driver` 不是 smoke passed，也不是 package-consumer-runtime proof。

### TensorRT line 混用

TRT8、TRT10、TRT11 的 header、library、runtime DLL 和 manifest 必须一致。跨版本只读 wrapper 可以共享高层语义，但 native ABI 入口和 version guard 不能模糊。

### 构建成功但不能发布

本地 build、local feed、ProjectReference、direct `.nupkg`、pre-publish smoke、dependency probe 和 failedBlockerCount=0 都不能替代真实公开包 proof。最终发布仍需要 Owner 提供 CleanConsumer、PostPublish、公开包 URL/SHA256、rollback review 和 release close approval。

## 下一步

完成 native bridge 编译后，继续阅读：

- `source-build-cmake-presets-and-bindings.md`
- `nuget-github-dual-package-strategy.md`
- `onnx-to-engine-trtexec-parity-roadmap.md`
- `tensorrtexec-console-winforms-application-roadmap.md`
