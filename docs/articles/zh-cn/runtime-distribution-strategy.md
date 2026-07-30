# Bridge 包分发策略

TensorRtSharp 只发布 managed C# 包、项目自有 `.Bridge` 包和 tracked-files-only 源码归档。CUDA、cuDNN、TensorRT、NVRTC、parser、plugin 与 builder resource 都是用户自行安装的依赖，不得进入 NuGet 包或 GitHub Release 资产。

## 兼容键

每个 bridge 构建使用明确的部署组合：

- TensorRT major/minor，例如 `trt10.11`；
- CUDA major/minor，例如 `cuda12.9`；
- cuDNN major/minor，例如 `cudnn9.22`；
- RID；Linux 还要包含发行版版本。

示例：

```text
win-x64-trt10.11-cuda12.9-cudnn9.22
JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge
```

兼容键选择 bridge 编译使用的 header/import library，并声明 smoke 所需的机器依赖。它不授权把 NVIDIA 原厂库装入包内。

## Windows 矩阵

当前 Windows x64 bridge 目标：

- `win-x64-trt8.6-cuda11.8-cudnn8.9`；
- `win-x64-trt8.6-cuda12.1-cudnn8.9`；
- `win-x64-trt10.11-cuda11.8-cudnn8.9`；
- `win-x64-trt10.11-cuda12.9-cudnn9.22`；
- `win-x64-trt11.0-cuda12.9-cudnn9.22`；
- `win-x64-trt11.0-cuda13.2-cudnn9.22`。

编译成功只算 build evidence。每一行的 runtime proof 仍需要兼容 driver、机器安装的 TensorRT/CUDA/cuDNN、仓库外 consumer、enqueue/readback、日志和 hash。CUDA 12.9 主机不能关闭 CUDA 13.2 runtime 行。

## Linux 矩阵

Linux key 包含发行版和架构。当前 x64 模型覆盖 NVIDIA 仓库实际支持的 Ubuntu 20.04、22.04 与 24.04 组合。ARM64/SBSA、Jetson/L4T 和非 Ubuntu 发行版需要独立 bridge identity、runner/container、依赖来源和 runtime 证据。

`pack/runtime/linux-runtime-targets.manifest.json` 仍是目标目录。`runtime_key_set` 可以选择一组构建验证行，但每个公开 `.Bridge` 包仍有独立精确 key 和 proof row。

## 包内容

唯一可 pack 的 native role 是 `Bridge`：

- Windows：`runtimes/win-x64/native/jyppxtrtbridge.dll`；
- Linux：`runtimes/<linux-rid>/native/libjyppxtrtbridge.so`。

历史 `CudaCudnn`、`TensorRt`、`CudaRtc`、collection、meta 和 vendor-bundling identity 只保留用于清理与兼容审计，对应项目保持不可打包。`eng/Test-ExternalVendorRuntimePackagePolicy.ps1` 会在当前 pack/upload 路径拒绝这些 identity 和 NVIDIA binary。

## 公开通道

相同的 managed + bridge-only 边界通过两个通道提供：

1. NuGet-compatible source，使用标准 `PackageReference` restore。
2. GitHub Release `.nupkg` 资产，使用不可变 URL 和 GitHub SHA256 digest。

GitHub Release 不是 NuGet feed。`eng/Invoke-PublicReleaseBridgePackageConsumer.ps1` 会下载 managed/bridge 资产，验证远端 digest、包身份、nuspec repository URL/commit 和 bridge-only 内容，再把验证后的文件放入隔离 restore staging。direct `.nupkg` 或 DLL 引用不被接受。

managed 与 bridge 的 nuspec 必须指向正式仓库的同一源码提交。`-AllowCrossCommitPair` 只能产生 diagnostic-only 记录，不能晋级 public asset consumer evidence、package-consumer-runtime proof 或 post-publish proof。

## 构建流程

先解析并验证本机输入，再只打 bridge role：

```powershell
$roots = powershell -ExecutionPolicy Bypass -File .\eng\Resolve-RuntimeRoots.ps1 `
  -RuntimePackageKey win-x64-trt10.11-cuda12.9-cudnn9.22 | ConvertFrom-Json

powershell -ExecutionPolicy Bypass -File .\eng\Validate-WindowsRuntimeInputs.ps1 `
  -RuntimePackageKey win-x64-trt10.11-cuda12.9-cudnn9.22 `
  -TensorRtRoot $roots.tensorRtRoot `
  -CudaRoot $roots.cudaRoot `
  -CudnnRoot $roots.cudnnRoot

powershell -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 `
  -RuntimePackageKey win-x64-trt10.11-cuda12.9-cudnn9.22 `
  -SplitPackageRole bridge
```

`eng/Invoke-LocalRuntimePackage.ps1` 已退役并 fail closed，非 bridge split role 同样 fail closed。

正式发布只从 `guojin-yan/TensorRT-CSharp-API` 执行。`grape-yan` 仓库仅用于 Actions 编译验证，不含 package push 或 Release upload lane。当前 release workflow 只可发布：

- `JYPPX.TensorRT.CSharp.API`；
- 匹配的 `.Bridge` 包；
- tracked-files-only 源码归档。

## 历史清理

2026 年 6 月发布的 vendor-bearing GitHub Package 版本与对应 Release 资产，已在 2026-07-30 经过精确指纹确认后删除。历史 release tag 与 manifest identity 可能继续出现在审计记录中，但它们不是当前包来源，也不能重新发布。

## 证据边界

公开发布闭环需要目标矩阵行具备：

- Owner 授权的 managed、bridge 和源码发布；
- 公开 URL、package identity、version、size 与 SHA256；
- managed/bridge 同提交 provenance；
- 仓库外 restore/build/runtime smoke；
- driver、GPU、TensorRT、CUDA、cuDNN 与可选 NVRTC 主机 metadata；
- runtime stdout/stderr 与结构化报告 hash；
- post-publish clean consumer；
- strict validator 与最终 Owner 决策。

local pack、local feed、ProjectReference、direct `.nupkg`、dependency probe、build-only 输出、历史 vendor 包证据或绿色 dashboard 都不能替代这些记录。
