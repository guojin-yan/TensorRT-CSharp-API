# TensorRtSharp4.0

TensorRtSharp4.0 是面向生产部署的 TensorRT / CUDA .NET 桥接工程。

## 项目范围

- 托管程序集：`JYPPX.TensorRtSharp`、`JYPPX.CudaSharp`
- NuGet 主包：`JYPPX.TensorRT.CSharp.API`
- 原生桥接库：`jyppxtrtbridge`
- 首批发布目标：Windows x64 和 Linux x64
- TensorRT 版本线：8.x、10.x、11.x
- CUDA 版本线：11.x、12.x、13.x

## 当前验证状态

截至 2026-06-12：

- TensorRT interface coverage matrix：`0` 个 missing rows。
- CUDA runtime interface coverage matrix：`0` 个 missing rows。
- Manifest API inventory：`3271` 条 API 记录，分布在 `102` 份 manifests 中。
- 最新覆盖报告：`artifacts/interface-coverage/interface-coverage-summary.md`。
- 高版本原生验证：`win-x64-trt11-cuda13-release` 可配置并可编译。
- 托管验证：solution build 和 project quality tests 通过。
- DocFX 验证：文档构建为 `0` warning、`0` error。

当前仓库已经为本工作区扫描到的 TensorRT 8/10/11 和 CUDA 11/12/13 头文件建立 manifest/native-source 覆盖。一部分高风险或低频 CUDA runtime API 被明确记录为 deferred boundary，而不是包装成高层托管 API。典型 deferred 场景包括 callback 生命周期策略、裸 driver entrypoint 指针、external resource descriptor、IPC 所有权、CUDA library JIT option arrays、texture/surface descriptor、green/execution-context resource handle，以及 user-object destructor 所有权。

## 当前阶段

原始接口追平阶段已完成。当前主线是发布硬化：

- 持续保持覆盖矩阵、build、tests 和 DocFX 绿色。
- 保持 sample runners 准确、可复现。
- 对依赖外部资产的 sample 目录保留 README/roadmap，不放空壳目录。
- 验证 runtime-package asset collection、package consumer restore/build/run 路径。
- 只有在 ABI、所有权、版本保护和托管生命周期清晰时，才把 deferred boundary 晋升为高层 public API。

## 部署验证路径

本地 Windows 部署 sanity check 推荐顺序：

```powershell
dotnet restore .\TensorRtSharp.sln
powershell -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
dotnet build .\TensorRtSharp.sln -c Debug --no-restore
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build
dotnet docfx .\docs\docfx.json
```

如果修改了 manifest、native 或 generated 文件，还应运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
powershell -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
```

本轮还补跑了 TensorRT 8 兼容分支：

```powershell
cmake --build --preset win-x64-trt8-cuda11-release --parallel
cmake --preset win-x64-trt8-cuda12-release
cmake --build --preset win-x64-trt8-cuda12-release --parallel
```

推荐验证顺序：

1. `CudaSmokeRunner`
2. `TensorRtSmokeRunner`
3. `LifecycleSmokeRunner`
4. `OnnxToEngineSmokeRunner`
5. `NetworkBuilderSmokeRunner`
6. 各类 layer-specific network runners

常用用户示例请从 `samples/README.md` 进入，例如 `MultiStream`、`DynamicShape`、`InferenceBindings`、`OnnxToEngine`、`Classification` 和 `YoloDet`。

## Samples

可运行部署示例位于 `samples/`，入口清单见 `samples/README.md`。

近期 sample 成熟度状态：

- `MultiStream` 是真实 CUDA multi-stream/event ordering 示例，并已纳入 solution。
- `DynamicShape` 是真实 TensorRT dynamic-shape/profile/binding 示例，并已纳入 solution。
- `InferenceBindings` 是真实 TensorRT inference-binding 示例，并已纳入 solution。
- `OnnxToEngine` 现在是可运行的常用 ONNX-to-engine 示例，并已纳入 solution。
- `Classification` 和 `YoloDet` 是依赖用户自备 ONNX 模型、labels 和 input-shape metadata 的可运行示例。
- CUDA custom-kernel preprocessing 先保留为文档路线图，等待安全的公开 `CudaModule` / `CudaKernel` wrapper 后再加入 samples。

## Runtime Packages

runtime 包为一个明确 TensorRT / CUDA / cuDNN 组合承载原生部署资产。当前 Windows runtime package keys：

- `win-x64-trt8.6-cuda11.8-cudnn8.9`
- `win-x64-trt8.6-cuda12.1-cudnn8.9`
- `win-x64-trt10.11-cuda11.8-cudnn8.9`
- `win-x64-trt10.11-cuda12.9-cudnn9.22`
- `win-x64-trt11.0-cuda12.9-cudnn9.22`
- `win-x64-trt11.0-cuda13.2-cudnn9.22`

当前本地状态：

- TensorRT 10 + CUDA 11.8 是稳定的真实 vendor-backed smoke 路径，已有 package-consumer smoke 证据。
- TensorRT 10 + CUDA 12.9 和 TensorRT 11 + CUDA 12.9 已完成本地 runtime/package 验证和 package-consumer smoke。
- TensorRT 11 + CUDA 13.2 bridge 可编译、可收集 assets、可打包，并通过 package consumer restore/build/native-copy；runtime/builder smoke 仍等待 CUDA 13-capable driver/runtime 环境。
- Linux runtime 包名现在显式包含系统版本和架构。Ubuntu 22.04 x64 是默认 hosted 矩阵并覆盖 6 个组合；Ubuntu 24.04 x64 只覆盖 NVIDIA 官方仓库已提供的 TensorRT 10/11 现代组合；Ubuntu 20.04 x64 只走 self-hosted。

相关文档：

- `docs/articles/zh-cn/runtime-packages.md`
- `docs/articles/zh-cn/runtime-distribution-strategy.md`
- `docs/articles/zh-cn/package-consumer-validation.md`
- `docs/articles/zh-cn/release-candidate-gate.md`
- `docs/articles/zh-cn/api-reference.md`

## 发布自动化

这个仓库支持两种发布执行方式：

1. 用 `gh` 从 GitHub 远端触发工作流，再由本机的 self-hosted Windows runner 执行 Windows runtime 打包。
2. 直接运行本地脚本，做纯工作站上的验证闭环，不在 GitHub Actions 中留下运行记录。

也可以用 `act` 在本机做 workflow dry-run，例如解析 `release-bundle.yml` 或 `runtime-linux.yml` 的调度图。`act` 适合做轻量检查，但不能替代正式发布证据：Windows hosted job 不能被 Linux 容器可靠复刻，self-hosted runtime job 仍依赖真实本机/runner 上的 CUDA、cuDNN、TensorRT 和签名环境。详见 `docs/articles/zh-cn/local-actions.md`。

runtime 包现在和 managed 包独立版本。日常维护优先只发布 `JYPPX.TensorRT.CSharp.API` 到 nuget.org 和 GitHub Packages；CUDA/cuDNN/TensorRT 这类大组件保持在 GitHub Packages 或 GitHub Releases。每个 NVIDIA 依赖版本只发布一次 vendor 组件包；后续本地 C ABI bridge 变化时，只重发 `bridge,collection`。

截至 2026-06-17 的远端发布映射：

| Release tag | 内容 |
| --- | --- |
| `v4.0.6170` | 只有 managed 包：`JYPPX.TensorRT.CSharp.API.4.0.6170.nupkg`。 |
| `v4.0.6156` | Windows x64 runtime 矩阵：6 个 Windows TensorRT/CUDA/cuDNN 组合。 |
| `v4.0.6167` | Linux x64 Ubuntu 22.04 runtime 矩阵：6 个 hosted Ubuntu 22.04 组合。 |
| `v4.0.6169` | Linux x64 Ubuntu 24.04 runtime 矩阵：3 个 hosted Ubuntu 24.04 现代组合。 |

最新 managed release 不应该被理解为“包含全部 runtime asset”。需要核对完整 runtime 到 release tag 的对应关系时，看 `release-publication-audit.yml` 上传的 `artifacts/publication-index/runtime-publication-index.md`。

远端 managed-only 发布示例：

```powershell
gh workflow run release-bundle.yml `
  --ref TensorRtSharp4.0 `
  -f version=4.0.1 `
  -f publish_managed_to_nuget=true `
  -f publish_managed_to_github_packages=true `
  -f attach_runtime_to_github_release=true
```

远端首次发布或升级 CUDA/cuDNN/TensorRT 时刷新 vendor 组件示例：

```powershell
gh workflow run release-bundle.yml `
  --ref TensorRtSharp4.0 `
  -f version=4.0.0 `
  -f runtime_version=4.0.0 `
  -f run_windows_runtime_packaging=true `
  -f windows_runtime_keys=win-x64-trt11.0-cuda12.9-cudnn9.22 `
  -f windows_runtime_delivery_mode=split `
  -f windows_split_package_roles=cuda-cudnn,tensorrt `
  -f publish_runtime_to_github_packages=true `
  -f attach_runtime_to_github_release=true
```

远端本地封装代码变化后刷新 bridge 和 collection 示例：

```powershell
gh workflow run release-bundle.yml `
  --ref TensorRtSharp4.0 `
  -f version=4.0.1 `
  -f runtime_version=4.0.1 `
  -f run_windows_runtime_packaging=true `
  -f windows_runtime_keys=win-x64-trt11.0-cuda12.9-cudnn9.22 `
  -f windows_runtime_delivery_mode=split `
  -f windows_split_package_roles=bridge,collection `
  -f windows_cuda_cudnn_package_version=4.0.6156 `
  -f windows_cuda_cudnn_package_release_tag=v4.0.6156 `
  -f windows_tensorrt_package_version=4.0.6156 `
  -f windows_tensorrt_package_release_tag=v4.0.6156 `
  -f publish_managed_to_github_packages=true `
  -f publish_runtime_to_github_packages=false `
  -f attach_runtime_to_github_release=true
```

本地 managed-only 示例：

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Invoke-LocalReleaseBundle.ps1 `
  -Version 4.0.1 `
  -SkipWindowsRuntime
```

本地刷新 bridge 和 collection 示例：

```powershell
gh release download v4.0.6156 `
  --pattern "JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda12.9.cudnn9.22.*.4.0.6156.nupkg" `
  --dir .\artifacts\stable-runtime-package-source\win-x64-trt11.0-cuda12.9-cudnn9.22 `
  --repo guojin-yan/TensorRT-CSharp-API

powershell -ExecutionPolicy Bypass -File .\eng\Invoke-LocalReleaseBundle.ps1 `
  -Version 4.0.1 `
  -RuntimeVersion 4.0.1 `
  -WindowsRuntimeKeys win-x64-trt11.0-cuda12.9-cudnn9.22 `
  -WindowsRuntimeDeliveryMode split `
  -WindowsSplitPackageRoles bridge,collection `
  -WindowsCudaCudnnPackageVersion 4.0.6156 `
  -WindowsTensorRtPackageVersion 4.0.6156 `
  -WindowsAdditionalPackageSource .\artifacts\stable-runtime-package-source\win-x64-trt11.0-cuda12.9-cudnn9.22
```

`release-bundle.yml` 默认不再触发 runtime 打包。需要 runtime 时显式设置 `run_windows_runtime_packaging=true` 或 `run_linux_runtime_packaging=true`；如果启用 Linux runtime 但 `linux_runtime_keys` 为空，Linux 模块会干净 no-op。

发布到 `nuget.org` 时，仓库 secret `NUGET_API_KEY` 应填写 NuGet 官网生成的纯文本 ASCII API key。managed-package workflow 会在发布前校验该 secret；不要把加密后的本机凭据或机器导出的 token 片段填进 `NUGET_API_KEY`。

## 仓库布局

```text
build/      CMake modules and build helpers
docs/       DocFX site and conceptual documentation
eng/        automation and dependency discovery scripts
native/     C ABI bridge and TensorRT/CUDA adapters
pack/       NuGet packaging projects
samples/    user-facing common examples and documented sample roadmaps
smoke/      validation runners for release gates, packaging, and regression checks
src/        managed libraries
tests/      managed integration and unit tests
third_party/local dependency drop folder (not committed)
```

## 构建前置条件

- .NET SDK 10.0.300 或更高版本。
- CMake 3.27 或更高版本。
- Windows 上需要 Visual Studio C++ toolchain。
- native build 和 runtime-package validation 需要匹配的本地 TensorRT / CUDA / cuDNN roots。

## 快速开始

```powershell
dotnet restore .\TensorRtSharp.sln
dotnet build .\TensorRtSharp.sln -c Debug --no-restore
powershell -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build
```

构建当前高版本 native preset：

```powershell
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
```

## 依赖发现

使用以下脚本检查本地 TensorRT/CUDA/cuDNN roots：

- `eng/Get-Dependencies.ps1`
- `eng/get-dependencies.sh`

Windows 本地 root 不写入公开 runtime manifest。请使用 `pack/runtime/runtime-packages.local.json` 保存机器本地覆盖配置；可从 `pack/runtime/runtime-packages.local.example.json` 复制后修改。

托管 runtime loading 面向生产部署：

- 常规探测检查 app base directory 和 `runtimes/<rid>/native`。
- 显式桥接库路径使用 `JYPPX_NATIVE_BRIDGE_PATH`。
- 显式 vendor roots 使用 `JYPPX_TENSORRT_ROOT` 和 `JYPPX_CUDA_ROOT`。
- 本地 `build-out` / `third_party` 开发扫描需要 `JYPPX_ENABLE_DEVELOPMENT_PROBING=1`。

## 文档入口

- `docs/index.md`
- `docs/articles/zh-cn/getting-started.md`
- `docs/articles/zh-cn/installation-layout.md`
- `docs/articles/zh-cn/api-coverage-and-deferred-boundaries.md`
- `docs/articles/zh-cn/sample-runners.md`
- `docs/articles/zh-cn/runtime-packages.md`
- `docs/articles/zh-cn/package-consumer-validation.md`
- `docs/articles/zh-cn/release-candidate-gate.md`

构建文档：

```powershell
dotnet docfx .\docs\docfx.json
```

## 注意事项

- NVIDIA 二进制文件不提交到仓库。
- `third_party/` 仅作为本地依赖投放目录。
- runtime 包计划在完成再分发许可和包体积复核后，才随匹配 TensorRT、CUDA、cuDNN 动态库一起交付。
- 手写 public C# wrapper 应包含有用的 XML documentation；生成 API 可以使用生成注释。
