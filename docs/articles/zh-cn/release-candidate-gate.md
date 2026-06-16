# 发布候选质量门禁

发布候选门禁用于确认当前包不是“能编译就算完成”，而是具备可复现的消费端证据链。

## 当前基线

截至 2026-06-12：

- TensorRT interface coverage：`0` missing rows。
- CUDA runtime interface coverage：`0` missing rows。
- Manifest inventory：`3271` 条 API records，`102` 份 manifests。
- Deferred API 是明确的 manifest/native 边界，不能当作安全 public wrapper 宣传。

当前 package-consumer 证据（2026-06-12）：

- `win-x64-trt10.11-cuda11.8-cudnn8.9`：restore/build/native-copy/smoke 通过，native assets 为 `16/16`，探针输出 TensorRT `10.11.0`、CUDA `11.8`。
- `win-x64-trt10.11-cuda12.9-cudnn9.22`：restore/build/native-copy/smoke 通过，native asset patterns 为 `19/19`，探针输出 TensorRT `10.11.0`、CUDA `12.9`。
- `win-x64-trt11.0-cuda12.9-cudnn9.22`：restore/build/native-copy/smoke 通过，native asset patterns 为 `19/19`，探针输出 TensorRT `11.0.0`、CUDA `12.9`。
- `win-x64-trt11.0-cuda13.2-cudnn9.22`：2026-06-14 已完成完整 split 组件包与 collection 包本地打包，restore/build/native-copy 通过，native asset patterns 为 `19/19`；但 package consumer smoke 仍 pending，CUDA 13 runtime/builder 验证完成前 publish readiness 必须 blocked。

## 本地质量门

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
dotnet restore .\TensorRtSharp.sln
dotnet build .\TensorRtSharp.sln -c Debug --no-restore
powershell -ExecutionPolicy Bypass -File .\eng\Test-PublicApiBilingualDocumentation.ps1
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build
dotnet docfx .\docs\docfx.json
```

如果修改了 native、manifest 或 generated 文件，还需要运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
powershell -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
```

期望结果：

- TensorRT missing rows 保持 `0`。
- CUDA runtime missing rows 保持 `0`。
- `dotnet build` 为 `0` warning、`0` error。
- 公开 API XML 注释同时包含英文和中文。
- `JYPPX.ProjectQuality.Tests` 通过。
- DocFX 为 `0` warning、`0` error。

## 推荐 smoke 顺序

1. `CudaSmokeRunner`
2. `MultiStream`
3. `TensorRtSmokeRunner`
4. `LifecycleSmokeRunner`
5. `OnnxToEngineSmokeRunner`
6. `DynamicShape`
7. `InferenceBindings`
8. `NetworkBuilderSmokeRunner`
9. 各类 layer-specific network runners

以下 asset-dependent sample project 是可执行项目，但需要用户提供模型与 metadata：

- `Classification`
- `YoloDet`

这些项目只要 README 明确说明所需外部资产和可运行替代路径，就不单独作为 release blocker。CUDA custom-kernel preprocessing 先保留为文档路线图，等待安全 public `CudaModule` / `CudaKernel` wrapper 后再加入可运行 sample。

## Runtime package 门禁

发布候选涉及 runtime package 时，必须提供：

- 精确 TensorRT、CUDA、cuDNN root 校验。
- 匹配 CMake preset 输出的 runtime asset collection。
- managed package 与 matching runtime package pack 证据。
- package consumer restore/build/native asset copy 验证。
- 兼容机器上的 package consumer smoke 证据；如果被 WDAC、driver/runtime 不兼容阻塞，应记录为环境 blocker，而不是 package layout failure。
- private-feed 或 split-delivery readiness 必须要求 `local-validated`；`pending-local-validation` 不能视为 ready。

CUDA `12.9` 和 TensorRT 11 Windows 组合必须使用精确 CUDA/TensorRT/cuDNN 依赖链证据。`trt11.0-cuda13.2-cudnn9.22` 可以构建，但当前机器 runtime/builder 创建仍依赖 CUDA 13-capable driver/runtime 环境。

Linux Ubuntu 22.04 x64 是当前 hosted 发布主线，远程 workflow 必须完成 build、asset collection、pack、consumer validation 和 release/package 上传后才算发布证据。Ubuntu 20.04 x64 仍然需要 `runner_mode=self-hosted` 或手动 root；Ubuntu 24.04 x64 只覆盖现代组合；ARM/Jetson 目标需要单独建包线。

## 远端发布前置条件

在启用远端发布链前，请先确认：

- `package-managed.yml` 在 `publish_to_nuget=true` 时，如果仓库 secret `NUGET_API_KEY` 存在，就会使用该值，并要求它是纯文本 ASCII 的 nuget.org API key。也可以不设置该 secret，让 self-hosted Windows runner 使用当前用户的 NuGet 配置兜底；兜底逻辑会校验 `nuget.org`、`https://api.nuget.org/v3/index.json`、`https://www.nuget.org` 等常见 nuget.org alias，然后让 NuGet 或 `nuget.exe` 直接读取原始用户配置。不要填加密后的本机凭据或其它机器导出的 token 片段。
- `runtime-windows.yml` 要求 Windows self-hosted runner 在线，并带有 `self-hosted`、`windows`、`x64` 标签。
- `runtime-linux.yml` 可以通过 GitHub-hosted runner 发布 Ubuntu 22.04 x64 默认矩阵。Ubuntu 20.04 x64 必须使用 `runner_mode=self-hosted`，Ubuntu 24.04 x64 只覆盖现代组合，ARM/Jetson 目标需要单独建包线后才能发布。
