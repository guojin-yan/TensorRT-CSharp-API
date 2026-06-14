# Windows API 最新状态

截至 2026-06-12，本机扫描头文件的接口覆盖已经清零：

- TensorRT interface coverage：`0` missing rows。
- CUDA runtime interface coverage：`0` missing rows。
- Manifest inventory：`3271` 条 API records，`102` 份 manifests。

这意味着当前主线已经不再是继续追逐 coverage missing rows，而是发布加固：

- 保持 coverage matrix、build、tests、DocFX 全绿。
- 保持样例 runner 可运行、可跳过、可复现。
- 保持 asset-dependent 样例目录 README 化，而不是空壳。
- 验证 runtime package collection、runtime nupkg、package consumer restore/build/run。
- 只有在 ABI、ownership、version guard 和 C# 生命周期明确时，才把 deferred boundary 提升为安全 public wrapper。

## 当前样例状态

- `MultiStream`：真实 CUDA multi-stream / event-ordering 样例，已加入解决方案。
- `DynamicShape`：真实 TensorRT dynamic-shape / optimization profile / inference binding 样例，已加入解决方案。
- `Classification`、`OnnxToEngine`、`YoloDet`：现在都是用户侧常用的可执行 sample；其中 Classification/YoloDet 需要用户提供可再分发 ONNX 资产。
- `CustomKernelPreprocess`：保持 roadmap 状态，等待安全 public CUDA module/kernel wrapper。

## 当前发布门

推荐本地门禁：

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
dotnet build .\TensorRtSharp.sln -c Debug --no-restore
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build
dotnet docfx .\docs\docfx.json
```

如修改 manifest/native/generated，还需要运行 generator determinism 和 native build gate。

## Runtime 状态

- TensorRT 10 + CUDA 11.8 是当前稳定 real vendor-backed smoke path。
- TensorRT 10 + CUDA 12.9 和 TensorRT 11 + CUDA 12.9 已有本地 runtime/package 验证证据。
- TensorRT 11 + CUDA 13.2 bridge 可以构建，但当前机器 runtime/builder 创建仍等待 CUDA 13-capable driver/runtime 环境。
- Linux runtime package 结构已保留，但真实 Linux 发布证据需要 self-hosted Linux x64 runner。
