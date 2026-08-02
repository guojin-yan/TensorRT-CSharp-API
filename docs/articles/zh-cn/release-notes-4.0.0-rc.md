# TensorRtSharp 4.0.0 RC 发布说明草案

TensorRtSharp 4.0.0 RC 是第一版公开发布候选。本版优先保证稳定的 C# 接口、主要 TensorRT/CUDA 工作流、核心演示和可重复打包；其余接口与完整运行时矩阵将在保持公开接口兼容的前提下持续补强。

## 第一版范围

- C# 主包 `JYPPX.TensorRT.CSharp.API`，包含 managed wrapper，不包含厂商 native runtime。
- YoloVision 扩展包 `JYPPX.TensorRT.CSharp.API.YoloVision`，提供无裸指针泄漏的配置、预处理、后处理、报告和命令入口。
- 按 TensorRT/CUDA 组合编译的 `.Bridge` 包；每个包只允许包含项目自有的 `jyppxtrtbridge.dll` 或 `libjyppxtrtbridge.so`。
- Git 跟踪的源码、演示和文档。源码归档不得混入 CUDA、cuDNN、TensorRT 或 NVRTC 二进制。
- CUDA、cuDNN、TensorRT、NVRTC 和兼容驱动由用户自行安装，并与所选 bridge key 对齐。

## 本轮重点

- TensorRT/CUDA manifest 与 generated binding 保持零 missing 基线；deferred 项仍按真实可用性记录，不能冒充已完成能力。
- Plugin registry、runtime diagnostics、ONNX parser/refitter、CUDA graph、memory range 等主要接口已具备 managed 入口。
- DebugListener callback 默认不启用未经证明的 non-null attach，不安装虚构 vtable，也不伪造 invocation。
- managed 与 YoloVision `4.0.0` 已完成本地 pack、内容策略、公共 API 表面和仓库外纯 `PackageReference` 消费者 dry run。
- 第一版 runtime workflow 只运行 manifest、managed、bridge、vendor policy、YoloVision 和 release automation 的核心契约测试；当前 480 类完整 ProjectQuality 矩阵不是本版发布前置条件。

## 候选交付物

- `JYPPX.TensorRT.CSharp.API 4.0.0`
- `JYPPX.TensorRT.CSharp.API.YoloVision 4.0.0`
- `JYPPX.TensorRT.CSharp.API.Runtime.<rid>.<trt>.<cuda>.<cudnn>.Bridge 4.0.0`
- GitHub 自动生成的源码归档和仓库文档

旧的 full-runtime、CUDA/cuDNN、TensorRT component 与 collection/meta 包均已退休，不属于 4.0.0 发布集合，也不得继续上传。

## 已知限制

- 用户主机必须自行提供与 bridge key 兼容的 NVIDIA driver、CUDA、cuDNN、TensorRT 和按需使用的 NVRTC。
- `IDebugListener::processDebugTensor` 等 callback 能力仍需兼容 GPU 主机上的真实运行证明；缺少该证明不会阻塞本版已经验证的主要接口。
- callback、裸指针、外部资源 ownership 和跨 ABI trampoline 继续按明确的 deferred 边界迭代。
- 本地构建和 dry run 只用于检查候选包是否可构建、可检查、可消费；local feed 仅供本地开发。
- 公开渠道发布与 post-publish proof 当前均未发生，必须在 Owner 授权发布后另行核验。

## 验证入口

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalVendorRuntimePackagePolicy.ps1 -StaticOnly

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionManagedPackageDryRun.ps1 `
  -PackageDirectory .\artifacts\managed `
  -PackageVersion 4.0.0 `
  -Configuration Release

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 `
  -SourceRuntimeKey win-x64-trt11.0-cuda13.2-cudnn9.22 `
  -Version 4.0.0 `
  -SplitPackageRole bridge `
  -Configuration Release
```

公开发布仍需 Owner 对目标版本、包哈希和渠道执行最终确认；本说明本身不构成发布授权。
