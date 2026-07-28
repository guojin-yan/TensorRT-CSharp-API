# CUDA Kernel Wrapper 路线图

自定义 CUDA 预处理是很常见的部署场景，但本仓库不应该发布一个要求用户直接调用 raw generated launch entry point 的示例。

当前托管 CUDA API 已经适合用于 device memory、pinned memory、stream、event、memory pool、graph、copy、fill 和 diagnostics 等部署基础能力。CUDA 12.9+ 还已提供 owner-safe `CudaKernelLibrary.Load(byte[])`、复制型 inventory 和按名称查询，但尚未提供 public owner-bound kernel launch，也没有覆盖 CUDA 11.8/12.1 的统一 module owner。

Raw CUDA launch entry point 目前只作为 generated/internal native 边界存在。不应该在 sample 中直接使用它们，因为参数打包、module 所有权、stream ordering 以及不同 CUDA toolkit 版本的 launch configuration 都需要更高层的 C# API 来封装。

从 CUDA C++ 源码编译到 PTX/CUBIN/LTO IR、再到 owner-safe 加载和启动的完整设计，见 [CUDA Runtime Compilation（NVRTC）接入路线图](cuda-runtime-compilation-roadmap.md)。

当前可以先运行这个真实示例：

```powershell
dotnet run --project .\samples\MultiStream
```

它验证了未来 GPU preprocessing demo 会依赖的 CUDA memory 和 stream primitives。

## 计划中的 API 形态

- 扩展 `CudaKernelLibrary` 的按名称 owner-bound launch，并为旧 Toolkit 审计统一 `CudaModule` / `CudaKernel` owner 的必要性。
- 增加 typed kernel argument packing，避免暴露 raw pointer arrays。
- 对 block/grid dimensions 和 shared memory 做 launch configuration validation。
- 对不同 toolkit line 中存在差异的 CUDA launch API 增加 version guards。
- 只有当 public API 足够安全后，再加入 normalization 或 NHWC-to-NCHW 这类可运行 preprocessing sample。
