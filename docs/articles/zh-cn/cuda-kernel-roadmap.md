# CUDA Kernel Wrapper 路线图

自定义 CUDA 预处理是很常见的部署场景，但本仓库不应该发布一个要求用户直接调用 raw generated launch entry point 的示例。

当前托管 CUDA API 已经适合用于 device memory、pinned memory、stream、event、memory pool、graph、copy、fill 和 diagnostics 等部署基础能力。CUDA 12.9+ 已提供 owner-safe `CudaKernelLibrary.Load(byte[])`、复制型 inventory、按名称查询和 typed `CudaKernelLibrary.Launch(...)`；`CudaDriverModule.Load(...)` / `Launch(...)` 则提供动态加载的统一 Driver module owner，不暴露 `CUmodule` 或 `CUfunction`。

Raw CUDA launch entry point 目前只作为 generated/internal native 边界存在。不应该在 sample 中直接使用它们，因为参数打包、module 所有权、stream ordering 以及不同 CUDA toolkit 版本的 launch configuration 都需要更高层的 C# API 来封装。

从 CUDA C++ 源码编译到 PTX/CUBIN/LTO IR、再到 owner-safe 加载和启动的完整设计，见 [CUDA Runtime Compilation（NVRTC）接入路线图](cuda-runtime-compilation-roadmap.md)。

当前可以先运行这个真实示例：

```powershell
dotnet run --project .\samples\Performance\01.MultiStream
```

它验证了未来 GPU preprocessing demo 会依赖的 CUDA memory 和 stream primitives。

## 已实现与计划中的 API 形态

- 已为 CUDA 12.9+ Runtime-library 路径和 `CudaDriverModule` 路径实现按名称 owner-bound launch、typed kernel argument packing，以及 block/grid/shared-memory validation。
- `driver.cpp` 已分别通过 CUDA 11.8/12.1/12.9/13.2 header 编译，本机 Driver 12090 已完成 11.8/12.1/12.9 PTX 的 launch/readback。
- 对不同 toolkit line 中存在差异的 CUDA launch API 增加 version guards。
- 只有当 public API 足够安全后，再加入 normalization 或 NHWC-to-NCHW 这类可运行 preprocessing sample。
