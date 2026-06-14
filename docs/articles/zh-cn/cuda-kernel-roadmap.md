# CUDA Kernel Wrapper 路线图

自定义 CUDA 预处理是很常见的部署场景，但本仓库不应该发布一个要求用户直接调用 raw generated launch entry point 的示例。

当前托管 CUDA API 已经适合用于 device memory、pinned memory、stream、event、memory pool、graph、copy、fill 和 diagnostics 等部署基础能力。它还没有暴露一个安全的 public abstraction 来加载 CUDA module、打包 kernel arguments、启动任意 kernel，并管理 function/module 生命周期。

Raw CUDA launch entry point 目前只作为 generated/internal native 边界存在。不应该在 sample 中直接使用它们，因为参数打包、module 所有权、stream ordering 以及不同 CUDA toolkit 版本的 launch configuration 都需要更高层的 C# API 来封装。

当前可以先运行这个真实示例：

```powershell
dotnet run --project .\samples\MultiStream
```

它验证了未来 GPU preprocessing demo 会依赖的 CUDA memory 和 stream primitives。

## 计划中的 API 形态

- 增加安全的 `CudaModule` 和 `CudaKernel` wrapper，并明确所有权和释放语义。
- 增加 typed kernel argument packing，避免暴露 raw pointer arrays。
- 对 block/grid dimensions 和 shared memory 做 launch configuration validation。
- 对不同 toolkit line 中存在差异的 CUDA launch API 增加 version guards。
- 只有当 public API 足够安全后，再加入 normalization 或 NHWC-to-NCHW 这类可运行 preprocessing sample。
