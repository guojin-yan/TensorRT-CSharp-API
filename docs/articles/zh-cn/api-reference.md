# API Reference

当前托管 API 参考由 DocFX 基于 `src/` 下的 C# 项目自动生成。

主要入口：

- [JYPPX.TensorRtSharp](../../api/JYPPX.TensorRtSharp.yml)
- [JYPPX.CudaSharp](../../api/JYPPX.CudaSharp.yml)
- [JYPPX.TensorRtSharp.Shared](../../api/JYPPX.TensorRtSharp.Shared.yml)

公共 C# 命名空间只使用两个产品根：TensorRT 及共享桥接类型位于 `JYPPX.TensorRtSharp` 下，CUDA 类型位于 `JYPPX.CudaSharp` 下。`JYPPX.Shared.dll` 是内部程序集拆分名，其中的类型统一使用 `JYPPX.TensorRtSharp.Shared` 命名空间，不构成第三个公共命名空间根。

建议把这一页作为稳定的 API 文档入口，用于浏览：

- TensorRT 托管封装
- CUDA 托管封装
- shared bridge/runtime 辅助类型
