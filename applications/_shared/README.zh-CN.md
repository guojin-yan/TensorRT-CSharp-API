# 应用共享源码

[English](README.md) | 简体中文

该目录保存 `applications/` 下完整应用共用的应用层源码，不属于用户案例目录，其中的项目也不发布 NuGet 包。

`JYPPX.TensorRtSharp.ApplicationTools` 为 `OnnxToEngine` 和 `TensorRtExec` 链接当前 `JYPPX.TensorRtSharp.Tools` 源码，但使用已经发布的 4 系列 `JYPPX.TensorRT.CSharp.API` 包进行编译。这样两个应用的 TensorRT/CUDA 核心类型来自公共 NuGet，同时主库仍可保留使用当前源码的 Tools 项目进行开发和测试。

该共享项目禁止添加到 `src/JYPPX.CudaSharp` 或 `src/JYPPX.TensorRtSharp` 的 `ProjectReference`。CUDA、cuDNN、TensorRT 和 NVRTC 继续由用户自行安装。
