# CUDA 运行时编译案例

[English](README.md) | 简体中文

该案例演示 `JYPPX.CudaSharp` 的 NVRTC 安全封装，包括虚拟头文件、名称表达式、PTX 元数据复制、预期编译失败日志，以及通过 CUDA Runtime 和 Driver Module 两条路径完成向量加法启动与 GPU 结果读回。

## 环境

- 安装与 Bridge 包匹配的 CUDA Toolkit。
- NVRTC 由用户安装，Bridge 包不会携带 `nvrtc` 动态库。
- 可通过环境变量显式选择本机 Bridge 和 NVRTC。

```powershell
$env:JYPPX_NATIVE_BRIDGE_PATH = '<bridge-path>'
$env:JYPPX_NVRTC_LIBRARY = '<nvrtc-library-path>'
dotnet run --project .\samples\Cuda\01.RuntimeCompilation
```

成功运行会输出编译产物信息、类型化 kernel 启动结果和预期失败诊断。当前可发布文章、真实 Windows Terminal 截图和运行边界见 [CUDA RTC 技术文章](../../../docs/articles/zh-cn/cuda-runtime-compilation-technical-article.md)。

该案例只能证明当前机器上的 CUDA RTC 与 kernel 路径；它不代表其他 CUDA 版本、Linux 环境或公开 NuGet 消费者已经验证。
