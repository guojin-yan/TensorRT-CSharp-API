![TensorRtSharp](https://socialify.git.ci/guojin-yan/TensorRT-CSharp-API/image?description=1&descriptionEditable=TensorRT%20wrapper%20for%20.NET&forks=1&issues=1&logo=https%3A%2F%2Fs2.loli.net%2F2023%2F04%2F11%2FOtsq6zAaZnwxP1U.png&name=1&owner=1&pattern=Circuit%20Board&pulls=1&stargazers=1&theme=Light)

简体中文| [English](README.md)

# TensorRtSharp

[![NuGet](https://img.shields.io/nuget/v/JYPPX.TensorRT.CSharp.API)](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API)[![License](https://img.shields.io/github/license/guojin-yan/TensorRT-CSharp-API)](LICENSE)

**TensorRtSharp** 是对 NVIDIA TensorRT 的完整 C# 封装，让 .NET 开发者能够在不离开 C# 生态的情况下享受高性能 GPU 推理。

[English](README_EN.md) | 简体中文

## 🚀 特性

- ✅ **完整的 API 覆盖** - 支持 TensorRT 核心功能，包括模型构建、推理执行、动态形状等
- ✅ **类型安全** - 强类型系统，编译时错误检查
- ✅ **自动资源管理** - 基于 RAII 和 Dispose 模式的资源管理，防止内存泄漏
- ✅ **跨平台支持** - 支持 Windows、Linux，兼容 .NET 5.0-10.0、.NET Core 3.1、.NET Framework 4.7.1+
- ✅ **高性能异步执行** - 支持 CUDA Stream、多执行上下文并行推理
- ✅ **开箱即用** - NuGet 包含所有依赖，无需复杂配置

## 📦 安装

### 通过 NuGet 安装

```bash
# 安装接口包
dotnet add package JYPPX.TensorRT.CSharp.API

# 安装运行时包（根据您的 CUDA 版本选择）
# CUDA 12.x 版本
dotnet add package JYPPX.TensorRT.CSharp.API.runtime.win-x64.cuda12

# CUDA 11.x 版本
dotnet add package JYPPX.TensorRT.CSharp.API.runtime.win-x64.cuda11
```

### 系统要求

| 要求 | 说明 |
|------|------|
| **操作系统** | Windows 10+、Linux（Ubuntu 18.04+） |
| **.NET 版本** | .NET 5.0-10.0、.NET Core 3.1、.NET Framework 4.7.1+ |
| **GPU** | NVIDIA GPU（支持 CUDA 11.x 或 12.x） |
| **依赖** | NVIDIA TensorRT 10.x、CUDA Runtime |

> **⚠️ 重要提醒**：TensorRtSharp 3.0 基于 TensorRT 10.x 开发，不支持 TensorRT 8.x 或 9.x 版本。

### 环境配置

安装 NuGet 包后，需要将以下路径添加到系统 PATH：

- CUDA 的 `bin` 目录（如 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin`）
- TensorRT 的 `lib` 目录（如 `C:\TensorRT-10.13.0.35\lib`）

## 🎯 快速开始

### 基础推理示例

```csharp
using JYPPX.TensorRtSharp.Cuda;
using JYPPX.TensorRtSharp.Nvinfer;

// 加载引擎
byte[] engineData = File.ReadAllBytes("model.engine");
Runtime runtime = new Runtime();

using CudaEngine engine = runtime.deserializeCudaEngineByBlob(engineData, (ulong)engineData.Length)
using ExecutionContext context = engine.createExecutionContext()
using CudaStream stream = new CudaStream()
using Cuda1DMemory<float> input = new Cuda1DMemory<float>(3 * 640 * 640)
using Cuda1DMemory<float> output = new Cuda1DMemory<float>(1000)
{
    // 绑定张量地址
    context.setInputTensorAddress("images", input.get());
    context.setOutputTensorAddress("output", output.get());

    // 准备输入数据
    float[] inputData = PreprocessImage("image.jpg");
    input.copyFromHost(inputData);

    // 执行推理
    context.executeV3(stream);
    stream.Synchronize();

    // 获取结果
    float[] outputData = new float[1000];
    output.copyToHost(outputData);
}
```

### 模型构建（ONNX → Engine）

```csharp
using Builder builder = new Builder();
using NetworkDefinition network = builder.createNetworkV2(TrtNetworkDefinitionCreationFlag.kEXPLICIT_BATCH)
using BuilderConfig config = builder.createBuilderConfig()
using OnnxParser parser = new OnnxParser(network)
{
    // 解析 ONNX 模型
    parser.parseFromFile("model.onnx", verbosity: 2);

    // 启用 FP16
    config.setFlag(TrtBuilderFlag.kFP16);

    // 构建并序列化
    using HostMemory serialized = builder.buildSerializedNetwork(network, config);

    // 保存到文件
    File.WriteAllBytes("model.engine", serialized.getByteData());
}
```

## 📚 文档

完整文档请访问：

- [使用指南](https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp3.0/docs/TensorRtSharp%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97-202601-0.0.5.md)
- [示例代码](https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp3.0/samples)

## 🏗️ 架构设计

TensorRtSharp 采用清晰的三层架构：

```
┌─────────────────────────────────────────┐
│         高级 API 层                     │
│   Runtime, Builder, Engine, Context     │
└─────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────┐
│        资源管理层                        │
│   DisposableTrtObject, RAII 模式        │
└─────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────┐
│       P/Invoke 互操作层                  │
│   NativeMethodsTensorRt*, NativeCuda*   │
└─────────────────────────────────────────┘
```

## 💡 核心类

### Runtime
TensorRT 推理的入口点，负责从序列化的引擎文件创建推理引擎。

```csharp
Runtime runtime = new Runtime();
using CudaEngine engine = runtime.deserializeCudaEngineByBlob(data, size);
runtime.setMaxThreads(4);
```

### Builder
从 ONNX 模型构建 TensorRT 引擎。

```csharp
Builder builder = new Builder();
bool hasFP16 = builder.platformHasFastFp16();
using NetworkDefinition network = builder.createNetworkV2(flags);
```

### CudaEngine
推理的核心对象，包含优化后的模型计算图。

```csharp
string inputName = engine.getIOTensorName(0);
Dims inputShape = engine.getTensorShape(inputName);
using ExecutionContext context = engine.createExecutionContext();
```

### ExecutionContext
管理单次推理的执行环境。

```csharp
context.setInputTensorAddress("images", inputPtr);
context.setOutputTensorAddress("output", outputPtr);
context.executeV3(stream);
```

## 📊 性能

使用 TensorRT 通常可以获得：

- 📈 **推理速度提升 2-10 倍**（相比原生框架）
- 💾 **显存占用降低 50% 以上**（通过精度优化和层融合）
- ⚡ **延迟降低至毫秒级**（满足实时应用需求）

## 🆚 与其他库对比

| 特性 | TensorRtSharp | ML.NET | ONNX Runtime |
|------|--------------|--------|--------------|
| **编程语言** | C# | C# | C++/Python |
| **性能** | 原生速度 | 中等 | 原生速度 |
| **TensorRT 支持** | ✅ 完整 | ❌ 无 | ⚠️ 有限 |
| **自定义算子** | ✅ 支持 | ⚠️ 困难 | ✅ 支持 |
| **动态形状** | ✅ 支持 | ⚠️ 有限 | ✅ 支持 |
| **多 GPU** | ✅ 支持 | ⚠️ 有限 | ✅ 支持 |

## 🐛 常见问题

### 问题一：找不到 DLL 模块

**错误信息**：
```
Unable to load DLL 'TensorRT-C-API'
```

**解决方案**：
1. 检查是否安装了对应版本的 Runtime NuGet 包
2. 确认系统 PATH 中包含 TensorRT 的 lib 目录和 CUDA 的 bin 目录
3. 确认 TensorRT 版本为 10.x

### 问题二：SEHException 异常

**错误信息**：
```
System.Runtime.InteropServices.SEHException
```

**解决方案**：
1. 确认 TensorRT 版本为 10.x
2. 检查 CUDA 版本是否匹配
3. 重新生成 Engine 文件

## 🤝 贡献

欢迎贡献代码！请随时提交 Pull Request。

## 📄 许可证

本项目采用Apache-2.0 license 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 👨‍💻 作者

**Guojin Yan**

## 📮 技术支持

- 📧 **GitHub Issues**：[提交问题](https://github.com/guojin-yan/TensorRT-CSharp-API/issues)
- 💬 **QQ 交流群**：加入 **945057948**，回复更快更方便

![QQ群二维码](https://ygj-images1.oss-cn-hangzhou.aliyuncs.com/images/202502242110187.png)

## 🙏 致谢

- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) - 高性能深度学习推理优化器
- [.NET 社区](https://dotnet.microsoft.com/) - 感谢提供优秀的开发者平台

---

**用 ❤️ 打造，献给 .NET 社区**
