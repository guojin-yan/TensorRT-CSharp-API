![TensorRtSharp](https://socialify.git.ci/guojin-yan/TensorRT-CSharp-API/image?description=1&descriptionEditable=TensorRT%20wrapper%20for%20.NET&forks=1&issues=1&logo=https%3A%2F%2Fs2.loli.net%2F2023%2F04%2F11%2FOtsq6zAaZnwxP1U.png&name=1&owner=1&pattern=Circuit%20Board&pulls=1&stargazers=1&theme=Light)

[简体中文](README_cn.md) | English

# TensorRtSharp

[![NuGet](https://img.shields.io/nuget/v/JYPPX.TensorRT.CSharp.API)](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API)[![License](https://img.shields.io/github/license/guojin-yan/TensorRT-CSharp-API)](LICENSE)

**TensorRtSharp** is a complete C# wrapper for NVIDIA TensorRT, enabling .NET developers to leverage high-performance GPU inference without leaving the C# ecosystem.

English | [简体中文](README_CN.md)

## 🚀 Features

- ✅ **Complete API Coverage** - Full support for TensorRT core features including model building, inference execution, and dynamic shapes
- ✅ **Type Safety** - Strong type system with compile-time error checking
- ✅ **Automatic Resource Management** - RAII and Dispose pattern-based resource management prevents memory leaks
- ✅ **Cross-Platform** - Supports Windows, Linux, .NET 5.0-10.0, .NET Core 3.1, .NET Framework 4.7.1+
- ✅ **High Performance** - Async execution with CUDA Stream, multi-context parallel inference
- ✅ **Ready to Use** - NuGet packages include all dependencies, no complex configuration required

## 📦 Installation

### Via NuGet

```bash
# Install the API package
dotnet add package JYPPX.TensorRT.CSharp.API

# Install the runtime package (choose based on your CUDA version)
# For CUDA 12.x
dotnet add package JYPPX.TensorRT.CSharp.API.runtime.win-x64.cuda12

# For CUDA 11.x
dotnet add package JYPPX.TensorRT.CSharp.API.runtime.win-x64.cuda11
```

### System Requirements

| Requirement | Description |
|-------------|-------------|
| **OS** | Windows 10+, Linux (Ubuntu 18.04+), |
| **.NET** | .NET 5.0-10.0, .NET Core 3.1, .NET Framework 4.7.1+ |
| **GPU** | NVIDIA GPU (supports CUDA 11.x or 12.x) |
| **Dependencies** | NVIDIA TensorRT 10.x, CUDA Runtime |

> **⚠️ Important**: TensorRtSharp 3.0 is based on TensorRT 10.x and does not support TensorRT 8.x or 9.x.

### Configuration

After installing the NuGet packages, configure your system PATH to include:

- CUDA `bin` directory (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin`)
- TensorRT `lib` directory (e.g., `C:\TensorRT-10.13.0.35\lib`)

## 🎯 Quick Start

### Basic Inference Example

```csharp
using JYPPX.TensorRtSharp.Cuda;
using JYPPX.TensorRtSharp.Nvinfer;

// Load the engine
byte[] engineData = File.ReadAllBytes("model.engine");
Runtime runtime = new Runtime();

using CudaEngine engine = runtime.deserializeCudaEngineByBlob(engineData, (ulong)engineData.Length)
using ExecutionContext context = engine.createExecutionContext()
using CudaStream stream = new CudaStream()
using Cuda1DMemory<float> input = new Cuda1DMemory<float>(3 * 640 * 640)
using Cuda1DMemory<float> output = new Cuda1DMemory<float>(1000)
{
    // Bind tensor addresses
    context.setInputTensorAddress("images", input.get());
    context.setOutputTensorAddress("output", output.get());

    // Prepare input data
    float[] inputData = PreprocessImage("image.jpg");
    input.copyFromHost(inputData);

    // Execute inference
    context.executeV3(stream);
    stream.Synchronize();

    // Get results
    float[] outputData = new float[1000];
    output.copyToHost(outputData);
}
```

### Model Building (ONNX → Engine)

```csharp
using Builder builder = new Builder();
using NetworkDefinition network = builder.createNetworkV2(TrtNetworkDefinitionCreationFlag.kEXPLICIT_BATCH)
using BuilderConfig config = builder.createBuilderConfig()
using OnnxParser parser = new OnnxParser(network)
{
    // Parse ONNX model
    parser.parseFromFile("model.onnx", verbosity: 2);

    // Enable FP16
    config.setFlag(TrtBuilderFlag.kFP16);

    // Build and serialize
    using HostMemory serialized = builder.buildSerializedNetwork(network, config);

    // Save to file
    File.WriteAllBytes("model.engine", serialized.getByteData());
}
```

## 📚 Documentation

For complete documentation, please visit:

- [User Guide](https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp3.0/docs/TensorRtSharp%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97-202601-0.0.5.md)
- [Examples](https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp3.0/samples)

## 🏗️ Architecture

TensorRtSharp uses a clear three-layer architecture:

```
┌─────────────────────────────────────────┐
│    High-Level API Layer                │
│  Runtime, Builder, CudaEngine, Context  │
└─────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────┐
│    Resource Management Layer           │
│  DisposableTrtObject, RAII pattern     │
└─────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────┐
│    P/Invoke Interop Layer              │
│  NativeMethodsTensorRt*, NativeCuda*    │
└─────────────────────────────────────────┘
```

## 💡 Core Classes

### Runtime
Entry point for TensorRT inference, responsible for deserializing engine files.

```csharp
Runtime runtime = new Runtime();
using CudaEngine engine = runtime.deserializeCudaEngineByBlob(data, size);
runtime.setMaxThreads(4);
```

### Builder
Builds TensorRT engines from ONNX models.

```csharp
Builder builder = new Builder();
bool hasFP16 = builder.platformHasFastFp16();
using NetworkDefinition network = builder.createNetworkV2(flags);
```

### CudaEngine
Core inference object containing the optimized model computation graph.

```csharp
string inputName = engine.getIOTensorName(0);
Dims inputShape = engine.getTensorShape(inputName);
using ExecutionContext context = engine.createExecutionContext();
```

### ExecutionContext
Manages the execution environment for single inference.

```csharp
context.setInputTensorAddress("images", inputPtr);
context.setOutputTensorAddress("output", outputPtr);
context.executeV3(stream);
```

## 📊 Performance

TensorRT typically provides:

- 📈 **2-10x inference speedup** (compared to native frameworks)
- 💾 **50%+ memory reduction** (through precision optimization and layer fusion)
- ⚡ **Sub-millisecond latency** (meeting real-time application requirements)

## 🆚 Comparison

| Feature | TensorRtSharp | ML.NET | ONNX Runtime |
|---------|--------------|--------|--------------|
| **Language** | C# | C# | C++/Python |
| **Performance** | Native | Medium | Native |
| **TensorRT Support** | ✅ Complete | ❌ | ⚠️ Limited |
| **Custom Operators** | ✅ | ⚠️ Difficult | ✅ |
| **Dynamic Shapes** | ✅ | ⚠️ Limited | ✅ |
| **Multi-GPU** | ✅ | ⚠️ Limited | ✅ |

## 🐛 Troubleshooting

### Issue: Unable to load DLL

**Error**: `Unable to load DLL 'TensorRT-C-API'`

**Solution**:
1. Verify Runtime NuGet package is installed
2. Check PATH includes TensorRT lib and CUDA bin directories
3. Confirm TensorRT version is 10.x

### Issue: SEHException

**Error**: `System.Runtime.InteropServices.SEHException`

**Solution**:
1. Confirm TensorRT version is 10.x
2. Check CUDA version compatibility
3. Regenerate Engine file on current device

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the Apache-2.0 license License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Guojin Yan**

## 📮 Support

- 📧 **GitHub Issues**: [Submit an issue](https://github.com/guojin-yan/TensorRT-CSharp-API/issues)
- 💬 **QQ Group**: Join **945057948** for faster responses

## 🙏 Acknowledgments

- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) - High-performance deep learning inference optimizer
- [.NET community](https://dotnet.microsoft.com/) - For the amazing developer platform

![QQ群二维码](https://ygj-images1.oss-cn-hangzhou.aliyuncs.com/images/202502242110187.png)

---

**Made with ❤️ by the .NET community**
