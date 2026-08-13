# TensorRT CSharp API v4.0 接口使用

本模块按“类职责 + 功能边界”介绍 TensorRT CSharp API v4.0 的公开接口。它不是把 C++ 头文件逐项翻译成 C# 名称，而是说明一个接口在构建、运行、诊断和资源管理中的责任、生命周期、所有权和可验证输出。阅读时应同时关注托管对象、native Bridge、TensorRT/CUDA 运行库和应用代码四层边界。

TensorRT CSharp API v4.0 是 TensorRT CSharp API 项目的 4.0.0 正式接口线，源码和示例统一位于以下地址：

```text
项目源码：https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0
接口源码：https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/src
示例源码：https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/samples
核心 NuGet：https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/4.0.0
Runtime Bridge 包列表：https://www.nuget.org/packages?q=+JYPPX.TensorRT.CSharp&includeComputedFrameworks=true&prerel=true&sortby=relevance
```

## 1. 阅读方式

先阅读对象模型文章，再按应用需要选择构建接口、运行时接口、CUDA 资源接口或诊断回调接口。每篇接口文章都会给出最小代码、程序出处和输出判定；代码中的 `Dispose`、`using`、异步边界和 GPU 内存所有权不能省略。

## 2. 接口分组

| 分组 | 主要类或功能 | 说明 |
| --- | --- | --- |
| 构建 | `TensorRtBuilder`、`TensorRtNetwork`、`TensorRtBuilderConfig`、`TensorRtOptimizationProfile` | 从网络定义、ONNX 或层 API 构建可序列化 Engine |
| 运行时 | `TensorRtRuntime`、`TensorRtEngine`、`TensorRtExecutionContext` | 反序列化 Engine、创建 Context、提交推理 |
| 绑定 | `TensorRtTensor`、`InferenceBindings`、Shape/Datatype 元数据 | 维护输入输出名称、Shape、数据类型和设备指针 |
| 模型 | `OnnxParser`、`OnnxParserRefitter`、序列化接口 | 导入 ONNX、收集解析诊断、替换权重并持久化 |
| 诊断 | `TensorRtLogger`、`TensorRtProfiler`、`TensorRtProgressMonitor`、`TensorRtDebugListener` | 把 native 回调转换为可控的托管生命周期 |
| CUDA | `CudaDevice`、`CudaStream`、`CudaEvent`、显存分配和 Graph | 管理设备、异步执行、同步和 CUDA 资源所有权 |

## 3. 当前文章与后续规划

| ID | 主题 | 状态 |
| --- | --- | --- |
| `MSC-007` | TensorRT 对象模型与 API 调用顺序 | `ready` |
| [`API-001`](tensorrt/api-001-builder-network-config-profile.md) | Builder、Network、BuilderConfig 与 OptimizationProfile | `ready` |
| [`API-002`](tensorrt/api-002-runtime-engine-context-bindings.md) | Runtime、Engine、ExecutionContext 与推理绑定 | `ready` |
| [`API-003`](cuda/api-003-cuda-device-memory-stream-event-graph.md) | CUDA Device、Memory、Stream、Event 与 Graph | `ready` |
| [`API-004`](onnx/api-004-onnx-parser-parser-refitter-diagnostics.md) | ONNX Parser、ParserRefitter 与诊断信息 | `ready` |
| [`API-005`](diagnostics/api-005-callbacks-logger-profiler-progress-debug-listener.md) | Logger、Profiler、ProgressMonitor、DebugListener | `ready` |
| [`API-006`](engine/api-006-serialization-engine-inspector-error-boundary.md) | 序列化、Engine Inspector 与错误边界 | `ready` |

截至 2026-08-13，本模块 7 篇 canonical 文章均为 `ready`。`API-001` 至 `API-006` 已在同一台 Windows TensorRT 10.11/CUDA 12.9 主机完成 API line、进程级 DLL 路径、GPU/CUDA 正路径和受控失败复核；批次命令、退出码、日志哈希和证明边界记录在 [`api-runtime-evidence-20260813.json`](api-runtime-evidence-20260813.json)。这些记录不外推为 TensorRT 8/11、真实模型精度、独立包消费者或性能基准证明。

## 4. 接口文章限制

1. 文章必须标出使用的具体类、方法和适用 TensorRT API line。
2. 代码必须来自源码仓库中的真实示例或明确标注为最小缩减片段。
3. 输出必须区分真实运行、静态检查和期望结果，不能用截图替代运行证据。
4. 所有 native 句柄、CUDA 指针和回调对象都要说明创建者、使用者和释放者。
