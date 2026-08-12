# TensorRT CSharp API v4.0 ONNX Parser：模型导入、ParserRefitter 与诊断

<!-- public-article-layout:start -->
<style>
.content article { min-width: 0; overflow-wrap: anywhere; }
.content article a, .content article :not(pre) > code { overflow-wrap: anywhere; word-break: break-word; }
.content article pre:not(.mermaid), .content article table { display: block; max-width: 100%; overflow-x: auto; }
.content pre.mermaid { max-width: 640px; margin: 16px auto; overflow-x: auto; }
.content pre.mermaid svg { display: block; width: 100%; max-width: 640px; height: auto; }
</style>
<!-- public-article-layout:end -->

> 文章编号：API-004；适用版本：4.0.0；当前状态：review。

## 1. 前言
<!-- public-article-project-preface:start -->
TensorRT CSharp API v4.0 是一个面向 C#/.NET 开发者的 TensorRT 与 CUDA 工程化接口项目。它把 NVIDIA 原生运行时、生成式绑定、C++ Bridge、托管对象模型和可验证的示例程序组织成一条完整链路，使使用者可以在熟悉的 .NET 项目中完成 Engine 构建、反序列化、ExecutionContext 管理、CUDA 内存操作、异步流同步和结果校验。项目的目标不是隐藏 TensorRT 的概念，而是把这些概念转换为有明确生命周期、所有权和错误边界的 C# API。

4.0.0 是一次完整重构后的正式版本。核心接口、Bridge 边界、Runtime 包命名、样例目录和验证方式都以 4.x 设计为准，不能把 3.x 的类型名、旧包名或旧 DLL 目录直接复制到新项目。托管包只提供项目接口和自有 Bridge；TensorRT、CUDA、cuDNN、显卡驱动以及对应许可证仍由使用者按目标平台安装和管理。

单篇文章也应能够独立阅读：读者可以先从项目入口确认源码和包，再根据本文的程序路径准备依赖，最后用输出中的状态、计数、Shape、哈希或结果图片判断流程是否真的完成。对于尚未具备兼容 GPU 的环境，本文会把静态检查、期望输出和真实运行结果分开标记，不把帮助命令或 build-only 结果包装成推理成功。

项目、包和源码入口（以下地址保留明文，便于复制到不完整支持 Markdown 链接的平台）：

项目主页：

```text
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0
```

核心 NuGet：

```text
https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/4.0.0
```

Runtime Bridge 包列表：

```text
https://www.nuget.org/packages?q=+JYPPX.TensorRT.CSharp&includeComputedFrameworks=true&prerel=true&sortby=relevance
```

运行库清单：

```text
https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/pack/runtime/runtime-packages.manifest.json
```

### 1.1 程序出处与输出说明

本文涉及的程序、脚本或命令均以仓库中的实现为准；对应源码入口：

```text
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0
```

运行示例必须同时给出程序输出和判定标准。终端中的 `status`、`Ready`、`Bound`、`enqueueCount`、`OutputMatch`、进程退出码、报告文件或结果图片分别说明不同层次的事实；只有明确写出这些结果，读者才能区分程序启动、Engine 构建、GPU enqueue 和业务结果语义。
<!-- public-article-project-preface:end -->

本文介绍 `TensorRtOnnxParser`、`TensorRtOnnxParserRefitter` 与 `TensorRtRefitter` 的职责。Parser 把 ONNX 图导入 Network；ParserRefitter 从新的 ONNX 模型读取可替换权重；Refitter 检查缺失权重并把更新提交到已有 Engine。三者都提供复制型诊断，但诊断成功不能代替 Engine 构建、反序列化或输出校验。

## 2. Parser 在构建链中的位置

ONNX Parser 不直接生成 Engine。它依赖已经创建的 Logger 和 Network，把模型节点、张量与权重导入 Network，随后仍由 BuilderConfig 和 Builder 完成构建。

```text
ONNX 文件或字节
  -> TensorRtOnnxParser
  -> TensorRtNetworkDefinition
  -> TensorRtBuilderConfig + OptimizationProfile
  -> BuildSerializedNetwork
```

Parser 会借用 Logger 和 Network 对应的原生对象。应让 Logger、Network 和 Parser 覆盖整个解析调用，并在构建结束后按 `Parser -> Network -> Builder` 的依赖方向释放。

## 3. 从文件或内存解析 ONNX

仓库同时提供文件、字节数组、`ArraySegment<byte>`、`ReadOnlySpan<byte>` 和 `Stream` 入口。最常见的文件解析如下：

```csharp
using TensorRtBuilder builder = new TensorRtBuilder(logger);
using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
using TensorRtNetworkDefinition network =
    builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
using TensorRtOnnxParser parser = new TensorRtOnnxParser(logger, network);

if (!parser.ParseFromFile(modelPath))
{
    throw new InvalidDataException(parser.GetDiagnosticSummary());
}

using TensorRtHostMemory plan = builder.BuildSerializedNetwork(network, config);
plan.SaveToFile(enginePath);
```

内存和 Stream 重载会在进入 native interop 前复制为受控的托管字节。`modelPath` 参数用于补充 TensorRT 诊断上下文，不表示 Parser 会替调用方管理外部数据文件。

## 4. 解析失败时收集完整诊断

只保留 `ParseFromFile=false` 会丢失真正的失败位置。应记录 `ErrorCount`，遍历 `GetDiagnostics()`，并保存节点名、算子、文件、行号、本地函数栈和描述。

```csharp
if (!parser.ParseFromFile(modelPath))
{
    foreach (TensorRtOnnxParserDiagnostic diagnostic in parser.GetDiagnostics())
    {
        Console.Error.WriteLine(
            $"#{diagnostic.Index} code={diagnostic.Code} " +
            $"node={diagnostic.NodeName} op={diagnostic.NodeOperator}: " +
            diagnostic.Description);
    }

    throw new InvalidDataException(parser.GetDiagnosticSummary());
}
```

诊断对象是复制到托管侧的只读值，不暴露 TensorRT 拥有的 parser-error 指针。它能证明“本次解析返回了哪些诊断”，不能证明模型可构建或可推理。

## 5. 动态形状与算子支持

解析成功后仍要检查 Network 输入维度。存在 `-1` 时，需要按输入名称建立 OptimizationProfile；缺少 Profile 通常在构建阶段失败，而不是 Parser 阶段。

真实模型失败还可能来自不支持的 ONNX 算子、opset 差异、插件缺失、外部 initializer 路径或数据类型限制。Parser 的模型与算子支持查询可用于预检，但最终结论仍以实际解析和构建结果为准。

## 6. ParserRefitter 与 Refitter 的分工

Refit 流程要求原 Engine 在构建时启用相应 Refit 标志。`TensorRtRefitter` 绑定已有 Engine，`TensorRtOnnxParserRefitter` 读取新 ONNX 模型中的 initializer，最终由 Refitter 提交更新。

```csharp
using TensorRtRefitter refitter = engine.CreateRefitter(logger);
using TensorRtOnnxParserRefitter parserRefitter =
    refitter.CreateOnnxParserRefitter(logger);

bool modelAccepted = parserRefitter.RefitFromFile(updatedModelPath);
int missingWeights = refitter.MissingWeightCount;
bool committed = modelAccepted && missingWeights == 0 && refitter.RefitCudaEngine();

if (!committed)
{
    throw new InvalidOperationException(
        parserRefitter.GetDiagnosticSnapshot().ToString());
}
```

`RefitFromFile=true` 只表示 ParserRefitter 接受了模型读取过程。还必须检查缺失权重、`RefitCudaEngine()` 返回值，并对更新前、更新后和持久化重载后的输出分别校验。

## 7. 版本与所有权边界

Parser Flag、模型加载入口和部分诊断能力会随 TensorRT 8、10、11 变化。跨版本程序应选择明确的 `TensorRtApiLine`，对不支持选项保留 `NotSupportedException`，不要通过吞掉异常伪装成兼容。

Parser、ParserRefitter 和 Refitter 都保持对借用 Logger 或父对象的托管引用。调用方不应提前释放 Logger、Network、Engine 或 Refitter；诊断快照可以在对应 native owner 释放后继续读取，因为其中不保留原生指针。

## 8. 源码与样例入口

```text
ONNX Parser 源码：
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/src/JYPPX.TensorRtSharp/Parsing

ONNX 构建与运行样例：
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/samples/Inference/03.OnnxBuildAndRun

Refitted Plan 样例：
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/samples/Inference/04.RefittedPlan
```

## 9. 验证命令与判定

```powershell
dotnet run --project .\samples\Inference\03.OnnxBuildAndRun -- --synthetic --tensor-rt-line 10
dotnet run --project .\samples\Inference\04.RefittedPlan -- --synthetic --tensor-rt-line 10
```

Parser 链路应同时看到解析成功、Engine 构建完成、反序列化成功、Enqueue 完成和输出匹配。Refit 链路还应看到 Engine 可 refit、缺失权重清零、提交成功、输出发生预期变化，并且保存后重新加载的输出仍匹配。

本文已完成源码、样例和复制型诊断边界复核，状态保持 `review`。发布前应在目标 TensorRT line 上保留一次成功路径和一次故意解析失败的完整诊断记录。

## 10. 小结

ONNX Parser 负责把模型转换为 Network，ParserRefitter 负责从 ONNX 提取替换权重，Refitter 负责验证并提交更新。可靠流程必须把解析、构建、提交、序列化和输出校验作为独立阶段记录。

<!-- public-article-declaration:start -->
## 11. 文章声明

### 11.1 开源协议声明
作者所有开源项目代码均遵循 Apache License 2.0 开源协议。
特别说明：本项目集成了若干第三方库。若任何第三方库的许可协议与 Apache 2.0 协议存在冲突或不一致，均以该第三方库的原始许可协议为准。本项目不包含也不代表这些第三方库的授权声明，使用前请务必阅读并遵守第三方库的相关许可。

### 11.2 代码开发与质量说明
AI 辅助开发：本代码在开发过程中使用了人工智能（AI）辅助生成与优化，并非完全由人工逐行编写。
安全性承诺：作者郑重声明，本代码中绝无任何有意设置的后门、病毒、木马或旨在破坏用户设备、窃取数据的恶意代码。
技术局限性：受限于作者个人的技术水平与能力，代码中可能存在因逻辑不严谨、优化不足或经验欠缺导致的低级问题（例如但不限于内存泄漏、偶发崩溃、资源未释放等）。这些问题纯属能力不足所致，并非主观故意。
测试范围：由于作者精力有限，未对本软件进行全方位、覆盖所有边缘场景的完整测试。

### 11.3 免责声明（重要）
请在将本代码应用于任何实际项目（特别是商业、工业或关键任务环境）之前，务必进行详尽、严格的自行测试与验证。 鉴于上述可能存在的代码缺陷及测试覆盖不足，因使用本代码而导致的任何直接或间接损失（包括但不限于设备故障、数据丢失、系统瘫痪或利润损失等），本作者概不负责。 一旦您开始使用本代码，即表示您已知晓上述风险并同意自行承担一切后果，相关问题与本作者无关。

### 11.4 代码开源范围
本项目承诺核心逻辑代码完全开源，但上述提到的“第三方库”的二进制文件、源代码或相关资源不在本项目的开源义务范围内，请根据其各自的指引获取。

### 11.5 社区与反馈
尽管存在上述不足，我们仍欢迎大家下载使用、提交 Issue 或参与测试，共同完善项目。如果您在使用过程中发现 Bug、内存溢出或有改进建议，欢迎通过项目主页提供的联系方式与作者取得联系，我们将尽力在有限的时间内提供协助。

<img src="../../../../images/personal-contact-banner-v6-zh.png" alt="作者联系方式" width="640" style="display:block;max-width:100%;height:auto;margin:16px auto;" />
<!-- public-article-declaration:end -->
