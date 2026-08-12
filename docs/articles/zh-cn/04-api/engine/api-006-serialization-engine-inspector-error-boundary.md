# TensorRT CSharp API v4.0 序列化：Engine Inspector 与错误边界

<!-- public-article-layout:start -->
<style>
.content article { min-width: 0; overflow-wrap: anywhere; }
.content article a, .content article :not(pre) > code { overflow-wrap: anywhere; word-break: break-word; }
.content article pre:not(.mermaid), .content article table { display: block; max-width: 100%; overflow-x: auto; }
.content pre.mermaid { max-width: 640px; margin: 16px auto; overflow-x: auto; }
.content pre.mermaid svg { display: block; width: 100%; max-width: 640px; height: auto; }
</style>
<!-- public-article-layout:end -->

> 文章编号：API-006；适用版本：4.0.0；当前状态：review。

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

本文介绍 `TensorRtHostMemory`、`TensorRtSerializationConfig` 和 `TensorRtEngineInspector`。序列化负责把 Engine 变成可持久化字节，Inspector 负责读取 Engine 或 Layer 的复制型描述，ErrorRecorder Snapshot 负责复制诊断。这三类能力都很适合排障，但都不能单独证明推理输出正确。

## 2. 构建产物与 Engine 再序列化

Builder 的 `BuildSerializedNetwork()` 返回 `TensorRtHostMemory`。它可以复制为字节数组、保存到文件或转换为只读 Stream。

```csharp
using TensorRtHostMemory plan = builder.BuildSerializedNetwork(network, config);
Console.WriteLine($"PlanBytes={plan.SizeInBytes}");
plan.SaveToFile(enginePath);

using TensorRtEngine engine = runtime.DeserializeFromFile(enginePath);
using TensorRtHostMemory serializedAgain = engine.Serialize();
serializedAgain.SaveToFile(roundTripPath);
```

文件生成后应记录字节数和 SHA256，并执行一次独立反序列化。两个序列化文件不必逐字节相同；是否等价应结合 Engine 元数据、运行环境和实际输出判断。

## 3. TensorRtHostMemory 的所有权

HostMemory 包装 TensorRT 分配的原生 Host Buffer。`ToArray()` 会复制内容到托管数组，`SaveToFile()` 会把当前内容落盘。调用方必须在对象释放前完成复制或保存。

| 操作 | 返回内容 | 释放边界 |
|---|---|---|
| `ToArray()` | 独立托管字节数组 | HostMemory 释放后仍可使用 |
| `OpenRead()` | 基于复制字节的只读 Stream | Stream 由调用方释放 |
| `SaveToFile()` | 文件副本 | 文件生命周期独立于 HostMemory |
| `SizeInBytes` | 当前原生 Buffer 大小 | 只用于容量和审计，不证明格式有效 |

仅检查 `SizeInBytes > 0` 不足以证明文件能被 Runtime 接受；还要在匹配的 TensorRT/GPU 环境中执行 `DeserializeFromFile()`。

## 4. SerializationConfig 的版本边界

TensorRT 10 和 11 可以通过 Engine 创建 SerializationConfig，再调用带 Config 的序列化重载。TensorRT 8 应使用无参数 `Serialize()`。

```csharp
using TensorRtSerializationConfig serializationConfig =
    engine.CreateSerializationConfig();

serializationConfig.ClearFlag(TensorRtSerializationFlag.ExcludeWeights);
bool excludesWeights =
    serializationConfig.GetFlag(TensorRtSerializationFlag.ExcludeWeights);

if (excludesWeights)
{
    throw new InvalidOperationException("Refittable weights are still excluded.");
}

using TensorRtHostMemory fullPlan = engine.Serialize(serializationConfig);
fullPlan.SaveToFile(fullPlanPath);
```

这在 stripped/refitted plan 持久化时尤其重要。写入 Flag 后必须读回，不应只根据调用没有抛异常判断配置已生效。Config 和 Engine 必须属于同一 `TensorRtApiLine`。

## 5. 创建 Engine Inspector

Inspector 从已有 Engine 创建，可以返回整个 Engine 的 Oneline 或 JSON 描述，也可在绑定 ExecutionContext 后输出与 Context 相关的信息。

```csharp
using TensorRtEngineInspector inspector = engine.CreateInspector();

string oneLine = inspector.GetEngineInformation(
    TensorRtLayerInformationFormat.Oneline);
string json = inspector.GetEngineInformation(
    TensorRtLayerInformationFormat.Json);

Console.WriteLine(oneLine);
File.WriteAllText(layerInfoPath, json);
```

如果需要 Context-aware 信息，可在 Context 存活期间调用 `SetExecutionContext(context)`。Inspector 会借用 Context 关联状态，因此释放或清理 Context 前应先结束相关读取。

## 6. Inspector 能证明什么

Inspector 返回的是复制型文本和元数据。它适合核对 Layer 数量、名称、实现、Tensor 格式、Profile 和构建配置投影，但不绑定输入输出、不 Enqueue，也不读取输出张量。

```text
Inspector evidence
- engine deserialized: yes
- inspector information copied: yes
- layer information length: greater than 0
- inference enqueued by inspector: no
- output validated by inspector: no
```

即使 JSON 信息可稳定哈希，也只能证明两份复制型描述是否一致，不能把它升级为真实模型、包消费者或发布运行证明。

## 7. ErrorRecorder 的复制型边界

Engine、Inspector、Runtime、Context 或 Refitter 的部分接口可以报告是否附加 ErrorRecorder，并尝试获取 `TensorRtErrorRecorderSnapshot`。Snapshot 复制错误码与描述，不暴露 native recorder 指针，也不允许调用方控制引用计数。

```csharp
if (inspector.TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot))
{
    Console.WriteLine(snapshot.ToString());
}
```

Snapshot 可用于故障报告；它不能证明自定义 ErrorRecorder 生命周期安全，也不能表示错误已经恢复。读取完成后是否调用 `ClearErrorRecorder()`，应由明确的诊断流程决定，避免在记录证据前清空错误。

## 8. 安全持久化与复核流程

建议按下面顺序生成可审计 Engine 产物：

1. 构建或加载 Engine，并记录 TensorRT line、GPU 和版本。
2. 读取 Engine 基础元数据与 Inspector 描述。
3. 使用正确的 SerializationConfig 生成目标字节。
4. 保存文件、长度和 SHA256。
5. 释放原 Engine owner。
6. 在新的 Runtime/Engine owner 中重新加载文件。
7. 再次核对元数据，并以真实输入执行输出校验。

这样可以区分序列化成功、文件可重载、元数据一致和推理结果一致四个结论。

## 9. 源码与样例入口

```text
Engine Inspector 源码：
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/src/JYPPX.TensorRtSharp/Engine

序列化封装源码：
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/src/JYPPX.TensorRtSharp/Serialization

Refitted Plan 持久化样例：
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/samples/Inference/04.RefittedPlan

TensorRtExec 应用：
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/applications/TensorRtExec
```

## 10. 验证命令与判定

```powershell
dotnet run --project .\samples\Inference\04.RefittedPlan -- --synthetic --tensor-rt-line 10
dotnet run --project .\applications\TensorRtExec -- --help
```

第一条命令用于验证 Engine 序列化、文件重载和输出一致性。第二条只验证工具入口与参数可见，不能证明 Inspector 或推理已经执行。Inspector 专题发布前还应保存实际 Engine 的 Oneline/JSON 输出、文件 SHA256、重载结果和输出校验。

本文已完成源码和能力边界复核，状态保持 `review`。SerializationConfig 和部分 Inspector/ErrorRecorder 控制存在 TensorRT line 差异，发布记录必须注明实际运行版本。

## 11. 小结

序列化负责交付 Engine 字节，Inspector 负责解释复制型结构，ErrorRecorder Snapshot 负责保留诊断。三者共同提高可审计性，但只有独立重载、真实 Enqueue 和输出校验才能完成运行闭环。

<!-- public-article-declaration:start -->
## 12. 文章声明

### 12.1 开源协议声明
作者所有开源项目代码均遵循 Apache License 2.0 开源协议。
特别说明：本项目集成了若干第三方库。若任何第三方库的许可协议与 Apache 2.0 协议存在冲突或不一致，均以该第三方库的原始许可协议为准。本项目不包含也不代表这些第三方库的授权声明，使用前请务必阅读并遵守第三方库的相关许可。

### 12.2 代码开发与质量说明
AI 辅助开发：本代码在开发过程中使用了人工智能（AI）辅助生成与优化，并非完全由人工逐行编写。
安全性承诺：作者郑重声明，本代码中绝无任何有意设置的后门、病毒、木马或旨在破坏用户设备、窃取数据的恶意代码。
技术局限性：受限于作者个人的技术水平与能力，代码中可能存在因逻辑不严谨、优化不足或经验欠缺导致的低级问题（例如但不限于内存泄漏、偶发崩溃、资源未释放等）。这些问题纯属能力不足所致，并非主观故意。
测试范围：由于作者精力有限，未对本软件进行全方位、覆盖所有边缘场景的完整测试。

### 12.3 免责声明（重要）
请在将本代码应用于任何实际项目（特别是商业、工业或关键任务环境）之前，务必进行详尽、严格的自行测试与验证。 鉴于上述可能存在的代码缺陷及测试覆盖不足，因使用本代码而导致的任何直接或间接损失（包括但不限于设备故障、数据丢失、系统瘫痪或利润损失等），本作者概不负责。 一旦您开始使用本代码，即表示您已知晓上述风险并同意自行承担一切后果，相关问题与本作者无关。

### 12.4 代码开源范围
本项目承诺核心逻辑代码完全开源，但上述提到的“第三方库”的二进制文件、源代码或相关资源不在本项目的开源义务范围内，请根据其各自的指引获取。

### 12.5 社区与反馈
尽管存在上述不足，我们仍欢迎大家下载使用、提交 Issue 或参与测试，共同完善项目。如果您在使用过程中发现 Bug、内存溢出或有改进建议，欢迎通过项目主页提供的联系方式与作者取得联系，我们将尽力在有限的时间内提供协助。

<img src="../../../../images/personal-contact-banner-v6-zh.png" alt="作者联系方式" width="640" style="display:block;max-width:100%;height:auto;margin:16px auto;" />
<!-- public-article-declaration:end -->
