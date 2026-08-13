# TensorRT CSharp API v4.0 Runtime：Engine、ExecutionContext 与推理绑定

<!-- public-article-layout:start -->
<style>
.content article { min-width: 0; overflow-wrap: anywhere; }
.content article a, .content article :not(pre) > code { overflow-wrap: anywhere; word-break: break-word; }
.content article pre:not(.mermaid), .content article table { display: block; max-width: 100%; overflow-x: auto; }
.content pre.mermaid { max-width: 640px; margin: 16px auto; overflow-x: auto; }
.content pre.mermaid svg { display: block; width: 100%; max-width: 640px; height: auto; }
</style>
<!-- public-article-layout:end -->

> 文章编号：API-002；适用版本：4.0.0；当前状态：ready。

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

本文梳理 TensorRT 运行阶段的对象链：`TensorRtRuntime` 负责反序列化，`TensorRtEngine` 保存已构建网络，`TensorRtExecutionContext` 保存一次执行所需状态，`TensorRtInferenceBindings` 管理输入输出地址和推理准备度。理解这些边界，可以避免把 Engine、Context、设备内存和 Stream 的生命周期混在一起。

## 2. 从 Engine 文件到 ExecutionContext

典型运行流程如下：

```text
Engine 字节
  -> TensorRtRuntime
  -> TensorRtEngine
  -> TensorRtExecutionContext
  -> 输入形状与张量地址
  -> EnqueueAsync
  -> Stream 同步或事件等待
```

Runtime 在 Engine 反序列化期间必须有效；Engine 在 Context 存活期间必须有效；Context 使用的设备内存、输入输出 Buffer 和 CUDA Stream 必须覆盖异步推理完成之前的整个时间段。

## 3. 绑定输入输出的基本顺序

静态形状和动态形状都应遵循“设置形状、准备内存、绑定地址、检查就绪、提交执行”的顺序。

```csharp
using TensorRtRuntime runtime = new TensorRtRuntime(logger);
using TensorRtEngine engine = runtime.Deserialize(engineBytes);
using TensorRtExecutionContext context = engine.CreateExecutionContext();
using CudaStream stream = new CudaStream();
using TensorRtInferenceBindings bindings = new TensorRtInferenceBindings(engine, context, profileIndex);

TensorRtDims runtimeShape = new TensorRtDims(new[] { 1, 3, 640, 640 });
bindings.SetInputShape("images", runtimeShape)
        .CopyInputFromHost("images", inputValues, runtimeShape);
bindings.AllocateDeviceBuffer("output0", outputShape);
bindings.BindAll();

TensorRtExecutionContextReadiness readiness = bindings.GetReadiness(runShapeInference: true);
if (!readiness.IsReadyForEnqueue)
{
    throw new InvalidOperationException(readiness.ToString());
}

bindings.EnqueueAsync(stream, synchronize: false, runShapeInference: false);
float[] outputValues = bindings.ReadOutputSingles("output0", outputElementCount);
```

实际方法签名以当前包版本和样例为准。关键要求是：动态输入形状必须落在构建 Profile 范围内，所有必需输入输出地址都已设置，并且异步执行期间对应 Buffer 不得释放或复用。

## 4. Readiness 诊断比直接 Enqueue 更重要

`GetReadiness()` 用于在提交推理前集中检查缺失条件。它可以把“输入形状未设置”“张量地址缺失”“输出形状尚不可解析”等问题提前转化为可读诊断，而不是等到 `EnqueueAsync` 返回失败后再猜测。

建议在开发、集成测试和故障排查阶段始终打印 readiness 结果；在性能稳定后，也可以保留失败分支，避免生产日志只剩一个布尔返回值。

```text
Readiness verification
- input shapes specified: yes
- tensor addresses specified: yes
- output shapes resolved: yes
- ready for enqueue: yes
```

## 5. 动态输出与内存大小

对于动态输出，不能在设置输入形状之前按构建时占位维度分配内存。应先设置本次输入形状，查询 Context 解析后的输出维度，再按元素数量和数据类型计算字节数。

内存计算应同时检查：维度中是否仍有负值、乘法是否溢出、数据类型字节数是否匹配、多个输出是否分别分配，以及 Host 与 Device 拷贝方向是否正确。

## 6. Context、设备内存与并发

同一个 Engine 可以创建多个 ExecutionContext，但每个并发执行单元应拥有独立的运行状态、绑定地址和必要的 Context Device Memory。不要在异步任务之间共享一块仍在使用的输入或输出 Buffer。

并发设计通常采用以下单位：

| 并发资源 | 建议归属 |
|---|---|
| ExecutionContext | 每个并发执行槽独立持有 |
| CUDA Stream | 每个执行槽独立持有，或由明确的调度器管理 |
| 输入输出设备内存 | 覆盖该 Stream 上的全部未完成任务 |
| Engine | 可供多个 Context 共享，但释放晚于所有 Context |

## 7. 源码与样例入口

```text
推理绑定样例：
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/samples/Inference/01.Bindings

动态形状样例：
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/samples/Inference/02.DynamicShapes

TensorRT 托管封装源码：
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/src/JYPPX.TensorRtSharp
```

## 8. 验证命令与判定

```powershell
$tensorRtRoot = '<TensorRT-10.11-root>'
$env:PATH = (Join-Path $tensorRtRoot 'lib') + ';' + $env:PATH
$env:JYPPX_NATIVE_BRIDGE_PATH = '<trt10-cuda12-bridge>\jyppxtrtbridge.dll'
dotnet .\samples\Inference\01.Bindings\bin\Debug\net8.0\InferenceBindings.dll `
  --tensor-rt-line 10 --batch 2
```

可接受结果应同时满足：进程退出码为 `0`、readiness 为可执行状态、`EnqueueAsync` 成功、Stream 完成同步、输出张量形状和字节数符合模型契约。仅成功反序列化 Engine 不能证明推理链路正确。

### 8.1 2026-08-13 实机输出

环境与 Loader 审计和 API-001 相同，实际加载 TensorRT 10.11/CUDA 12 Bridge。稳定托管包版本为 4.0.0；运行源码基线为 `d514fe91`，工作树存在未提交的稳定包消费兼容调整，因此这条证据分类为源码树/稳定包合成运行，不提升为干净独立消费者证明。

```text
InferenceBindings TensorRtLine=10 TRT=10.11.0 CUDA=12.9 Batch=2
BindingReport Ready=True Inputs=1 Outputs=1
Readiness Ready=True Bound=True ActiveProfile=0
Execution profile=0 bound=2 synchronized=False ready=True ElapsedMs=2.559 OutputMatch=True
input Input Float shape=[2, 4] bytes=32 bound=True
output Output Float shape=[2, 4] bytes=32 bound=True
InferenceBindings Passed=True
ProcessExitCode=0
```

`synchronized=False` 表示 `EnqueueAsync` 本身没有请求同步，不代表读回发生在 GPU 完成之前；样例在读取输出时完成必要同步。该结果证明两个张量完成地址绑定、GPU enqueue 和 FP32 输出一致性，不证明外部业务模型精度或并发性能。机器可读记录见 `api-runtime-evidence-20260813.json`：<https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/docs/articles/zh-cn/04-api/api-runtime-evidence-20260813.json>。

<img src="../../../../images/inference-bindings-runtime-terminal.png" alt="InferenceBindings 示例在 TensorRT 10 与 CUDA 12 环境中的真实终端记录" width="640" style="display:block;max-width:100%;height:auto;margin:16px auto;" />

## 9. 小结

Runtime、Engine 和 Context 分别对应加载、不可变执行计划和可变执行状态。推理绑定不是简单地传入几个指针，而是一个由形状、内存、地址、Stream 和生命周期共同组成的运行期契约。

<!-- public-article-declaration:start -->
## 10. 文章声明

### 10.1 开源协议声明
作者所有开源项目代码均遵循 Apache License 2.0 开源协议。
特别说明：本项目集成了若干第三方库。若任何第三方库的许可协议与 Apache 2.0 协议存在冲突或不一致，均以该第三方库的原始许可协议为准。本项目不包含也不代表这些第三方库的授权声明，使用前请务必阅读并遵守第三方库的相关许可。

### 10.2 代码开发与质量说明
AI 辅助开发：本代码在开发过程中使用了人工智能（AI）辅助生成与优化，并非完全由人工逐行编写。
安全性承诺：作者郑重声明，本代码中绝无任何有意设置的后门、病毒、木马或旨在破坏用户设备、窃取数据的恶意代码。
技术局限性：受限于作者个人的技术水平与能力，代码中可能存在因逻辑不严谨、优化不足或经验欠缺导致的低级问题（例如但不限于内存泄漏、偶发崩溃、资源未释放等）。这些问题纯属能力不足所致，并非主观故意。
测试范围：由于作者精力有限，未对本软件进行全方位、覆盖所有边缘场景的完整测试。

### 10.3 免责声明（重要）
请在将本代码应用于任何实际项目（特别是商业、工业或关键任务环境）之前，务必进行详尽、严格的自行测试与验证。 鉴于上述可能存在的代码缺陷及测试覆盖不足，因使用本代码而导致的任何直接或间接损失（包括但不限于设备故障、数据丢失、系统瘫痪或利润损失等），本作者概不负责。 一旦您开始使用本代码，即表示您已知晓上述风险并同意自行承担一切后果，相关问题与本作者无关。

### 10.4 代码开源范围
本项目承诺核心逻辑代码完全开源，但上述提到的“第三方库”的二进制文件、源代码或相关资源不在本项目的开源义务范围内，请根据其各自的指引获取。

### 10.5 社区与反馈
尽管存在上述不足，我们仍欢迎大家下载使用、提交 Issue 或参与测试，共同完善项目。如果您在使用过程中发现 Bug、内存溢出或有改进建议，欢迎通过项目主页提供的联系方式与作者取得联系，我们将尽力在有限的时间内提供协助。

<img src="../../../../images/personal-contact-banner-v6-zh.png" alt="作者联系方式" width="640" style="display:block;max-width:100%;height:auto;margin:16px auto;" />
<!-- public-article-declaration:end -->
