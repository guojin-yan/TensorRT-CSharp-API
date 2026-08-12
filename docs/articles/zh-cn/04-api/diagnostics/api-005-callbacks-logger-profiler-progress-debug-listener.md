# TensorRT CSharp API v4.0 回调：Logger、Profiler、ProgressMonitor 与 DebugListener

<!-- public-article-layout:start -->
<style>
.content article { min-width: 0; overflow-wrap: anywhere; }
.content article a, .content article :not(pre) > code { overflow-wrap: anywhere; word-break: break-word; }
.content article pre:not(.mermaid), .content article table { display: block; max-width: 100%; overflow-x: auto; }
.content pre.mermaid { max-width: 640px; margin: 16px auto; overflow-x: auto; }
.content pre.mermaid svg { display: block; width: 100%; max-width: 640px; height: auto; }
</style>
<!-- public-article-layout:end -->

> 文章编号：API-005；适用版本：4.0.0；当前状态：review。

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

本文从公开接口角度介绍 `TensorRtLogger`、`TensorRtProfiler`、`TensorRtProgressMonitor` 和 `TensorRtDebugListenerCallbackOwner`。四类对象都把 TensorRT native 回调转入托管委托，但触发阶段、借用方和释放顺序不同。正确使用的核心是先解除 native borrower，再释放托管 callback owner。

## 2. 四类回调的职责

| 回调 | 触发阶段 | 典型挂载点 | 主要信息 |
|---|---|---|---|
| Logger | Runtime、Builder、Parser 和 Engine 操作 | 构造相关 TensorRT 对象时传入 | 严重级别与日志文本 |
| ProgressMonitor | Engine 构建 | `TensorRtBuilderConfig` | 构建阶段、步骤和进度 |
| Profiler | ExecutionContext 执行 | `TensorRtExecutionContext` | Layer 名称与执行时间 |
| DebugListener | ExecutionContext 调试张量回调 | `TensorRtExecutionContext` | 已复制的张量名称、形状和调试元数据 |

Logger 常常覆盖整个 TensorRT 对象链；其余三个回调分别绑定 Config 或 Context。它们不是全局事件总线，也不应由短生命周期局部变量隐式持有。

## 3. Logger 的创建与借用关系

Logger 可接收日志委托和最低严重级别。Runtime、Builder、Parser、Refitter 等对象会借用 native logger 指针，封装层会维护 borrower 计数，阻止在借用期间错误释放。

```csharp
using TensorRtLogger logger = new TensorRtLogger(
    TensorRtApiLine.TensorRt10,
    (severity, message) => Console.WriteLine($"[{severity}] {message}"),
    TensorRtLogSeverity.Verbose);

using TensorRtRuntime runtime = new TensorRtRuntime(logger);
using TensorRtBuilder builder = new TensorRtBuilder(logger);
```

委托内部应快速返回。写文件、网络发送或复杂格式化应放入有界队列，由应用线程处理，避免阻塞 TensorRT 调用线程。

## 4. ProgressMonitor 的构建生命周期

ProgressMonitor 挂到 BuilderConfig 后，由 Engine 构建过程调用。构建完成或失败后都应通过 `ClearProgressMonitor()` 解除关系，再释放 Monitor。

```csharp
using TensorRtProgressMonitor progress = new TensorRtProgressMonitor(
    line,
    item => Console.WriteLine(item));

config.SetProgressMonitor(progress);
try
{
    using TensorRtHostMemory plan = builder.BuildSerializedNetwork(network, config);
    plan.SaveToFile(enginePath);
}
finally
{
    config.ClearProgressMonitor();
}
```

Progress 事件只能证明回调被调用和构建阶段推进，不能单独证明 Engine 产物有效。仍需检查构建返回、产物字节数和后续反序列化。

## 5. Profiler 的执行生命周期

Profiler 挂到 ExecutionContext。设置 `EnqueueEmitsProfile=true` 后，推理执行可以触发逐 Layer 计时。清理时先调用 `ClearProfiler()`，再释放 Profiler 和 Context。

```csharp
using TensorRtProfiler profiler = new TensorRtProfiler(
    line,
    record => Console.WriteLine(record));

context.SetProfiler(profiler);
context.EnqueueEmitsProfile = true;
try
{
    context.EnqueueAsync(stream);
    stream.Synchronize();
}
finally
{
    context.ClearProfiler();
}
```

Profiler 数据是本次 Context 执行的性能诊断。没有预热、重复次数、GPU 信息和统计口径时，单次 Layer 时间不应被写成跨机器性能结论。

## 6. DebugListener 与借用张量

DebugListener 用于接收被标记为 Debug 的张量信息。回调期间的 native 张量地址属于 TensorRT，不能保存到回调外。项目封装只保留复制型名称、形状和元数据，并明确报告 `BorrowedPointerExposed=false`。

```csharp
using TensorRtDebugListenerCallbackOwner listener =
    new TensorRtDebugListenerCallbackOwner(
        line,
        metadata =>
        {
            Console.WriteLine($"{metadata.TensorName} {metadata.ShapeSummary}");
            return true;
        });

context.SetDebugListener(listener);
context.SetTensorDebugState(outputName, true);
try
{
    context.EnqueueAsync(stream);
    stream.Synchronize();
}
finally
{
    context.ClearDebugListener();
}
```

Debug tensor 需要 Network 构建阶段正确标记，并由 Context 启用对应状态。只创建 Listener 不会自动产生调试事件。

## 7. 异常隔离与状态诊断

托管委托不能让异常穿过 native ABI。封装会捕获回调异常并记录调用次数、失败次数和最近状态。应用应把 `CallbackFailureCount > 0` 当作明确失败，而不是因为 TensorRT 主调用仍返回就忽略。

```text
Callback verification
- attached while native owner uses callback: yes
- invocation count: greater than 0
- callback failure count: 0
- in-flight callback count before dispose: 0
- native borrower cleared before owner dispose: yes
```

释放前还应确认没有 in-flight callback。对于异步 Enqueue，必须先同步 Stream 或等待完成 Event，再解除 Listener/Profiler 并释放其托管 owner。

## 8. 源码与样例入口

```text
回调封装源码：
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/src/JYPPX.TensorRtSharp/Callbacks

Callback 生命周期样例：
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/samples/Diagnostics/01.CallbackLifecycle

系列案例文章：
https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/docs/articles/zh-cn/02-samples/smp-008-callback-lifecycle.md
```

## 9. 验证命令与判定

```powershell
dotnet run --project .\samples\Diagnostics\01.CallbackLifecycle -- --tensor-rt-line 10
```

可接受结果要求 Logger、ProgressMonitor、Profiler 和 DebugListener 均产生至少一次回调，失败次数为 `0`，ProgressMonitor 在构建后解除，Profiler 与 DebugListener 在 Context 完成后解除，DebugListener 不暴露借用指针，并且推理输出仍匹配。

本文已完成接口和样例复核，状态保持 `review`。TensorRT 8 不具备组合样例要求的全部 Progress/Debug 能力；正式发布前应分别保存 TensorRT 10 和 11 的生命周期输出。

## 10. 小结

四类回调的共同原则是：native 侧只借用 callback owner，托管委托异常不能跨 ABI，异步工作完成后先解除挂载再释放对象。只要释放顺序不明确，回调功能再丰富也不足以安全使用。

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
