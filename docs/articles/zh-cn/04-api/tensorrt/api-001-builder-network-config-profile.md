# TensorRT CSharp API v4.0 Builder：Network、BuilderConfig 与 OptimizationProfile

<!-- public-article-layout:start -->
<style>
.content article { min-width: 0; overflow-wrap: anywhere; }
.content article a, .content article :not(pre) > code { overflow-wrap: anywhere; word-break: break-word; }
.content article pre:not(.mermaid), .content article table { display: block; max-width: 100%; overflow-x: auto; }
.content pre.mermaid { max-width: 640px; margin: 16px auto; overflow-x: auto; }
.content pre.mermaid svg { display: block; width: 100%; max-width: 640px; height: auto; }
</style>
<!-- public-article-layout:end -->

> 文章编号：API-001；适用版本：4.0.0；当前状态：review。

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

本文聚焦 TensorRT 构建阶段的四个核心对象：`TensorRtBuilder`、`TensorRtNetworkDefinition`、`TensorRtBuilderConfig` 与 `TensorRtOptimizationProfile`。目标不是只展示一次 `BuildSerializedNetwork` 调用，而是说明静态网络、动态维度、构建配置和最终序列化 Engine 之间的责任边界。

## 2. 对象关系与生命周期

| 对象 | 主要职责 | 生命周期建议 |
|---|---|---|
| `TensorRtBuilder` | 创建 Network、Config 和 Profile，触发 Engine 构建 | 覆盖一次或一组构建任务 |
| `TensorRtNetworkDefinition` | 描述输入、层、输出和张量维度 | 在 Engine 构建完成后释放 |
| `TensorRtBuilderConfig` | 配置工作区、精度、Tactic、Profile 等构建策略 | 与本次构建过程绑定 |
| `TensorRtOptimizationProfile` | 为动态输入定义最小、最优和最大维度 | 加入 Config 后由构建流程使用 |
| 序列化网络 | 构建产物，可保存为 Engine 文件 | 由调用方持有并落盘 |

托管对象仍然包装原生资源，因此建议用 `using` 明确释放顺序。Network、Config 和 Profile 不应被理解为 Engine 运行期的长期依赖；运行期应从序列化结果创建 Runtime、Engine 和 ExecutionContext。

## 3. 最小静态网络构建流程

下面的骨架展示构建阶段的最小闭环。具体层定义可替换为 ONNX Parser 或手工 Network API。

```csharp
TensorRtApiLine line = TensorRtApiLine.TensorRt10;
using TensorRtLogger logger = new TensorRtLogger(line);
using TensorRtBuilder builder = new TensorRtBuilder(logger);
using TensorRtNetworkDefinition network = builder.CreateNetwork(stronglyTyped: false);
using TensorRtBuilderConfig config = builder.CreateBuilderConfig();

config.SetMemoryPoolLimit(
    TensorRtMemoryPoolType.Workspace,
    256UL * 1024UL * 1024UL);

// 在这里创建输入、网络层并标记输出。
using TensorRtHostMemory serializedNetwork = builder.BuildSerializedNetwork(network, config);
File.WriteAllBytes("model.engine", serializedNetwork.ToArray());
```

构建失败时，应先检查 Logger 输出、Network 是否存在未解析维度、输出是否已经标记，以及当前 TensorRT 版本是否支持所选配置。不要只根据序列化对象是否为空判断根因。

## 4. 动态维度与 OptimizationProfile

当输入包含 `-1` 动态维度时，必须为每个动态输入提供 Profile。三个维度集合分别表达可接受下界、常用最优形状和可接受上界。

```csharp
using TensorRtOptimizationProfile profile = builder.CreateOptimizationProfile();

profile.SetShape(
    "images",
    new TensorRtDims(new[] { 1, 3, 320, 320 }),
    new TensorRtDims(new[] { 1, 3, 640, 640 }),
    new TensorRtDims(new[] { 4, 3, 1280, 1280 }));

config.AddOptimizationProfile(profile);
```

Profile 的输入名称必须与 Network 输入名称完全一致；最小、最优、最大维度的秩必须相同，并满足逐维 `Min <= Opt <= Max`。Engine 能否接受某个输入形状，由构建时 Profile 决定，而不是由运行期临时放宽。

## 5. BuilderConfig 的设置与读回

建议把“写入配置”和“读回配置”作为同一段诊断流程。对工作区限制、构建 Flag、Profiling Verbosity、Tactic Source 和 Optimization Level 等选项，应在构建前输出最终值，避免日志只记录调用意图。

```text
BuilderConfig verification
- Workspace bytes: 268435456
- Optimization profiles: 1
- Network input: images[-1,3,-1,-1]
- Build result: serialized network created
```

不同 TensorRT 大版本对可用属性和默认值存在差异。跨版本代码应优先调用 TensorRT CSharp API v4.0 已封装的统一入口，并以实际读回值和 Logger 输出作为版本差异证据。

## 6. 源码与样例入口

```text
Builder 封装：
https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/src/JYPPX.TensorRtSharp/Builder/TensorRtBuilder.cs

动态维度样例：
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/samples/Inference/02.DynamicShapes

ONNX 转 Engine 应用：
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/applications/OnnxToEngine
```

## 7. 结果判定与边界

完成构建验证时，至少应保留以下证据：构建命令退出码为 `0`、Logger 中不存在致命错误、序列化产物大小大于 `0`、动态输入 Profile 可被读回，并且生成的 Engine 能在匹配的 Runtime 环境中反序列化。

本文的代码与接口关系已按仓库源码和现有样例复核。由于 Engine 构建结果受 GPU、TensorRT、CUDA 和模型共同影响，文章状态保持 `review`；发布前还需在目标环境补充一次真实模型构建记录。

## 8. 小结

构建阶段的关键不是把所有选项堆进 BuilderConfig，而是让 Network、Profile 和配置形成可检查的输入契约。静态网络可以省略 Profile；动态网络必须先固定允许范围，再生成供运行期消费的 Engine。

<!-- public-article-declaration:start -->
## 9. 文章声明

### 9.1 开源协议声明
作者所有开源项目代码均遵循 Apache License 2.0 开源协议。
特别说明：本项目集成了若干第三方库。若任何第三方库的许可协议与 Apache 2.0 协议存在冲突或不一致，均以该第三方库的原始许可协议为准。本项目不包含也不代表这些第三方库的授权声明，使用前请务必阅读并遵守第三方库的相关许可。

### 9.2 代码开发与质量说明
AI 辅助开发：本代码在开发过程中使用了人工智能（AI）辅助生成与优化，并非完全由人工逐行编写。
安全性承诺：作者郑重声明，本代码中绝无任何有意设置的后门、病毒、木马或旨在破坏用户设备、窃取数据的恶意代码。
技术局限性：受限于作者个人的技术水平与能力，代码中可能存在因逻辑不严谨、优化不足或经验欠缺导致的低级问题（例如但不限于内存泄漏、偶发崩溃、资源未释放等）。这些问题纯属能力不足所致，并非主观故意。
测试范围：由于作者精力有限，未对本软件进行全方位、覆盖所有边缘场景的完整测试。

### 9.3 免责声明（重要）
请在将本代码应用于任何实际项目（特别是商业、工业或关键任务环境）之前，务必进行详尽、严格的自行测试与验证。 鉴于上述可能存在的代码缺陷及测试覆盖不足，因使用本代码而导致的任何直接或间接损失（包括但不限于设备故障、数据丢失、系统瘫痪或利润损失等），本作者概不负责。 一旦您开始使用本代码，即表示您已知晓上述风险并同意自行承担一切后果，相关问题与本作者无关。

### 9.4 代码开源范围
本项目承诺核心逻辑代码完全开源，但上述提到的“第三方库”的二进制文件、源代码或相关资源不在本项目的开源义务范围内，请根据其各自的指引获取。

### 9.5 社区与反馈
尽管存在上述不足，我们仍欢迎大家下载使用、提交 Issue 或参与测试，共同完善项目。如果您在使用过程中发现 Bug、内存溢出或有改进建议，欢迎通过项目主页提供的联系方式与作者取得联系，我们将尽力在有限的时间内提供协助。

<img src="../../../../images/personal-contact-banner-v6-zh.png" alt="作者联系方式" width="640" style="display:block;max-width:100%;height:auto;margin:16px auto;" />
<!-- public-article-declaration:end -->
