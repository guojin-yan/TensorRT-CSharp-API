# TensorRT CSharp API v4.0 CUDA：Device、Memory、Stream、Event 与 Graph

<!-- public-article-layout:start -->
<style>
.content article { min-width: 0; overflow-wrap: anywhere; }
.content article a, .content article :not(pre) > code { overflow-wrap: anywhere; word-break: break-word; }
.content article pre:not(.mermaid), .content article table { display: block; max-width: 100%; overflow-x: auto; }
.content pre.mermaid { max-width: 640px; margin: 16px auto; overflow-x: auto; }
.content pre.mermaid svg { display: block; width: 100%; max-width: 640px; height: auto; }
</style>
<!-- public-article-layout:end -->

> 文章编号：API-003；适用版本：4.0.0；当前状态：ready。

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

TensorRT 推理最终运行在 CUDA 资源之上。本文按 `Device -> Memory -> Stream -> Event -> Graph` 的顺序整理 TensorRT CSharp API v4.0 中 CUDA 侧的常用能力，并强调异步调用真正完成之前的资源所有权。

## 2. Device 与当前上下文

进程启动后应先确认设备数量、目标设备编号和设备属性，再进入内存分配或 Engine 加载。多 GPU 程序必须让创建 Engine、Context、Stream 和 Device Memory 的线程处于预期设备上下文中。

```text
Device verification
- device count: greater than 0
- selected device: explicit
- compute capability: compatible with the Engine
- free memory: sufficient for Engine and buffers
```

Engine 通常不能跨不兼容的 GPU 架构直接复用。设备可见不等于 Engine 可执行，仍需结合构建目标、TensorRT 版本和插件依赖判断。

## 3. Host 与 Device Memory

设备内存负责 TensorRT 输入、输出和工作区；Host 内存负责应用侧数据。采用异步拷贝时，页锁定 Host Memory 通常更适合稳定吞吐，但也应限制分配数量并及时释放。

一个可靠的内存流程包括：

1. 根据解析后的张量形状计算元素数。
2. 根据数据类型计算总字节数并检查溢出。
3. 分配 Host 与 Device Memory。
4. 在同一 Stream 上排入 Host-to-Device、推理和 Device-to-Host。
5. 等待 Event 或同步 Stream 后再读取或释放结果。

包装对象释放只代表托管侧发出了释放动作；如果设备上仍有未完成任务，提前释放同样会造成未定义行为。

## 4. Stream 表达执行顺序

同一 Stream 内的任务按提交顺序执行，不同 Stream 之间默认不存在应用所需的业务依赖。多流能否提升吞吐，取决于拷贝、Kernel 和 TensorRT 执行是否存在可重叠空间，而不是 Stream 数量越多越好。

```text
stream-0: H2D(batch-0) -> inference(batch-0) -> D2H(batch-0)
stream-1: H2D(batch-1) -> inference(batch-1) -> D2H(batch-1)
```

每条并发链应有独立或明确分片的 Buffer、ExecutionContext 和完成信号。共享 Engine 通常可行，共享仍在执行的 Context 或 Buffer 则会破坏状态隔离。

## 5. Event 用于依赖与计时

CUDA Event 可以记录在某条 Stream 上，并由 Host 或另一条 Stream 等待。它适合表达“生产者已经完成，消费者可以继续”，也可用于测量 GPU 时间。

计时时应把起止 Event 记录在实际工作所在的 Stream 上，并等待结束 Event 完成。Host 侧秒表包含线程调度、提交和同步开销，与 Event 计时代表的 GPU 时间不是同一指标。

## 6. Graph 捕获的适用条件

CUDA Graph 适合重复执行、拓扑稳定的操作序列。捕获前应先完成容易触发延迟初始化的操作，并保证捕获期间使用的 Stream、Context、Memory 和参数地址在 Graph 生命周期内持续有效。

不适合直接捕获的情况包括：每次输入导致不同控制流、Buffer 地址频繁变化、动态形状尚未完成稳定预热，以及调用内部包含不支持捕获的同步行为。

安全的 Graph 生命周期可概括为：

```text
分配长期资源 -> 预热 -> 开始捕获 -> 提交固定序列 -> 结束捕获
-> 实例化 GraphExec -> 重复 Launch -> 等待全部完成 -> 释放 Graph 相关资源
```

## 7. 源码与样例入口

```text
CUDA 托管封装：
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/src/JYPPX.CudaSharp

多流性能样例：
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/samples/Performance/01.MultiStream

CUDA Runtime Compilation 样例：
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/samples/Cuda/01.RuntimeCompilation
```

## 8. 验证命令与判定

```powershell
$env:JYPPX_NATIVE_BRIDGE_PATH = '<trt10-cuda12-bridge>\jyppxtrtbridge.dll'
dotnet .\samples\Performance\01.MultiStream\bin\Debug\net8.0\MultiStream.dll
dotnet .\samples\Cuda\01.RuntimeCompilation\bin\Debug\net8.0\CudaRuntimeCompilation.dll
dotnet run --project .\smoke\CudaGraphSmokeRunner\CudaGraphSmokeRunner.csproj --no-restore
```

验证记录至少应包含 GPU 型号、CUDA 与驱动版本、并发 Stream 数量、每条 Stream 的资源归属、同步方式、退出码和结果校验。性能结论还应包含预热次数、统计口径和单流基线。

### 8.1 2026-08-13 三条实机路径

本次在 RTX 3060 Laptop、驱动 576.02、CUDA 12.9.41 上完成三类运行，三条进程退出码均为 `0`：

```text
MultiStream:
IndependentStreams=True A=True B=True Bytes=4096
CrossStreamWait=True ProducerStream=NonBlocking ConsumerStream=NonBlocking
MultiStream Passed=True

RuntimeCompilation:
capability.version=12.9
launch.succeeded=True gpuReadback=True correctness=True maxAbsoluteError=0
driver.launch.succeeded=True gpuReadback=True correctness=True maxAbsoluteError=0
failure.success=False result=Compilation logLength=1299

CudaGraphSmokeRunner:
CudaGraphCaptureRoundTrip=True Bytes=64 Capture=None->Active->None
MemsetOutput=True
CudaGraphMemoryAllocation Bytes=64 Pattern=0x6B
```

Graph 路径真实覆盖 Capture、Instantiate、Launch、64 字节回读与 Graph Memory Allocation；同时保留版本失败边界：CUDA 12.9 不提供要求 CUDA 13.0 的 Graph ID API，旧 PTDS dependency-update 变体也被报告为版本不支持。本文只据此确认功能正确性，不宣称 Graph 比普通 Stream 更快，因为本批没有运行同环境预热、重复统计和单流对照。详细哈希见 `api-runtime-evidence-20260813.json`：<https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/docs/articles/zh-cn/04-api/api-runtime-evidence-20260813.json>。

<img src="../../../../images/cuda-multistream-runtime-terminal.png" alt="两条 CUDA Stream 与跨流 Event 等待的真实终端记录" width="640" style="display:block;max-width:100%;height:auto;margin:16px auto;" />

<img src="../../../../images/cuda-rtc-runtime-terminal.png" alt="NVRTC 编译、Runtime 和 Driver Kernel 启动与 GPU 读回的真实终端记录" width="640" style="display:block;max-width:100%;height:auto;margin:16px auto;" />

## 9. 小结

CUDA API 的主线不是孤立地记忆函数，而是维护资源所有权和执行依赖。Device 决定运行位置，Memory 承载数据，Stream 定义顺序，Event 表达依赖，Graph 则复用稳定的提交拓扑。

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
