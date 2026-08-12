# TensorRT CSharp API v4.0 TensorRtExec CLI 参数指南：从 trtexec 迁移到 C# 工具

<!-- public-article-layout:start -->
<style>
.content article { min-width: 0; overflow-wrap: anywhere; }
.content article a, .content article :not(pre) > code { overflow-wrap: anywhere; word-break: break-word; }
.content article pre:not(.mermaid), .content article table { display: block; max-width: 100%; overflow-x: auto; }
.content pre.mermaid { max-width: 640px; margin: 16px auto; overflow-x: auto; }
.content pre.mermaid svg { display: block; width: 100%; max-width: 640px; height: auto; }
</style>
<!-- public-article-layout:end -->

> 文章编号：`APP-EXEC-003`；适用版本：TensorRT CSharp API v4.0 `4.0.0`；当前状态：`review`。

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
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/applications
```

运行示例必须同时给出程序输出和判定标准。终端中的 `status`、`Ready`、`Bound`、`enqueueCount`、`OutputMatch`、进程退出码、报告文件或结果图片分别说明不同层次的事实；只有明确写出这些结果，读者才能区分程序启动、Engine 构建、GPU enqueue 和业务结果语义。
<!-- public-article-project-preface:end -->

NVIDIA `trtexec` 是 TensorRT 模型构建和性能诊断的重要工具。对于 .NET 项目，团队往往还需要把同一组模型路径、Shape、精度、运行输入和报告字段纳入 C# 应用、自动化脚本或桌面工具。TensorRT CSharp API v4.0：<https://github.com/guojin-yan/TensorRT-CSharp-API> 中的 `TensorRtExec` 提供了 trtexec 风格 CLI，并让它与 WinForms 共用同一参数模型和执行服务。

TensorRtExec 不是对 NVIDIA `trtexec` 的逐字复制。它保留熟悉的参数命名，同时为当前 C# API 已实现的能力提供构建、受限运行、输出捕获和 JSON/Markdown 报告；尚未形成真实实现的选项会明确留在 parse-only 或 diagnostic 状态。

### 1.2 项目、包与源码入口

| 项目 | 作用 | 链接 |
| --- | --- | --- |
| TensorRT CSharp API v4.0 | TensorRT 与 CUDA 的 C# API | GitHub 项目：<https://github.com/guojin-yan/TensorRT-CSharp-API> |
| 核心接口包 | .NET 托管接口 | `JYPPX.TensorRT.CSharp.API 4.0.0`：<https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/4.0.0> |
| Runtime Bridge | 匹配本机 TensorRT/CUDA/cuDNN | JYPPX NuGet 包列表：<https://www.nuget.org/profiles/JYPPX> |
| TensorRtExec | CLI 与 WinForms 源码 | 应用目录：<https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/applications/TensorRtExec> |
| CLI 入口 | help、参数解析和退出码 | `TensorRtExecCommand.cs`：<https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/applications/TensorRtExec/Console/TensorRtExecCommand.cs> |
| 参数模型 | CLI/GUI 共享参数 | `TensorRtExecOptions.cs`：<https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/applications/TensorRtExec/Core/TensorRtExecOptions.cs> |
| 兼容矩阵 | 参数实现状态与差异 | `tensor-rt-exec-trtexec-parity-matrix.md`：<https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.md> |

## 2. 启动与帮助

显示人工可读帮助：

```powershell
dotnet run --project .\applications\TensorRtExec -- --help
```

显示机器可读能力表：

```powershell
dotnet run --project .\applications\TensorRtExec -- --help-json
```

`--capabilities-json` 是同类入口。机器可读表适合工具生成器和版本差异审计，但它只描述参数能力面，不执行 TensorRT。

程序退出码约定：

| 退出码 | 含义 |
| --- | --- |
| `0` | 当前请求成功，或受控依赖探测按设计跳过 |
| `2` | 参数无效、reference mismatch 或报告判定失败 |
| `1` | 其他未归类运行错误 |

自动化脚本应同时检查退出码和报告 `State/Success`，不能只搜索控制台中的单个关键词。

## 3. 最小构建命令

静态 ONNX：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --tensor-rt-line 10 `
  --onnx .\models\model.onnx `
  --saveEngine .\work\model.plan `
  --workspace 512 `
  --builderOptimizationLevel 3 `
  --buildOnly `
  --exportReport .\work\model-build-report.json
```

动态 ONNX：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --tensor-rt-line 10 `
  --onnx .\models\model-dynamic.onnx `
  --saveEngine .\work\model-dynamic.plan `
  --minShapes images:1x3x320x320 `
  --optShapes images:1x3x640x640 `
  --maxShapes images:4x3x960x960 `
  --fp16 `
  --workspace 1024MiB `
  --buildOnly `
  --exportReport .\work\model-dynamic-report.json
```

## 4. 参数命名与常用别名

TensorRtExec 优先兼容常见 trtexec 拼写，同时保留仓库早期命令别名：

| 主要参数 | 兼容别名 | 作用 |
| --- | --- | --- |
| `--onnx` | `--model`、`--onnxFile` | 输入 ONNX |
| `--saveEngine` | `--save-engine`、`--plan`、`--engineFile` | 保存 Engine |
| `--loadEngine` | `--load-engine` | 加载 Engine |
| `--shapes` | `--inputShapes` | 固定运行 Shape 或 profile 别名 |
| `--timingCacheFile` | `--timingCache` | 导入 Timing Cache |
| `--exportReport` | `--report` | 导出 JSON/Markdown 报告 |
| `--dryRun` | `--previewOnly` | 只检查参数和预览 |
| `--plugins` | `--plugin`、`--dynamicPlugins`、`--setPluginsToSerialize` | plugin 路径诊断字段 |

别名会被归一化，报告保留 `NormalizedCommandLine` 和 SHA256，方便比较 GUI、CLI 与 CI 是否使用了同一组参数。

## 5. 输入、Engine 与 Shape 参数

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `--onnx` | 文件路径 | 构建新 Engine |
| `--saveEngine` | `.plan` 路径 | 保存 serialized Engine |
| `--loadEngine` | `.plan` 路径 | 加载已有 Engine |
| `--minShapes` | `name:dims` | Profile 最小 Shape |
| `--optShapes` | `name:dims` | Builder 优化 Shape |
| `--maxShapes` | `name:dims` | Profile 最大 Shape |
| `--shapes` | `name:dims` | 运行 Shape 或三元组别名 |
| `--batch` | 正整数 | 兼容批量参数，不能替代多输入显式合同 |

多输入示例：

```text
--shapes left:2x4,right:2x4
```

动态 profile 中的三份 map 必须覆盖同一批 tensor，并保持每个 tensor 的 rank 和逐维次序合法。

## 6. 构建和精度参数

### 6.1 已连接 Builder 的常用参数

| 参数 | 说明 |
| --- | --- |
| `--workspace` | Workspace MiB，支持 `MiB/GiB` 后缀 |
| `--fp16` | 请求 FP16 Builder flag |
| `--bf16` | 请求 BF16，受 API line 与硬件限制 |
| `--noTF32` | 关闭默认 TF32 策略 |
| `--builderOptimizationLevel` | Builder 优化等级 0..5 |
| `--maxAuxStreams` | Engine 辅助流上限 |
| `--memPoolSize` | `workspace:512,tacticDram:1024` 等 memory pool 配置 |
| `--tacticSources` | tactic 来源策略 |
| `--inputIOFormats` / `--outputIOFormats` | Tensor I/O 类型与格式约束 |

### 6.2 需要谨慎解读的参数

| 参数 | 当前边界 |
| --- | --- |
| `--int8` / `--calib` | 参数与报告已接入，不等于 calibrator 和精度验证完成 |
| `--fp8` / `--best` | 保持 parse/report 边界，不能宣称通用支持 |
| `--precisionConstraints` | TRT8/10 可设置并回读；TRT11 对已移除能力保持版本保护 |
| `--layerPrecisions` / `--layerOutputTypes` | TRT8/10 可匹配层并回读；TRT11 不伪造已移除 setter |
| `--plugins` | 当前是路径与序列化诊断，不代表动态库已完成 load/register 生命周期 |

### 6.3 I/O 与逐层精度示例

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --saveEngine .\work\model-policy.plan `
  --inputIOFormats fp32:chw `
  --outputIOFormats fp32:chw `
  --precisionConstraints prefer `
  --layerPrecisions "encoder*:fp16,encoder.norm:fp32" `
  --layerOutputTypes "head*:fp32" `
  --profilingVerbosity detailed `
  --exportLayerInfo .\work\model-layer-info.json `
  --buildOnly `
  --exportReport .\work\model-policy-report.json
```

精确层名优先于 wildcard；同一优先级下，靠后的规则覆盖靠前规则。没有匹配到任何层时应失败，避免模型改名后旧规则被静默忽略。

## 7. Runtime 与性能参数

| 参数 | 作用 |
| --- | --- |
| `--iterations` | 至少执行的 measurement rounds |
| `--warmUp` | 预热毫秒数 |
| `--duration` | 最低测量秒数 |
| `--streams` | 兼容并发流参数 |
| `--infStreams` | 有效 inference stream 数，优先于 `--streams` |
| `--threads` | 布尔开关；每个有效 stream 使用独立 host driver thread |
| `--avgRuns` | 每组平均的连续样本数 |
| `--percentile` | 输出指定百分位耗时 |
| `--useCudaGraph` | 请求捕获并启动 CUDA Graph，失败时记录回退原因 |
| `--useSpinWait` | 使用 CUDA Event readiness 主动等待 |
| `--sleepTime` | 含 `EnqueueDelay` 的源码构建会在同步 stream 中排入一次延迟并扇出；稳定包 `4.0.0` 消费模式下请求可解析，但 applied 值为 0 |
| `--idleTime` | measurement rounds 之间的 host idle gap |
| `--noDataTransfers` | 只测 enqueue，关闭输入 H2D 与输出 D2H |

这些选项只有在 compatible float Engine 的 bounded runtime 路径中才会真正执行。`--buildOnly` 下出现同名参数，并不产生性能数据。

## 8. 输入、输出与 Reference 校验

TensorRtExec 的通用运行路径支持多个 float input 和多个 float output：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --loadEngine .\work\add-sub.plan `
  --shapes left:2x4,right:2x4 `
  --loadInputs "left:.\inputs\left.bin,right:.\inputs\right.bin" `
  --referenceOutputs "sum:.\refs\sum.json,difference:.\refs\difference.json" `
  --referenceAbsTolerance 1e-5 `
  --referenceRelTolerance 1e-4 `
  --referenceNaNPolicy reject `
  --referenceInfinityPolicy exact `
  --dumpOutput `
  --exportOutput .\work\output.json `
  --dumpRawBindingsToFile .\work\output.raw `
  --exportReport .\work\run-report.json
```

完整覆盖是硬性条件：输入映射缺失、重复或存在未知名称会失败；reference 也必须覆盖全部 Engine output。校验顺序为名称、Shape、元素数量和逐值比较。

有限值满足下列任一条件即通过：

```text
abs(actual - expected) <= absoluteTolerance
abs(actual - expected) <= relativeTolerance * max(abs(actual), abs(expected))
```

`NaN` 默认拒绝；`equal` 只允许两边同时为 NaN。Infinity 的 `exact` 要求符号一致，`reject` 则拒绝任意 Infinity。

## 9. 输出和诊断参数

| 参数 | 产物 | 说明 |
| --- | --- | --- |
| `--exportReport` | JSON/Markdown | 最完整的配置、状态和边界记录 |
| `--exportOutput` | JSON | 所有捕获 output 的摘要、预览与哈希 |
| `--dumpRawBindingsToFile` | raw + manifest | 按 Engine 顺序保存原始 bytes 与偏移 |
| `--exportTimes` | JSON | benchmark timing 样本与统计 |
| `--exportLayerInfo` | JSON/文本 | Engine Inspector 层信息 |
| `--timingCacheFile` | cache input | 导入 Timing Cache |
| `--exportTimingCache` | cache output | 成功构建后导出 Timing Cache |
| `--evidenceSidecar` | JSON | 补充模型来源和资产信息，不提高结果等级 |

## 10. 如何判断参数是否真的执行

每份报告都会把参数分层：

| 字段 | 正确解读 |
| --- | --- |
| `ParsedOptions` | Parser 已接收，不代表 TensorRT 已执行 |
| `AppliedOptions` | 本次路径实际调用并获得必要 readback |
| `ParseOnlyOptions` | 只记录意图或当前版本不支持 |
| `CapabilityProbe` | API/依赖可见性探测，不是模型运行 |
| `BenchmarkSummary` | 只有真实 timing 样本时才有运行意义 |
| `ReferenceValidation` | 全部 output 完成比较后才可能通过 |

迁移官方命令时不要一次复制全部选项。先运行 `--help-json`，再以报告中的 `AppliedOptions` 为依据逐项增加配置。

## 11. 一条外部模型验证记录

2026-08-10 的已登记记录使用 YOLOv8n-cls ONNX、预处理后的 `images:[1,3,224,224]` 输入和独立 ONNX Runtime CPU reference，在 TensorRT 10.11 / CUDA 12.9 中完成：

| 检查项 | 结果 |
| --- | --- |
| Parser / Engine round-trip | 通过 |
| 输入 / 输出 | `images:[1,3,224,224]` / `output0:[1,1000]` |
| Reference 元素 | 1000 |
| mismatch | 0 |
| 最大绝对误差 | `5.364418e-7` |
| 最大相对误差 | `1.1431351e-5` |
| I/O format | `fp32:chw`，请求、应用、回读一致 |
| Layer policy | TRT10 目标卷积层 precision/output type 匹配并回读 |
| Inspector | 87 个 Engine layers，包含实际 I/O datatype/format 与 tactic |

该运行证明这组外部模型、输入和 reference 能走通 TensorRtExec 的构建与校验路径。工具仍把通用运行保守分类为 `synthetic-input-runtime`；图像类别语义由 YoloVision 的模型专属记录负责。

## 12. 与官方 trtexec 的主要差异

- 参数名称相似，但并非所有官方选项都已实现。
- TensorRtExec 强调 JSON/Markdown 报告、输入输出哈希和 GUI/CLI 共用参数。
- 部分 TensorRT 8/10 API 在 TensorRT 11 已移除，工具会版本保护而不是发送旧编号。
- plugin、完整 INT8 calibrator、FP8/best 和部分 debug/refit 诊断不能按字符串存在推断为运行完成。
- TensorRtExec 的 bounded runtime 面向受控 float tensor；模型业务预处理和后处理仍由专属应用承担。

## 13. 当前源码复核状态

2026-08-12 已使用稳定核心包 `4.0.0` 完成 TensorRtExec Release 构建和 `--help` 验证，结果为 0 警告、0 错误、退出码 0。本文涉及的参数组合尚未在本轮逐项执行，历史运行数据不冒充当前工作树重新执行结果，因此状态保持 `review`。

## 14. 总结

从官方 trtexec 迁移到 TensorRtExec，最重要的不是把参数名逐字替换，而是确认每个选项在当前 TensorRT line 和当前执行模式中的真实状态。使用 `--help-json` 了解能力面，用 `AppliedOptions` 核对执行，用 reference output 验证数值，再把模型任务语义交给专属 Runner，才能得到可维护的 .NET TensorRT 自动化流程。

<!-- public-article-declaration:start -->
## 15. 文章声明

### 15.1 开源协议声明
作者所有开源项目代码均遵循 Apache License 2.0 开源协议。
特别说明：本项目集成了若干第三方库。若任何第三方库的许可协议与 Apache 2.0 协议存在冲突或不一致，均以该第三方库的原始许可协议为准。本项目不包含也不代表这些第三方库的授权声明，使用前请务必阅读并遵守第三方库的相关许可。

### 15.2 代码开发与质量说明
AI 辅助开发：本代码在开发过程中使用了人工智能（AI）辅助生成与优化，并非完全由人工逐行编写。
安全性承诺：作者郑重声明，本代码中绝无任何有意设置的后门、病毒、木马或旨在破坏用户设备、窃取数据的恶意代码。
技术局限性：受限于作者个人的技术水平与能力，代码中可能存在因逻辑不严谨、优化不足或经验欠缺导致的低级问题（例如但不限于内存泄漏、偶发崩溃、资源未释放等）。这些问题纯属能力不足所致，并非主观故意。
测试范围：由于作者精力有限，未对本软件进行全方位、覆盖所有边缘场景的完整测试。

### 15.3 免责声明（重要）
请在将本代码应用于任何实际项目（特别是商业、工业或关键任务环境）之前，务必进行详尽、严格的自行测试与验证。 鉴于上述可能存在的代码缺陷及测试覆盖不足，因使用本代码而导致的任何直接或间接损失（包括但不限于设备故障、数据丢失、系统瘫痪或利润损失等），本作者概不负责。 一旦您开始使用本代码，即表示您已知晓上述风险并同意自行承担一切后果，相关问题与本作者无关。

### 15.4 代码开源范围
本项目承诺核心逻辑代码完全开源，但上述提到的“第三方库”的二进制文件、源代码或相关资源不在本项目的开源义务范围内，请根据其各自的指引获取。

### 15.5 社区与反馈
尽管存在上述不足，我们仍欢迎大家下载使用、提交 Issue 或参与测试，共同完善项目。如果您在使用过程中发现 Bug、内存溢出或有改进建议，欢迎通过项目主页提供的联系方式与作者取得联系，我们将尽力在有限的时间内提供协助。

<img src="../../../../images/personal-contact-banner-v6-zh.png" alt="作者联系方式" width="640" style="display:block;max-width:100%;height:auto;margin:16px auto;" />
<!-- public-article-declaration:end -->
