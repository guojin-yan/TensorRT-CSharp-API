# 使用 TensorRT CSharp API v4.0 在 C# 中运行 ResNet18 图像分类

<!-- public-article-layout:start -->
<style>
.content article { min-width: 0; overflow-wrap: anywhere; }
.content article a, .content article :not(pre) > code { overflow-wrap: anywhere; word-break: break-word; }
.content article pre:not(.mermaid), .content article table { display: block; max-width: 100%; overflow-x: auto; }
.content pre.mermaid { max-width: 640px; margin: 16px auto; overflow-x: auto; }
.content pre.mermaid svg { display: block; width: 100%; max-width: 640px; height: auto; }
</style>
<!-- public-article-layout:end -->

这篇文章完整演示如何获取 TorchVision ResNet18、转换为 ONNX、把模型暂存在 Git 仓库外、使用已发布的 TensorRT CSharp API v4.0 与 OpenCV NuGet 包生成 C# 预处理 tensor、建立独立 ONNX Runtime 参考并执行真实 TensorRT 推理。最终结果不是模板日志，而是原图上的 Top-5 叠加图和同次运行输出的终端截图。

> 本文是 TensorRT CSharp API v4.0 4.0.0 Samples 系列的 `SMP-009`，对应源码 `samples/ComputerVision/01.Classification`。重点不只是“跑出 Top-5”，还包括模型来源、预处理合同、标签映射、独立参考和结果语义边界。

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
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/samples
```

运行示例必须同时给出程序输出和判定标准。终端中的 `status`、`Ready`、`Bound`、`enqueueCount`、`OutputMatch`、进程退出码、报告文件或结果图片分别说明不同层次的事实；只有明确写出这些结果，读者才能区分程序启动、Engine 构建、GPU enqueue 和业务结果语义。
<!-- public-article-project-preface:end -->

### 1.2 项目简介

TensorRT CSharp API v4.0 是面向 C#/.NET 的 TensorRT 与 CUDA 推理 API；它不绑定某一个模型或视觉任务，而是提供从 ONNX 解析、Engine 构建、tensor 绑定到 CUDA 执行的基础能力。本文用 ResNet18 把这些基础能力连接到用户可见的图像分类结果，展示如何安装公开包、准备模型资产、复现预处理并检查输出。

### 1.3 项目链接与包列表

| 项目内容 | 入口 |
| --- | --- |
| 项目源码 | TensorRT-CSharp-API 4.0 分支：<https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0> |
| 本文案例源码 | samples/ComputerVision/01.Classification：<https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/samples/ComputerVision/01.Classification> |
| 程序入口 | Program.cs：<https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/samples/ComputerVision/01.Classification/Program.cs> |
| 核心 NuGet | JYPPX.TensorRT.CSharp.API 4.0.0：<https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/4.0.0> |
| 图像解码包 | JYPPX.OpenCV.CSharp.API：<https://github.com/guojin-yan/OpenCV-CSharp-API> |
| Runtime Bridge | NuGet 包列表：<https://www.nuget.org/packages?q=+JYPPX.TensorRT.CSharp&includeComputedFrameworks=true&prerel=true&sortby=relevance> |

本文涉及的模型、标签和图片不进入核心 NuGet 包；它们按照来源、许可证和 SHA256 单独管理。代码和命令以 GitHub 分支中的案例为准，包版本固定为 `4.0.0`。

### 1.4 本文结构

按“项目与包 → 模型/标签/图片资产 → ONNX 转换 → C# 预处理 → TensorRT 推理 → ONNX Runtime 对照 → 结果图与语义边界 → 复查”展开。结果图展示的是模型预测，不是人工标注的真实犬种。

## 2. 本文能解决什么问题

1. 固定 ResNet18 权重、TorchVision 版本、ONNX opset、标签和测试图片来源。
2. 用 C# 完成与 TorchVision 权重一致的 resize、center crop、RGB/NCHW 和 ImageNet 归一化。
3. 使用 TensorRT 构建 Engine、绑定输入输出、执行推理并生成 Top-5。
4. 用同一个 C# 预处理 tensor 对比 ONNX Runtime，排除“输入实际不同”的伪对照。
5. 正确解释结果：执行路径一致不代表模型的 Top-1 就是图片真实类别。

## 3. 本文使用的项目与库

本文使用 TensorRT CSharp API v4.0 仓库中的 `samples/ComputerVision/01.Classification`。这个样例负责图片预处理、TensorRT engine 构建与执行、Softmax、Top-K、结构化 JSON、独立参考比较和 SVG 结果图导出。

主要组件如下：

| 组件 | 本文中的职责 |
| --- | --- |
| `JYPPX.TensorRtSharp` | 解析 ONNX、构建 engine、绑定 tensor 并执行推理 |
| `JYPPX.CudaSharp` | 提供 CUDA 设备、内存和 stream 基础能力 |
| `samples/ComputerVision/01.Classification` | 完成图片预处理、Softmax、Top-5、结果 JSON 和可视化 |
| `JYPPX.TensorRT.CSharp.API` 4 系列包 | 提供已发布的 TensorRT 与 CUDA managed API |
| `JYPPX.OpenCV.CSharp.API` | 由项目作者维护，负责 JPEG/PNG/BMP 图片解码 |
| TorchVision `v0.25.0` | 提供 ResNet18 网络定义、权重和 ImageNet-1K 标签 |
| ONNX Runtime CPU | 对同一个 C# float32 输入 tensor 生成独立参考 |
| TensorRT `10.11` | 本文实际运行使用的推理后端 |

运行前需要安装 .NET 8 SDK、与本机 GPU 匹配的 CUDA 和 TensorRT，并按仓库构建说明生成 `jyppxtrtbridge.dll`。CUDA、cuDNN、TensorRT 和 NVRTC 都由用户自行安装，不进入源码包、NuGet 包或 GitHub Release。

## 4. 模型获取与许可证

本文固定使用 TorchVision ResNet18 `IMAGENET1K_V1`：

- 权重地址：`https://download.pytorch.org/models/resnet18-f37072fd.pth`
- TorchVision tag：`v0.25.0`
- 固定 commit：`8ac84ee75afb1c327902156b5336f56ad63b7e2f`
- 权重 SHA256：`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`
- TorchVision 源码许可证：BSD-3-Clause

仓库提供固定来源和哈希校验的获取入口：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File ./eng/Acquire-TorchVisionResNet18OfficialAssets.ps1 `
  -AllowDownload -ExportOnnx -PythonPath python
```

测试图片使用 Wikimedia Commons 上的 `Dog at Norre Vorupor Strand`：

- 说明页：`https://commons.wikimedia.org/wiki/File:Dog_at_N%C3%B8rre_Vorup%C3%B8r_Strand.jpg`
- 下载地址：`https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Dog_at_N%C3%B8rre_Vorup%C3%B8r_Strand.jpg/1280px-Dog_at_N%C3%B8rre_Vorup%C3%B8r_Strand.jpg`
- 许可证：CC0 1.0
- 本文下载文件 SHA256：`e678ecf8dab63da112812ada0553d72d46033e4e9ea3b568bf0e53d92e1d6910`

模型权重只用于本地转换和验证，当前没有模型再分发授权，因此不得提交到 Git、上传 GitHub Packages 或附加到 Release。CC0 图片允许公开再分发，文章中的结果图可以保留原图。

## 5. ONNX 转换与暂存

获取脚本内部使用下面的固定转换命令。这里保留完整命令，便于不使用 PowerShell 获取器时复现：

```text
python eng/Export-ClassificationResNet18Onnx.py --weights <downloads>/resnet18-f37072fd.pth --onnx <models>/resnet18-imagenet1k-v1.onnx --labels <models>/imagenet1k.names --report <models>/resnet18-onnx-export.json
```

转换环境为 PyTorch `2.10.0+cpu`、TorchVision `0.25.0+cpu`、ONNX opset 17。转换后的模型统一暂存在仓库外的：

```text
models/Classification/resnet18-torchvision-v0.25.0/resnet18-imagenet1k-v1.onnx
```

固定 ONNX 长度为 `46,748,553` bytes，SHA256 为：

```text
ead3558569edd88aa73a4eb46acbe6c38dee113933234547f04a0f6e48169903
```

模型合同是 `images:float32[1,3,224,224] -> logits:float32[1,1000]`。输入采用 RGB、NCHW、短边缩放到 256、中心裁剪 224 x 224、scale `1/255`、ImageNet mean/std；输出 logits 再执行 Softmax 和稳定 Top-5 排序。

## 6. 使用公开包准备案例

`samples/ComputerVision/01.Classification` 是可执行案例，不是 NuGet 库。项目设置为 `IsPackable=false`，
通过仓库共享 props 引用公开的 `JYPPX.TensorRT.CSharp.API` 4 系列包以及
OpenCV-CSharp-API：<https://github.com/guojin-yan/OpenCV-CSharp-API>。它不引用 `src` 下的 TensorRT/CUDA 项目，
也不存在 `JYPPX.TensorRT.CSharp.API.Classification` 这个待发布案例包。

在仓库外创建自己的项目时，可以不写死具体版本：

```powershell
dotnet add package JYPPX.TensorRT.CSharp.API --version "4.0.0"
dotnet add package JYPPX.OpenCV.CSharp.API --prerelease
dotnet add package JYPPX.OpenCV.runtime.win-x64 --prerelease
dotnet add package JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge --version "4.0.0"
```

最后一个包 ID 必须按本机环境选择。它只包含项目自有 bridge；CUDA、cuDNN 和 TensorRT 继续从用户安装目录加载。

先定义工作目录，正文后续命令都使用变量，避免绑定某台机器的盘符：

```powershell
$RepoRoot = (Resolve-Path .).Path
$WorkspaceRoot = Split-Path -Parent $RepoRoot
$ModelRoot = Join-Path $WorkspaceRoot 'models/Classification/resnet18-torchvision-v0.25.0'
$CaseRoot = Join-Path $WorkspaceRoot 'downloads/article-assets/classification-resnet18'
$ImageJpeg = Join-Path $CaseRoot 'dog.jpg'
$ImageBmp = Join-Path $CaseRoot 'dog.bmp'
New-Item -ItemType Directory -Force -Path $CaseRoot | Out-Null
```

下载 CC0 图片。案例使用 `JYPPX.OpenCV.CSharp.API` 直接解码 JPEG，不再要求为了运行案例额外转成 BMP：

```powershell
Invoke-WebRequest `
  'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Dog_at_N%C3%B8rre_Vorup%C3%B8r_Strand.jpg/1280px-Dog_at_N%C3%B8rre_Vorup%C3%B8r_Strand.jpg' `
  -OutFile $ImageJpeg

```

## 7. 编写程序入口

案例入口把命令参数交给 `ClassificationCommand.Run(string[])`，命令参数和退出码保持稳定：

```csharp
TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
Console.WriteLine($"BridgeTensorRt={snapshot.BuildInfo.TensorRtVersion}");
return ClassificationCommand.Run(args);
```

命令执行器内部主流程可以概括为下面五步：

```csharp
ClassificationImagePreprocessResult? preprocess = TryPreprocessImage(args);
OnnxSampleResult result = TensorRtOnnxSample.RunSingleFloatInputOutput(options);

float[] probabilities = ClassificationOutputProcessor.Transform(
    result.OutputValues,
    ClassificationScoreTransform.Softmax);
IReadOnlyList<ClassificationPrediction> top5 =
    ClassificationOutputProcessor.GetTopK(probabilities, labels, 5);

ClassificationOutputReportWriter.Write(
    outputJsonPath, options, result, preprocess, labelsPath, labelsSha256,
    labels.Count, transform, probabilities, top5, referenceContext, validation);

ClassificationVisualizationWriter.Write(
    visualizationPath, visualizationBackgroundPath, preprocess!, top5);
```

图片预处理结果会保存 source/tensor SHA256 和完整语义合同。可视化写入器只接受与预处理源图同尺寸的 JPEG、PNG 或 BMP 背景；尺寸不一致、没有真实图片输入或 Top-K 为空都会直接失败，防止把结果贴到错误图片上。

## 8. 编译并运行

先还原并检查依赖图，然后编译案例。这里不会打包或发布案例项目：

```powershell
dotnet restore ./samples/ComputerVision/01.Classification/Classification.csproj
dotnet list ./samples/ComputerVision/01.Classification/Classification.csproj package
dotnet build ./samples/ComputerVision/01.Classification/Classification.csproj -c Release --no-restore
```

本文实测环境选择 `win-x64-trt10.11-cuda12.9-cudnn9.22` bridge。设置本机 TensorRT 根目录；
CUDA 和 cuDNN 由运行时根目录解析器按运行时键检查：

```powershell
$env:TENSORRT_PATH = '<TensorRT-root>'
```

运行使用公开包构建的仓库案例，并让 C# 写出权威预处理 tensor，以便独立框架使用完全相同的输入：

```powershell
$InputTensor = Join-Path $CaseRoot 'classification-input.fp32.bin'
$OutputJson = Join-Path $CaseRoot 'classification-output.json'
$Visualization = Join-Path $CaseRoot 'classification-result.svg'

dotnet run --project ./samples/ComputerVision/01.Classification -c Release --no-build -- `
  --model (Join-Path $ModelRoot 'resnet18-imagenet1k-v1.onnx') `
  --labels (Join-Path $ModelRoot 'imagenet1k.names') `
  --image $ImageJpeg --preprocessed-output $InputTensor `
  --input-shape 1x3x224x224 --tensor-rt-line 10 `
  --image-resize shorter-side-center-crop --resize-shorter-side 256 `
  --tensor-layout NCHW --color-order RGB --scale 0.00392156862745098 `
  --mean 0.485,0.456,0.406 --std 0.229,0.224,0.225 `
  --score-transform softmax --top-k 5 --noTF32 `
  --output-json $OutputJson --visualization $Visualization `
  --visualization-background $ImageJpeg
```

为这个精确 tensor 生成独立参考：

```powershell
$InputSha = (Get-FileHash $InputTensor -Algorithm SHA256).Hash.ToLowerInvariant()
$ReferenceRoot = Join-Path $CaseRoot 'reference'

python ./eng/Invoke-ClassificationResNet18Reference.py `
  --weights (Join-Path $WorkspaceRoot 'downloads/resnet18-torchvision-v0.25.0/source/resnet18-f37072fd.pth') `
  --onnx (Join-Path $ModelRoot 'resnet18-imagenet1k-v1.onnx') `
  --labels (Join-Path $ModelRoot 'imagenet1k.names') `
  --input-tensor $InputTensor --output-directory $ReferenceRoot `
  --expected-input-sha256 $InputSha
```

生成独立参考后，使用同一条案例命令增加 `--reference-output` 重新运行，比较任务概率与原始 logits。
再复制参考 JSON，把索引 0 的参考值增加 `0.125` 后执行受控负例；进程必须返回非零，证明错误参考会失败关闭。

## 9. 已验证结果与语义边界

### 9.1 先给结论：类别表没有错位

结果中的类别名称**不是标签索引用错**。本文使用的 `imagenet1k.names` 由 torchvision `ResNet18_Weights.DEFAULT.meta["categories"]` 直接生成，共 1000 行；分类程序使用输出数组的零基索引读取同一行标签。下面五个 Top-5 索引已经逐项从标签文件复核：

| ImageNet 索引 | 标签文件中的类别 | 本次概率 | 排名 |
| ---: | --- | ---: | ---: |
| 200 | `Tibetan terrier` | `0.309864` | 1 |
| 194 | `Dandie Dinmont` | `0.242565` | 2 |
| 204 | `Lhasa` | `0.124855` | 3 |
| 155 | `Shih-Tzu` | `0.119756` | 4 |
| 266 | `miniature poodle` | `0.044701` | 5 |

容易产生误解的地方是：这些名称是 **ResNet18 对图片给出的 ImageNet-1K 模型预测**，并不是图片的人工标注或真实犬种结论。原图来源只描述它“可能为 Shih Tzu / Maltese 混种”，所以 Top-1 `Tibetan terrier` 与来源描述并不一致。这里应判断为模型在相似长毛犬种之间的不确定预测，而不是把 Top-1 改写成原图的真实类别。

### 9.2 运行环境与独立参考

本文实跑环境为 Windows 11、NVIDIA GeForce RTX 3060 Laptop GPU、CUDA 12.9、TensorRT 10.11。24-bit BMP 和同图 PPM 得到的 C# 输入 tensor 完全一致，SHA256 为 `43de394443f6fc3ccfd08cd9df61ee645ee5c51d1954c52267c221a438252f9e`。

独立参考先比较 PyTorch 与 ONNX Runtime，1000 个 logits 的最大绝对误差为 `1.049041748046875e-05`，argmax 一致。TensorRT 再与 ONNX Runtime 比较：

| 项目 | 比较值数 | mismatch | 最大绝对误差 |
| --- | ---: | ---: | ---: |
| Softmax 任务概率 | 1000 | 0 | `3.87e-7` |
| 原始 logits | 1000 | 0 | `6.68e-6` |

### 9.3 本次模型预测与运行状态

本次 TensorRT enqueue 记录为 `3.063 ms`，程序结束于 `OutputValidated=True` 和 `Classification Passed=True`。受控负例返回 `exit 1`、`mismatch 1`、`firstMismatch 0`，说明参考值错误时会失败关闭。

这里必须区分“推理实现一致”和“图片类别判断正确”。Wikimedia 原图说明：<https://commons.wikimedia.org/wiki/File:Dog_at_N%C3%B8rre_Vorup%C3%B8r_Strand.jpg>把这只狗描述为“可能为 Shih Tzu / Maltese 混种”，并没有把它标注为 Tibetan terrier。ResNet18 的 Top-1 只有约 31%，第 2 至第 4 也集中在外形相近的长毛犬种，其中 `Shih-Tzu` 位于第 4。这是一条真实但语义上不确定、且 Top-1 与来源说明不一致的模型输出。

TensorRT 与 ONNX Runtime 对全部 1000 个概率和 logits 的比较通过，只能证明两条执行路径输出一致，不能把模型预测提升为图片的真实犬种。

> 结果图阅读方式：蓝色条目是 ImageNet-1K 模型预测，不是 ground truth；Top-1 `Tibetan terrier` 保留为程序真实输出，不代表本文认定这只狗就是该犬种。

<img src="../../../images/classification-resnet18-annotated-cc0.webp" alt="ResNet18 对 CC0 狗图片给出的真实 ImageNet-1K Top-5 模型预测；不是犬种 ground truth，Top-1 与原图来源描述不一致" width="640" style="display:block;max-width:100%;height:auto;margin:16px auto;" />

<img src="../../../images/classification-resnet18-local-package-consumer-terminal.png" alt="Classification 真实运行输出" width="640" style="display:block;max-width:100%;height:auto;margin:16px auto;" />

终端截图来自真实运行的 stdout，只移除了机器路径并压缩成长短适合窗口展示的字段；shape、耗时、Top-5、比较数量、负例和通过状态均未修改。结果图使用同一个 CC0 输入和同一套 Classification 可视化写入器生成。它用于展示真实模型输出及其局限，不作为犬种识别正确样例。

## 10. 复查与边界

复查时至少确认：

```powershell
Get-FileHash (Join-Path $ModelRoot 'resnet18-imagenet1k-v1.onnx') -Algorithm SHA256
Get-FileHash $InputTensor -Algorithm SHA256
pwsh -NoProfile -ExecutionPolicy Bypass -File ./eng/Test-ClassificationLocalPackageConsumer.ps1
```

原始结构化结果保存在 `samples/assets/classification-resnet18-local-package-consumer-runtime-evidence.json`。该文件名保留发布前历史证据语义，不应改写为新的公开包证明。本文证明固定 ResNet18、固定 CC0 图片、固定预处理和 TensorRT 10.11 的推理与 fail-closed 对照；公共包的 post-publish 验证需要单独生成新的运行记录。

模型文件继续只放在外层 `models` 目录，不上传 GitHub；CUDA、cuDNN、TensorRT 和 NVRTC 继续由用户安装。本文内容完整不等于已授权公开发布，更不会触发包、Release 或版本发布。

## 11. 延伸阅读

- SMP-002：推理输入、显存绑定与 GPU 输出读回：<https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/docs/articles/zh-cn/02-samples/smp-002-inference-bindings.md>
- SMP-003：Dynamic Shape 与动态 Batch：<https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/docs/articles/zh-cn/02-samples/smp-003-dynamic-shapes.md>
- SMP-004：从 ONNX 到 TensorRT Engine：<https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/docs/articles/zh-cn/02-samples/smp-004-onnx-build-and-run.md>

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

<img src="../../../images/personal-contact-banner-v6-zh.png" alt="作者联系方式" width="640" style="display:block;max-width:100%;height:auto;margin:16px auto;" />
<!-- public-article-declaration:end -->
