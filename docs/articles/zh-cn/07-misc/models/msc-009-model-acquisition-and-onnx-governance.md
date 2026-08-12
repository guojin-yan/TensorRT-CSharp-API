# 演示模型获取与 ONNX 转换：版本、许可证和 SHA256 管理

<!-- public-article-layout:start -->
<style>
.content article { min-width: 0; overflow-wrap: anywhere; }
.content article a, .content article :not(pre) > code { overflow-wrap: anywhere; word-break: break-word; }
.content article pre:not(.mermaid), .content article table { display: block; max-width: 100%; overflow-x: auto; }
.content pre.mermaid { max-width: 640px; margin: 16px auto; overflow-x: auto; }
.content pre.mermaid svg { display: block; width: 100%; max-width: 640px; height: auto; }
</style>
<!-- public-article-layout:end -->

> 文章编号：`MSC-009`；适用版本：TensorRT CSharp API v4.0 `4.0.0`；当前状态：`ready`。

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

模型案例最容易出现一种隐蔽错误：代码和 Engine 都能运行，但作者与读者使用的不是同一个权重、同一个 ONNX、同一组 labels 或同一种预处理。此时界面上仍会出现 Top-K 或检测框，却无法复现文章结果。模型许可证和再分发边界如果没有记录，还可能让技术文章无意中把不应进入仓库的权重一起发布。

TensorRT CSharp API v4.0：<https://github.com/guojin-yan/TensorRT-CSharp-API> 将演示模型放在 Git 仓库外层，以机器可读 inventory 固定来源、revision、转换工具、文件长度和 SHA256。本文说明如何使用这套流程。它适用于 Samples、OnnxToEngine、TensorRtExec 和 YoloVision。

### 1.2 项目、包与源码入口

| 项目 | 链接 |
| --- | --- |
| GitHub 项目 | TensorRT-CSharp-API：<https://github.com/guojin-yan/TensorRT-CSharp-API> |
| 核心包 | `JYPPX.TensorRT.CSharp.API 4.0.0`：<https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/4.0.0> |
| Bridge 包列表 | JYPPX NuGet 包列表：<https://www.nuget.org/profiles/JYPPX> |
| 模型清单 | `demo-model-inventory.json`：<https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/samples/assets/demo-model-inventory.json> |
| 同步/校验脚本 | `Sync-DemoOnnxModels.ps1`：<https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/eng/Sync-DemoOnnxModels.ps1> |
| 模型获取脚本 | `eng`：<https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/eng> |
| Classification 示例 | 源码：<https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/samples/ComputerVision/01.Classification> |
| YoloVision 应用 | 源码：<https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/applications/YoloVision> |

## 2. 为什么模型不放进 Git 和 NuGet

项目策略为：

```text
workspace-root/
  TensorRtSharp4.0/   # Git 仓库，仅源码、脚本、清单和小型结果图
  downloads/          # 下载权重、图片和中间文件
  models/             # 经过身份校验的 ONNX
```

`samples/assets/demo-model-inventory.json` 明确记录：

- `workspaceModelRoot=../models`；
- `modelRootOutsideGitRepository=true`；
- `onnxFilesTrackedByGit=false`；
- `uploadsModelFiles=false`；
- `publishesModelFiles=false`；
- NVIDIA runtime 不随模型或项目包捆绑。

原因不只是文件大。权重、测试图片和标签可能有各自许可证；模型更新还会改变输出。把二进制放在仓库外，再通过清单验证身份，能同时控制版本、体积和授权边界。

## 3. 一个完整模型记录应包含什么

| 字段 | 目的 |
| --- | --- |
| `id` | 在文章、样例和证据中稳定引用 |
| `sourceUrl` | 回到上游官方来源 |
| `pinnedRevision` | 避免“最新版本”漂移 |
| `license` | 区分源码许可、权重许可和再分发审批 |
| `acquisition.script` | 自动化下载或核验步骤 |
| `conversion.command` | 从权重到 ONNX 的可复现命令 |
| `toolchain` | PyTorch/torchvision/Ultralytics/opset 等版本 |
| `workspaceRelativePath` | 统一外层模型位置 |
| `expectedLength` | 快速发现截断或错误文件 |
| `sha256` | 确认二进制身份 |
| `runtimeEvidence` | 指向已有运行记录，不把清单冒充运行结果 |

SHA256 只说明文件内容一致，不说明来源可信、许可证允许发布或模型准确率正确。

## 4. 当前 10 个演示模型清单

### 4.1 分类与 MNIST

| 模型 ID | 来源/Revision | 许可证边界 | ONNX SHA256 |
| --- | --- | --- | --- |
| `classification-resnet18-imagenet1k-v1` | TorchVision ResNet18 权重：<https://download.pytorch.org/models/resnet18-f37072fd.pth>，`torchvision-v0.25.0@8ac84ee75afb1c327902156b5336f56ad63b7e2f` | 源码 BSD-3-Clause；预训练权重再分发仍需审核 | `ead3558569edd88aa73a4eb46acbe6c38dee113933234547f04a0f6e48169903` |
| `onnxtoengine-nvidia-mnist-opset8` | ONNX Model Zoo MNIST：<https://github.com/onnx/models/tree/main/validated/vision/classification/mnist>，`TensorRT-10.11.0.33-sample-data` | NVIDIA TensorRT sample-data 条款与上游说明 | `2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf` |

### 4.2 检测与分类

| 模型 ID | 来源/Revision | 许可证边界 | ONNX SHA256 |
| --- | --- | --- | --- |
| `yolovision-yolov8n-detection-v8.3.0` | Ultralytics v8.3.0 权重：<https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt>，`ultralytics-v8.3.0@6e43d1e1e5db72afbf686dee6745669bcb124b0a` | AGPL-3.0-only；项目未批准公开再分发 | `db28a49ffbb0425f39ae56252e7e0b43d06b357416c7da58872e285560b4221e` |
| `yolovision-yolov10n-detection-v1.1` | THU-MIG YOLOv10 v1.1 ONNX：<https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.onnx>，`v1.1@799ff3be47d21173bcf29b351820d4b8e955e0fe` | AGPL-3.0-only；项目未批准公开再分发 | `7025ea1913f9a259cf8a8465ed608e10610d1bb376db2e0348b13e3bd286e0d3` |
| `yolovision-yolox-s-detection-0.1.1rc0` | Megvii YOLOX-S ONNX：<https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx>，`0.1.1rc0@e1052df71842031413f6030723c3607b839c80ce` | Apache-2.0；项目仍未批准公开再分发二进制 | `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063` |
| `yolovision-yolov8n-classification-v8.3.0` | Ultralytics v8.3.0 权重：<https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-cls.pt>，同一固定 revision | AGPL-3.0-only；项目未批准公开再分发 | `630c022a99885d59f633ab5a614738f8a49be7f361e340fd3ff89b8c19b0768f` |

### 4.3 分割、Pose 与 OBB

| 模型 ID | 来源/Revision | 许可证边界 | ONNX SHA256 |
| --- | --- | --- | --- |
| `yolovision-yolov8n-instance-segmentation-v8.3.0` | YOLOv8n-seg v8.3.0：<https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt>，同一固定 revision | AGPL-3.0-only；项目未批准公开再分发 | `08b5c61368d4ddec5e647522fc55a93c42a9e0c581770aae48b87bba65a9b21d` |
| `yolovision-yolov8n-pose-v8.3.0` | YOLOv8n-pose v8.3.0：<https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt>，同一固定 revision | AGPL-3.0-only；项目未批准公开再分发 | `ed1e8d2d2aeb8a2c66e642a16295a72a2990393e3a3843325537da7e11c8a899` |
| `yolovision-yolov8n-obb-v8.3.0` | YOLOv8n-obb v8.3.0：<https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-obb.pt>，同一固定 revision | AGPL-3.0-only；项目未批准公开再分发 | `5f2701ef5326fb5a691999438cfc55a69656323c21ffddebaff8968ab6de2e92` |
| `yolovision-lraspp-mobilenet-v3-large-v0.25.0` | TorchVision LRASPP 权重：<https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth>，`torchvision-v0.25.0@8ac84ee75afb1c327902156b5336f56ad63b7e2f` | 源码 BSD-3-Clause；项目未批准公开再分发权重 | `3cb94e561bdefe606ed7d1a2c4d0296409bec066f3a39a9fe9dabd72b23728f8` |

## 5. 先查看清单，再下载

```powershell
$inventory = Get-Content .\samples\assets\demo-model-inventory.json -Raw | ConvertFrom-Json
$inventory.models | Select-Object id,task,@{n='revision';e={$_.acquisition.pinnedRevision}},@{n='sha256';e={$_.onnx.sha256}}
```

这一步让使用者在联网前就知道将获取什么、放在哪里、如何验证。获取脚本的参数并不完全相同，执行前应查看对应脚本的 `param` 区域，不要给所有脚本机械添加同一个开关。

## 6. ResNet18：权重获取与本地导出

```powershell
$outerRoot = Split-Path $PWD -Parent
$python = 'python'

pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Acquire-TorchVisionResNet18OfficialAssets.ps1 `
  -AssetDirectory (Join-Path $outerRoot 'downloads\resnet18-torchvision-v0.25.0\source') `
  -ModelDirectory (Join-Path $outerRoot 'models\Classification\resnet18-torchvision-v0.25.0') `
  -PythonPath $python `
  -AllowDownload `
  -ExportOnnx
```

固定工具链为 PyTorch `2.10.0+cpu`、torchvision `0.25.0+cpu`、opset 17。预期 ONNX：

```text
models/Classification/resnet18-torchvision-v0.25.0/resnet18-imagenet1k-v1.onnx
length=46748553
sha256=ead3558569edd88aa73a4eb46acbe6c38dee113933234547f04a0f6e48169903
```

分类结果还依赖与 ImageNet-1K 输出索引一致的 labels。模型、labels、预处理必须作为同一个合同管理，不能只固定 ONNX。

## 7. Ultralytics YOLOv8：固定权重后导出

以检测模型为例，导出命令为：

```powershell
yolo export model=yolov8n.pt format=onnx imgsz=640 opset=17 simplify=True dynamic=False batch=1 device=cpu
```

其它任务：

```powershell
yolo export model=yolov8n-cls.pt  format=onnx imgsz=224  opset=17 simplify=True dynamic=False batch=1 device=cpu
yolo export model=yolov8n-seg.pt  format=onnx imgsz=640  opset=17 simplify=True dynamic=False batch=1 device=cpu
yolo export model=yolov8n-pose.pt format=onnx imgsz=640  opset=17 simplify=True dynamic=False batch=1 device=cpu
yolo export model=yolov8n-obb.pt  format=onnx imgsz=1024 opset=17 simplify=True dynamic=False batch=1 device=cpu
```

当前 inventory 记录的导出工具链为 Ultralytics `8.4.21`、opset 17。即使权重文件相同，Ultralytics、ONNX exporter、opset 或 simplify 版本变化都可能改变 ONNX SHA256，因此工具版本也是模型身份的一部分。

仓库提供相应脚本：

- `eng/Acquire-YoloV8DetectionOfficialAssets.ps1`
- `eng/Acquire-YoloV8ClassificationOfficialAssets.ps1`
- `eng/Acquire-YoloV8SegOfficialAssets.ps1`
- `eng/Acquire-YoloV8PoseOfficialAssets.ps1`
- `eng/Acquire-YoloV8ObbOfficialAssets.ps1`

为脚本显式传入外层 `AssetRoot`/`OutputRoot`，避免依赖某台开发机的默认目录。

## 8. 直接使用上游 ONNX

YOLOv10n 与 YOLOX-S 已有固定 release ONNX，不需要本地框架导出：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Acquire-YoloV10OfficialAssets.ps1 `
  -OutputRoot (Join-Path $outerRoot 'downloads\yolov10-agpl')

pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Acquire-YoloXOfficialAssets.ps1 `
  -OutputRoot (Join-Path $outerRoot 'downloads\yolox-apache')
```

“无需转换”不等于无需验证。仍应检查 release tag、URL、文件长度和 SHA256，并保存 acquisition report。

## 9. LRASPP 本地导出

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Acquire-TorchVisionLrasppOfficialAssets.ps1 `
  -AssetDirectory (Join-Path $outerRoot 'downloads\lraspp-mobilenet-v3-large-torchvision-v0.25.0\source') `
  -ModelDirectory (Join-Path $outerRoot 'models\YoloVision\SemanticSegmentation\lraspp-mobilenet-v3-large-torchvision-v0.25.0') `
  -PythonPath $python `
  -AllowDownload `
  -ExportOnnx
```

固定合同为 opset 17，输入 `images:[1,3,320,320]`，输出 `semantic:[1,21,320,320]`。输出类别图需要与相应 label/palette 解释一致。

## 10. 统一同步与验证

获取脚本将文件放入外层 downloads/models 后，用统一清单校验：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Sync-DemoOnnxModels.ps1
```

只检查、不复制：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Sync-DemoOnnxModels.ps1 `
  -VerifyOnly
```

脚本会拒绝把模型根目录放进 Git 仓库，逐项比较文件是否存在、长度和 SHA256，并输出 inventory validation report。验证失败时不要更新清单去迎合未知文件；先回到固定来源与转换工具链查明差异。

手工核对单个文件：

```powershell
$modelPath = Join-Path $outerRoot 'models\Classification\resnet18-torchvision-v0.25.0\resnet18-imagenet1k-v1.onnx'
Get-Item $modelPath | Select-Object FullName,Length,LastWriteTimeUtc
Get-FileHash -Algorithm SHA256 $modelPath
```

## 11. ONNX 合同必须单独记录

哈希一致后还要记录：

- 输入/输出 tensor 名称；
- dtype；
- 静态或动态 Shape；
- 图像尺寸、layout、颜色顺序；
- normalization 与 letterbox/crop；
- 输出轴语义；
- labels 的来源、顺序与 SHA256；
- 后处理阈值、NMS/End-to-End 规则。

可以用 ONNX 图检查工具或项目构建报告导出 I/O metadata。不要只看文件名中的 `640` 或 `cls` 猜测合同。

## 12. Engine 不是新的模型来源

从 ONNX 构建 `.plan` 后，应新增 Engine provenance，而不是覆盖 ONNX 记录：

| 字段 | 示例 |
| --- | --- |
| ONNX SHA256 | inventory 中的固定值 |
| TensorRT | `10.11.0.33` |
| CUDA/cuDNN | `12.9` / `9.22.0` |
| Build flags | FP16、profiles、workspace、strongly typed 等 |
| GPU | 构建/验证 GPU 型号 |
| Plan SHA256 | 构建后计算 |
| Output reference | 固定输入与输出 hash/容差 |

Plan 受 TensorRT、GPU 与兼容策略约束，不能用它替代可追溯的 ONNX 来源。

## 13. 图片与 Labels 也要治理

模型 hash 正确但分类类别离谱时，优先核对：

1. 图片内容和授权；
2. resize/crop/letterbox 是否匹配模型；
3. RGB/BGR 与 NCHW/NHWC；
4. normalization；
5. labels 文件是否对应同一模型输出顺序；
6. Top-K index 与 label index 是否偏移；
7. 展示图是否确实由当前结果生成。

文章中的“识别结果图”必须来自同一模型、输入、labels 和运行报告。不能拿另一模型的类别列表覆盖在当前图片上，也不能仅凭视觉相似度手工改 Top-1。

## 14. 许可证与再分发

以下结论必须分开：

- 上游源码采用某个许可证；
- 预训练权重是否采用同一许可证；
- 测试图片是否允许再分发；
- 项目维护者是否批准把二进制放进 GitHub Release/NuGet；
- 使用者自己的商业或内部使用是否满足许可证。

当前项目清单对多个模型明确记录 `public redistribution owner approval is false`。因此文章可以给出官方获取链接、固定 revision 和 SHA256，但项目不会把这些模型文件放入仓库、NuGet 或 4.0.0 Release。

## 15. 新增模型的检查清单

- [ ] 使用稳定 `id`，不以 `latest` 作为身份。
- [ ] 来源 URL 指向官方仓库、官方 release 或官方权重服务。
- [ ] 固定 tag/commit/revision。
- [ ] 记录源码、权重、图片和 labels 的许可证边界。
- [ ] 下载到仓库外层 `downloads`，ONNX 放外层 `models`。
- [ ] 固定转换工具版本、opset 和完整命令。
- [ ] 记录 ONNX length 与 SHA256。
- [ ] 记录输入输出、预处理、labels 和后处理合同。
- [ ] 用固定输入完成 TensorRT 输出与 reference 校验。
- [ ] 结果图与报告来自同一次真实运行。
- [ ] 未经批准不上传模型、权重、图片或 plan。

## 16. 总结

可复现的模型案例不是“提供一个下载地址”就完成了。完整链路应包含官方来源、固定 revision、许可证、转换工具链、ONNX 合同、文件长度、SHA256、外层存储位置、固定输入和输出验证。TensorRT CSharp API v4.0 的 inventory 与同步脚本把这些字段固化下来，让每篇模型文章、每次 TensorRT 构建和每张结果图都能回到同一个可核验模型身份。

<!-- public-article-declaration:start -->
## 17. 文章声明

### 17.1 开源协议声明
作者所有开源项目代码均遵循 Apache License 2.0 开源协议。
特别说明：本项目集成了若干第三方库。若任何第三方库的许可协议与 Apache 2.0 协议存在冲突或不一致，均以该第三方库的原始许可协议为准。本项目不包含也不代表这些第三方库的授权声明，使用前请务必阅读并遵守第三方库的相关许可。

### 17.2 代码开发与质量说明
AI 辅助开发：本代码在开发过程中使用了人工智能（AI）辅助生成与优化，并非完全由人工逐行编写。
安全性承诺：作者郑重声明，本代码中绝无任何有意设置的后门、病毒、木马或旨在破坏用户设备、窃取数据的恶意代码。
技术局限性：受限于作者个人的技术水平与能力，代码中可能存在因逻辑不严谨、优化不足或经验欠缺导致的低级问题（例如但不限于内存泄漏、偶发崩溃、资源未释放等）。这些问题纯属能力不足所致，并非主观故意。
测试范围：由于作者精力有限，未对本软件进行全方位、覆盖所有边缘场景的完整测试。

### 17.3 免责声明（重要）
请在将本代码应用于任何实际项目（特别是商业、工业或关键任务环境）之前，务必进行详尽、严格的自行测试与验证。 鉴于上述可能存在的代码缺陷及测试覆盖不足，因使用本代码而导致的任何直接或间接损失（包括但不限于设备故障、数据丢失、系统瘫痪或利润损失等），本作者概不负责。 一旦您开始使用本代码，即表示您已知晓上述风险并同意自行承担一切后果，相关问题与本作者无关。

### 17.4 代码开源范围
本项目承诺核心逻辑代码完全开源，但上述提到的“第三方库”的二进制文件、源代码或相关资源不在本项目的开源义务范围内，请根据其各自的指引获取。

### 17.5 社区与反馈
尽管存在上述不足，我们仍欢迎大家下载使用、提交 Issue 或参与测试，共同完善项目。如果您在使用过程中发现 Bug、内存溢出或有改进建议，欢迎通过项目主页提供的联系方式与作者取得联系，我们将尽力在有限的时间内提供协助。

<img src="../../../../images/personal-contact-banner-v6-zh.png" alt="作者联系方式" width="640" style="display:block;max-width:100%;height:auto;margin:16px auto;" />
<!-- public-article-declaration:end -->
