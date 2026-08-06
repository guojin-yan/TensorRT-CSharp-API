# YoloVision YOLOv8n Classification：Labels、中心裁剪与 Top-K 实战

## 适用读者

本文面向需要把 YOLOv8n-cls 部署到 TensorRT 10.x、核对 ImageNet labels 顺序，或为 C# 分类接口准备可复核演示材料的开发者。

## 解决问题

这条案例收口四个容易被忽略的问题：

1. 官方 ONNX 输出是 `[1,1000]` Softmax probabilities，不是 logits。
2. Ultralytics 8.4.21 使用短边缩放到 224 后中心裁剪，不使用检测任务的 letterbox。
3. `ImageNet.yaml` 的简化 `names` 与权重内嵌名称不完全一致，精确 labels 必须从 `map` 顺序派生。
4. Top-5 一致还不够，运行证据还要比较全部 1000 个原始概率，并提供受控负例。

## 背景与场景

分类模型没有 box、mask 和 NMS，看起来比检测简单，但 labels、预处理或 softmax 语义只要错一个，结果就会稳定地“看起来合理但实际错误”。本案例固定官方资产、导出版本、输入 tensor、完整输出 reference 和日志哈希，让每一层都能独立复查。

当前已验证环境为 Windows 11、RTX 3060 Laptop GPU、驱动 576.02、CUDA 12.9、TensorRT 10.11、Ultralytics 8.4.21。该记录是源码树 `real-model-runtime`，仍不是 `package-consumer-runtime` proof，也不是模型再分发或发布授权。

## 官方资产合同

机器可读清单位于：

- `samples/assets/yolovision-yolov8n-cls-official-assets.json`
- `samples/assets/yolovision-yolov8n-cls-real-model-runtime-evidence.json`
- `eng/Acquire-YoloV8ClassificationOfficialAssets.ps1`
- `eng/Invoke-YoloVisionClassificationReference.py`

固定资产如下：

| 资产 | 固定来源 | SHA256 |
| --- | --- | --- |
| `yolov8n-cls.pt` | Ultralytics assets `v8.3.0`，Release asset ID `195719213` | `11fa19f2...8245980a` |
| `ImageNet.yaml` | Ultralytics commit `6e43d1e1...124b0a` | `3f9b74af...0461a15` |
| `bus.jpg` | 同一 Ultralytics commit | `c02019c4...34bfbc63` |
| `LICENSE` | 同一 Ultralytics commit | `0d96a4ff...079abcb0` |

上游 Release API 没有为权重提供 digest。清单中的权重 SHA256 是从精确 Release asset ID 首次下载后固定的仓库校验值，不冒充上游签名。

## Labels 为什么取 `map`

`ImageNet.yaml` 同时包含简化的 `names` 和 WordNet `map`。权重内嵌 `model.names` 与 `map` 的 1000 个 value 按插入顺序完全一致；与简化 `names` 有 564 个字符串差异，其中 305 个不是简单的空格转下划线。

例如：

| 索引 | `model.names` / `map` | 简化 `names` |
| --- | --- | --- |
| 4 | `hammerhead` | `hammerhead shark` |
| 15 | `robin` | `American robin` |
| 20 | `water_ouzel` | `American dipper` |

派生后的 `imagenet-yolov8n-cls.names` 共 1000 行，SHA256 为 `dcc60e72...84240dd`。不要把 `names` 的可读性优化误当成权重的精确 label contract。

## 输出分数合同

YoloVision 现在提供显式 `--classification-score-mode`：

| 模式 | 行为 |
| --- | --- |
| `raw` | 保留旧接口语义，只做 finite 检查、阈值、排序和 Top-K |
| `logits` | 对有限 logits 执行数值稳定 softmax，再做 Top-K |
| `probabilities` | 要求每项有限且位于 `[0,1]`，总和与 1 的误差不超过 `0.001` |

配置了 `--class-count` 后，值必须与输出向量长度严格相等，不再静默截断。官方 YOLOv8n-cls ONNX 的末节点是 `Softmax`，所以必须使用 `--classification-score-mode probabilities`，不能再标记成 logits，也不能重复执行 softmax。

## 预处理合同

Ultralytics 8.4.21 的实际 transform 为：

```text
Resize(shorter-side=224, bilinear, antialias)
CenterCrop(224, 224)
RGB -> float32 NCHW
scale = 1/255
mean = 0,0,0
std = 1,1,1
```

这不是传统 ImageNet `mean/std`，也不是 letterbox。YoloVision 的 `cls` profile 现在默认 `1x3x224x224`、`shorter-side-center-crop`、RGB、NCHW、`1/255` 和 confidence 0。C# 抗锯齿缩放与 Ultralytics tensor 的最大像素差为 `1/255`、平均绝对误差 `0.000383371`，ORT Top-5 索引和顺序完全一致。其他模型若不符合这些默认值，仍应使用 owner-approved preprocess pipeline，并固定工具版本、参数和 tensor hash。

## 可复用资产目录与完整验证

大文件全部放在 E 盘，不提交到源码仓库：

```text
..\downloads\cases\yolov8n-cls\models
..\downloads\cases\yolov8n-cls\labels
..\downloads\cases\yolov8n-cls\images
..\downloads\cases\yolov8n-cls\tensors
..\downloads\cases\yolov8n-cls\engines
..\downloads\cases\yolov8n-cls\reports
..\downloads\cases\yolov8n-cls\logs
```

本仓库默认使用外层 `downloads/yolov8n-cls-ultralytics-v8.3.0`，脚本会拒绝 C 盘路径。

### 1. 获取资产

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Acquire-YoloV8ClassificationOfficialAssets.ps1 `
  -AssetRoot ..\downloads\cases\yolov8n-cls `
  -PythonPath C:\path\to\python.exe
```

脚本只下载、校验和派生 PPM/labels，`performsExport=false`、`performsRuntime=false`、`performsPublish=false`。资产获取本身不是 runtime proof。

### 2. 导出和生成独立 reference

```powershell
python .\eng\Invoke-YoloVisionClassificationReference.py `
  --weights ..\downloads\cases\yolov8n-cls\source\yolov8n-cls.pt `
  --imagenet-yaml ..\downloads\cases\yolov8n-cls\source\ImageNet.yaml `
  --image ..\downloads\cases\yolov8n-cls\source\bus.jpg `
  --onnx ..\downloads\cases\yolov8n-cls\source\yolov8n-cls.onnx `
  --output-directory ..\downloads\cases\yolov8n-cls\reports\independent-reference `
  --export-onnx
```

导出固定为 opset 17、static batch 1、simplify。ONNX checker 确认 `images:[1,3,224,224] -> output0:[1,1000]`，末节点为 Softmax。PyTorch 与 ONNX Runtime 的最大绝对误差为 `3.5762787e-7`。

### 3. 验证 C# 直接图片预处理

```powershell
dotnet run --project .\applications\YoloVision -c Release -- `
  --preprocess-only `
  --image ..\downloads\cases\yolov8n-cls\derived\bus.ppm `
  --preprocessed-output ..\downloads\cases\yolov8n-cls\tensors\bus-csharp.fp32.bin `
  --family v8 --task cls `
  --tensor-layout NCHW --color-order RGB `
  --resize shorter-side-center-crop --resize-shorter-side 224
```

### 4. TensorRT 完整运行

```powershell
dotnet run --project .\applications\YoloVision -c Release -- `
  --model ..\downloads\cases\yolov8n-cls\source\yolov8n-cls.onnx `
  --labels ..\downloads\cases\yolov8n-cls\reports\independent-reference\imagenet-yolov8n-cls.names `
  --input-data ..\downloads\cases\yolov8n-cls\reports\independent-reference\input-ultralytics-1x3x224x224.fp32.bin `
  --family v8 --task cls --classification-output output0 `
  --classification-score-mode probabilities --confidence 0 --top-k 5 `
  --reference-outputs output0:..\downloads\cases\yolov8n-cls\reports\independent-reference\output0.reference.json `
  --reference-abs-tolerance 0.001 --reference-rel-tolerance 0.001 `
  --output-json ..\downloads\cases\yolov8n-cls\reports\yolovision-output.json `
  --visualization-svg ..\downloads\cases\yolov8n-cls\reports\yolovision-output.svg
```

## 实测结果

TensorRT 对全部 1000 个概率完成 comparison：

```text
Passed=True Compared=1000 Mismatches=0 FirstMismatch=-1
MaxAbs=0.0005927086
YoloVision Passed=True
```

Top-5 为：

| 排名 | 索引 | 类别 | TensorRT score |
| --- | ---: | --- | ---: |
| 1 | 654 | `minibus` | 0.504948 |
| 2 | 734 | `police_van` | 0.292809 |
| 3 | 874 | `trolleybus` | 0.049876 |
| 4 | 575 | `golfcart` | 0.018730 |
| 5 | 612 | `jinrikisha` | 0.017923 |

结构化输出的 `postprocess.classificationScoreMode` 和 `outputs[0].role` 都应为 `probabilities`；`postprocess.topK=5`，prediction 包含 `classId`、`className` 和 `score`。

## 受控负例

reference 脚本只修改索引 0：`value += 0.125`。使用同一 `0.001` 容差运行后得到：

```text
exit code = 1
Mismatches=1
FirstMismatch=0
MaxAbs=0.125
YoloVision Passed=False
```

这证明容差可以吸收 TensorRT tactic 的微小数值波动，但不会放过受控数据篡改。

## Owner Backfill Checklist（候选模板字段）

通用回填仍从 `samples/assets/yolovision-yolov8-cls-candidate.template.json` 开始，字段包括 `model.sourceUrl`、`model.downloadUrl`、`model.license`、`model.sha256`、`labels.sourceUrl`、`labels.sha256`、`labels.classCount`、`input.imageSha256`、`input.preprocessedTensorSha256`、`outputMetadata.classificationOutput`、`outputMetadata.outputShape`、`outputMetadata.classCount`、`outputMetadata.labelsPath`、`outputMetadata.topK`、`outputMetadata.classScoreField` 和 `outputMetadata.postprocessMetadata.activation`。

哈希用标准命令复核：

```powershell
Get-FileHash -Algorithm SHA256 ..\downloads\cases\yolov8n-cls\source\yolov8n-cls.onnx
Get-FileHash -Algorithm SHA256 ..\downloads\cases\yolov8n-cls\reports\independent-reference\imagenet-yolov8n-cls.names
```

## 代码与文件入口

- `applications/YoloVision/YoloClassificationScoreMode.cs`
- `applications/YoloVision/YoloSampleRunner.cs`
- `applications/YoloVision/YoloImagePreprocessor.cs`
- `applications/YoloVision/YoloVisionOutputReport.cs`
- `applications/YoloVision/yolovision-task-output-contract.json`
- `applications/YoloVision/yolovision-output.schema.json`

## 验证命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionOutputReport.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionRealAssetCandidate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Release --filter FullyQualifiedName~YoloVision
```

## Proof Boundary（边界说明）

模型、ONNX、图片、labels、tensor、reference、SVG 和日志只保存在 E 盘，不进入 Git。源码树真实运行可以证明当前 C# 接口、bridge 和本机 TensorRT 的主链路，但仍不是 `package-consumer-runtime` proof。

build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 和截图都不能替代真实运行，也不能授权 tag、Release、NuGet 或 GitHub Packages 发布。

## 下一步

后续用同一 score-mode 合同覆盖 YOLOv11-cls、自定义 logits 模型和不同输入尺寸；每个模型必须重新确认输出是否已经包含 Softmax，不能复制本案例的 `probabilities` 设置后直接套用。
