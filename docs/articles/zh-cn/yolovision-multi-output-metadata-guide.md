# YoloVision 多输出 Metadata 指南：让 YOLO 系列样例从检测扩展到 Seg、Pose、OBB

很多项目里的 YOLO 示例只覆盖一个固定模型：输入写死为 `1x3x640x640`，输出写死为 `[1,84,8400]`，然后把后处理写进一个函数里。这样做适合演示，却不适合做一个可维护的 TensorRT/C# 项目。因为 YOLOv5、YOLOv6、YOLOv7、YOLOv8、YOLOv9、YOLOv10、YOLOv11 以及后续 family 的 ONNX 输出并不完全相同；同一个 family 里，det、cls、seg、obb、pose、sem 任务也会改变输出 tensor 的数量和含义。

`applications/YoloVision` 的定位是做一个 YOLO-family 配置底座，而不是把某个权重文件塞进仓库。它现在支持 family/task/profile 参数、常见 detection 输出布局、score filtering、NMS、classification、semantic map，以及 seg/pose/obb 的托管多输出 helper。本文讲清楚这套 metadata 怎么填、什么时候算 pipeline 证据、什么时候才能写成真实模型 smoke passed。

## 当前能力边界

`YoloVision` 当前可以直接运行单输出路径：

```powershell
dotnet run --project .\applications\YoloVision -- `
  --model .\models\yolo.onnx `
  --labels .\models\coco.names `
  --input-shape 1x3x640x640 `
  --tensor-rt-line 10 `
  --family v8 `
  --task det `
  --layout auto `
  --has-objectness auto `
  --confidence 0.25 `
  --iou-threshold 0.45
```

这个命令能证明 TensorRT ONNX 管线、engine 构建/执行和托管后处理路径是否通畅。但默认 synthetic input 不是图像质量证据，也不能证明某个 YOLO 模型在真实图片上检测正确。

多输出任务目前以托管 helper 的形式落地：

- `YoloSampleRunner.DecodeSegmentationOutputs(...)`
- `YoloSampleRunner.DecodePoseOutputs(...)`
- `YoloSampleRunner.DecodeObbOutputs(...)`
- `YoloMultiOutputMetadata`
- `YoloRuntimeOutputTensor`
- `YoloRuntimeOutputSet`

也就是说，样例已经有可测试的后处理结构，但真实 runtime 多 output capture 和真实模型资产 smoke 仍要单独补证据。

当前共享 ONNX sample support 已经准备了多输出快照结构：

```text
OnnxSampleOutputTensor
OnnxSampleMultiOutputResult
TensorRtOnnxSample.RunSingleFloatInputOutputs(...)
```

`YoloVision` 在此基础上用 `YoloRuntimeOutputTensor` 标记输出角色，例如 `Detection`、`MaskPrototypes`、`PoseKeypoints`、`ObbAngles`。这样后续真实 runtime 多输出模型接入时，不需要重写 seg/pose/obb 的托管后处理，只要把 runtime 输出按角色放进 `YoloRuntimeOutputSet`。

## Runtime Binding Metadata：先确认绑定，再解释输出

真实运行路径现在还会把已有 `TensorRtEngineBindingReport` 复制到输出 JSON 的 `bindingMetadata` 节点，并在控制台打印 `BindingReport` / `BindingMetadata` 摘要。每个 tensor 会记录：

- `index`、`name`、`ioMode`、`semanticRole`、`dataType`、`engineShape`；
- `location`、`format`、`formatDescription`、`vectorizedDimension`；
- `profileMinShape`、`profileOptShape`、`profileMaxShape`；
- `bytesPerComponent`、`componentsPerElement`、数据类型大小回退状态和非致命 diagnostics。

这条路径只复制 managed binding snapshot，不返回 native pointer，也不改变 engine/context ownership。`bindingMetadata.isRuntimeProof` 固定为 `false`：它证明的是绑定元数据已被读取和归档，不证明模型输出语义、图片质量、真实模型正确性或 package-consumer-runtime。

## Detection：先确定 box 输出

检测输出常见两种布局：

```text
[1, 84, 8400]   # channels-first
[1, 8400, 84]   # boxes-first
```

通道含义通常是：

```text
box(4) + class scores
box(4) + objectness(1) + class scores
```

对应参数：

```powershell
--layout auto|channels-first|boxes-first
--has-objectness auto|true|false
--class-count 80
--confidence 0.25
--iou-threshold 0.45
```

如果 `--class-count` 已知，`auto` 通常可以判断 objectness；如果模型输出里还拼接了 mask coefficients、angle 或 keypoints，就应显式记录 auxiliary metadata，避免把额外通道误当作类别分数。

## Segmentation：box、coefficients、prototypes 必须对齐

YOLO segmentation 常见输出由两部分组成：

```text
detection rows: box + class + mask coefficients
prototype tensor: [P,H,W] 或 [1,P,H,W]
```

在 `YoloVision` 里，应填写：

```json
"segmentation": {
  "metadataHelper": "YoloMultiOutputMetadata.ForSegmentation",
  "prototypeShape": "1x32x160x160",
  "coefficientCount": 32,
  "coefficientChannelStart": "auto-after-box-objectness-class-channels",
  "maskCompose": "linear-combination"
}
```

托管 helper 会按 `SourceIndex` 把 NMS/TopK 后保留下来的 detection 映射回原始候选框对应的 coefficients，再和 prototype 做线性组合。真实模型还需要补充 mask crop、resize、threshold、映射回原图尺寸等规则，这些不能靠通用代码猜。

如果 runtime 已捕获多个输出，组织方式类似：

```csharp
var outputs = new YoloRuntimeOutputSet(new[]
{
    new YoloRuntimeOutputTensor("boxes", YoloOutputTensorRole.Detection, boxValues, boxShape),
    new YoloRuntimeOutputTensor("proto", YoloOutputTensorRole.MaskPrototypes, protoValues, protoShape)
});

var result = YoloSampleRunner.DecodeRuntimeOutputs(
    outputs,
    profile,
    YoloMultiOutputMetadata.ForSegmentation(maskCoefficientCount: 32));
```

## Pose：keypoint count 和 stride 不能猜

Pose 输出常见结构是每个 detection 关联一组 keypoint：

```text
keypoint row = x, y, score, x, y, score, ...
```

metadata 应写清楚：

```json
"pose": {
  "metadataHelper": "YoloMultiOutputMetadata.ForPose",
  "keypointCount": 17,
  "keypointStride": 3,
  "keypointTensorLayout": "boxes-first"
}
```

如果某个模型只有 `x,y` 而没有 score，stride 应写成 `2`。如果输出是 channels-first，要在 manifest 里写清楚，否则 keypoint 会错位。

## OBB：角度单位和通道位置是关键

OBB 任务比普通检测多一个角度。角度可能是 degree，也可能是 radian；有些模型还会使用归一化角度或任务特定编码。最小 metadata 应写成：

```json
"obb": {
  "metadataHelper": "YoloMultiOutputMetadata.ForObb",
  "angleUnit": "degrees",
  "angleChannel": "owner-required",
  "angleTensorLayout": "boxes-first"
}
```

`YoloObbDecoder` 只负责把 angle 转成统一的 `AngleRadians`。模型特定的角度范围、旋转方向和后续可视化仍应在真实资产文章里说明。

## Semantic Segmentation：不是检测框

Semantic segmentation 输出通常是类别图或 logits map，例如：

```text
[1,C,H,W]
[1,H,W,C]
[C,H,W]
```

这类任务不应该走 detection decode。`YoloVision` 使用 `YoloSemanticMap` 表达 class-major 数据，后续可以根据文章或应用需要做 argmax、调色板渲染和 resize。

## Asset Manifest 应如何填写

真实模型接入前，先复制：

```powershell
Copy-Item .\samples\assets\yolovision-assets.template.json .\models\my-yolo.assets.json
```

至少补齐：

- 模型来源、下载 URL、license、SHA256、opset、export command。
- labels 来源、class count、license、SHA256。
- 输入图片来源、license、SHA256。
- input tensor name、shape、layout、dtype。
- 每个 output tensor 的 name、role、shape、layout。
- preprocess：resize、padding、color order、scale、mean/std。
- postprocess：layout、objectness、class count、NMS mode。
- seg/pose/obb/sem 的任务专属 metadata。
- 实际 run command 和日志。

在这些证据没有补齐前，`isSmokePassed` 必须保持 `false`。

## 推荐验证顺序

1. 用 `TensorRtExec` 或 `applications/OnnxToEngine` 做 `--buildOnly`，确认 ONNX 能构建 engine。
2. 用 `YoloVision` 单输出 runner 跑 det/cls/sem 这类可直接解码的路径。
3. 对 seg/pose/obb，用托管多输出 helper 先跑 synthetic tensor 单元测试。
4. 接入 `RunSingleFloatInputOutputs(...)` 或 host application 的 runtime 多输出 capture 后，再记录真实模型日志。
5. 最后才写 `YoloVision Passed=True`，并附模型、labels、图片、hash 和许可证证据。

真实运行后还要把 runner 证据结构化：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-SampleRunEvidenceRecordTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -InputPath .\models\yolovision-sample-run-evidence.json
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleAssetManifest.ps1
```

`samples/assets/yolovision-assets.template.json` 中的 `evidence.sampleRunEvidenceRecord` 指向真实 runner record，`evidence.sampleRunEvidenceValidation` 指向校验结果。manifest audit 会在 record 存在时检查 sampleName、model/labels/input SHA256，并拒绝 `package-consumer-runtime`。这能保证多输出 metadata、sidecar build report 和真实 `YoloVision Passed=True` 日志互相对齐。

## 结语

YoloVision 的价值不是“仓库里内置一个看起来能跑的 YOLO 模型”，而是把 YOLO 系列部署中最容易含糊的部分拆开：模型来源、输入预处理、输出布局、任务类型、后处理 metadata 和验证证据。这样写出来的样例和文章，才经得起用户换模型、换平台、换 TensorRT 版本。
