# YoloVision YOLOv8n Segmentation Mask 后处理指南

## 适用读者

本文适合准备将 YOLO segmentation 模型接入 TensorRT + C#、需要理解多输出张量和实例 mask 后处理的开发者，也适合负责模型资产、许可证和 runtime proof 审核的项目维护者。

读者应了解 detection 的 box、score、class 基础概念，并能够使用 Python 导出 ONNX、使用 TensorRtExec 构建 engine、使用 YoloVision 执行真实输入。

## 解决问题

本文重点解决：

1. detection branch、mask coefficient 和 prototype branch 如何对应。
2. letterbox、crop、resize、sigmoid 和 threshold 的执行顺序。
3. 如何用显式 output role map 避免根据 tensor 名猜测。
4. 如何保存模型、输入、tensor、engine、输出 JSON 和 overlay 的可审计证据。
5. 如何避免把 build-only、template 或截图误写成 segmentation runtime proof。

## 背景与场景

本文面向准备把 YOLOv8n-seg 接入 `samples/YoloVision` 的开发者。Segmentation 的难点通常不是 engine 能不能 build，而是 box 分支、mask prototype 分支、mask coefficient、letterbox 还原和阈值策略是否能被稳定解释。

本篇把一条可发布的 segmentation walkthrough 拆成资产准备、ONNX 导出、TensorRtExec build-only、YoloVision 运行、mask 输出解释和 proof boundary。它是一篇文章和真实资产模板说明，不是 runtime proof。

## 适用场景

YOLOv8n-seg 适合验证 YoloVision 的多输出 metadata 能力：检测框输出负责 class、score 和 box，prototype 输出负责 mask basis，保留下来的检测框再用 coefficient 组合成实例 mask。

如果你的目标是写公众号/博客文章，建议配一张原图、一张 mask overlay、一段输出 JSON 和一张“box branch + prototype branch”的流程图。仓库不应提交大模型或私有图片，真实资产由 owner 通过 SHA256 与证据记录回填。

## 操作路径

建议按以下顺序完成一条可复现的 segmentation 路径：

1. 获取模型、labels 和输入图片，记录来源、许可证与 SHA256。
2. 固定 Ultralytics、Python、opset、dynamic shape 和 imgsz 导出 ONNX。
3. 使用 Netron 或 parser diagnostics 确认 detection/prototype 输出名和 shape。
4. 使用 TensorRtExec 构建 engine，保存 build-only report。
5. 按模型要求生成 NCHW float32 输入 tensor。
6. 使用 YoloVision 显式绑定 detection 与 prototype output role。
7. 执行 decode、NMS、mask composition、crop、resize 和 threshold。
8. 保存输出 JSON、overlay、日志和全部 SHA256，再交由 owner validator 审核。

## 模型与许可证

owner 需要确认：

- `yolov8n-seg.pt` 来源和许可证。
- 导出后的 `yolov8n-seg.onnx` SHA256。
- `coco.names` 或自定义 labels SHA256。
- 输入图片许可证、来源和 SHA256。
- 预处理 tensor SHA256。
- mask 输出可视化是否可公开。

如果图片或模型许可证不允许公开，文章可以保留命令和字段说明，但不能附带资产，也不能声称已经完成公开 runtime proof。

## 导出 ONNX

```powershell
yolo export model=.\models\yolov8n-seg.pt format=onnx opset=12 dynamic=True simplify=True imgsz=640
```

导出后重点记录：

- detection branch 输出张量名与 shape。
- prototype branch 输出张量名与 shape。
- `maskCoefficientCount`，YOLOv8n-seg 常见值为 32。
- prototype 空间分辨率。
- class count。
- 是否存在 graph-side NMS 或 graph-side mask 后处理。

YoloVision 运行时可以用 `--output-role-map boxes:det,proto:mask-prototypes` 显式声明输出角色，避免仅依赖张量名猜测。

## TensorRtExec Build-Only

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\yolov8n-seg.onnx `
  --saveEngine .\models\yolov8n-seg.plan `
  --minShapes images:1x3x640x640 `
  --optShapes images:1x3x640x640 `
  --maxShapes images:4x3x640x640 `
  --fp16 `
  --buildOnly `
  --exportReport .\models\yolov8n-seg-build-report.json
```

build report 可以帮助定位 parser、profile、precision 和 output binding，但仍是 build-only evidence。它不能证明 mask 后处理正确，也不能替代真实 YoloVision run log。

## YoloVision 运行

```powershell
dotnet run --project .\samples\YoloVision -- `
  --model .\models\yolov8n-seg.onnx `
  --labels .\models\coco.names `
  --input-data .\models\yolov8n-seg-fp32.bin `
  --input-shape 1x3x640x640 `
  --family v8 `
  --task seg `
  --output-role-map boxes:det,proto:mask-prototypes `
  --mask-coefficient-count 32
```

真实日志至少应包含：

- `Profile Family=v8 Task=seg`
- `InputSource=external`
- `Segmentations=...`
- `Postprocess Task=seg`
- owner 提供的 YoloVision 成功标记日志行

如果只有 synthetic tensor，结果只能作为 managed pipeline evidence，不能作为真实图像 segmentation proof。

## Mask 输出解释

建议在输出 JSON 中记录：

- `maskCoefficientCount`
- `prototypeShape`
- `maskThreshold`
- `letterboxScale`
- `letterboxPadX`
- `letterboxPadY`
- `maskPixelCount`
- `boxBeforeCrop`
- `boxAfterResize`
- `className`
- `score`

后处理顺序建议为：decode boxes -> score filtering -> NMS -> coefficient 与 prototype 组合 -> sigmoid -> crop -> resize -> threshold。文章中要说明这些步骤和训练/导出预处理必须一致。

## 代码与文件入口

- `samples/YoloVision/YoloVisionRuntimePipeline.cs`：多输出 runtime 结果路由。
- `samples/YoloVision/YoloVisionSegmentationDecoder.cs`：box、coefficient 与 prototype 组合。
- `samples/YoloVision/YoloVisionMaskComposer.cs`：sigmoid、crop、resize 与 threshold。
- `samples/YoloVision/YoloVisionNms.cs`：保留 detection 与 mask coefficient 的索引一致性。
- `samples/YoloVision/yolovision-task-output-contract.json`：segmentation 输出角色契约。
- `samples/YoloVision/Program.cs`：`--task seg`、`--output-role-map` 和 mask 参数入口。
- `eng/Test-YoloVisionRealAssetCandidate.ps1`：模型、图片、labels、日志和 SHA256 校验。

## 图示建议

正式发布文章建议包含：

1. detection branch 与 prototype branch 的 ONNX/Netron 截图。
2. 原图、letterbox 输入图、检测框和最终 mask overlay。
3. coefficient × prototype -> sigmoid -> crop -> resize 流程图。
4. 输出 JSON 中 box、class、score、maskPixelCount 的片段。
5. TensorRtExec build report 与 YoloVision runtime log 的边界对照图。

overlay 和截图只用于人工理解，不能替代真实日志、结构化输出、SHA256 和 validator。

## 常见问题

如果 mask 整体偏移，优先检查 letterbox 的 scale 与 padding；如果 mask 轮廓正确但类别错误，检查 labels 顺序和 class count；如果 mask 全黑，检查 coefficient 数量、prototype role map 和 sigmoid/threshold；如果检测框正确但 mask 数量为 0，检查是否只捕获了 detection 输出而没有捕获 prototype 输出。

## 边界说明（Proof Boundary）

本文、mask overlay、TensorRtExec report、YoloVision matrix、OnnxToEngine report、sidecar-only report、template、dry-run、build-only、screenshot、local feed、ProjectReference、direct `.nupkg` 和 readonly diagnostics 都不是 runtime proof。

真实 `real-model-runtime` 候选必须有 owner 回填的模型、labels、输入、run log、SHA256、输出 JSON、许可证说明和 validator 结果。`package-consumer-runtime` 仍需要公开包来源和外部 clean consumer 运行记录。

## Owner Backfill Checklist

- 填写模型、labels、图片、preprocessed tensor 和 run log 的 SHA256。
- 保存 TensorRtExec build-only report，但不要把它当作 proof。
- 保存 YoloVision run command 和 `YoloVision Passed=True`。
- 记录 detection output、prototype output、mask coefficient 和 threshold。
- 记录至少一张可公开 overlay 或说明不能公开的原因。
- 让 owner review mask 数量、类别、面积和坐标还原是否可信。

## 下一步

完成 YOLOv8n-seg 后，应继续验证不同导出布局、不同 prototype 分辨率和 graph-side NMS 模型，并扩展 YOLOv5/v9/v11 segmentation。每种模型都要记录实际 output shape 和 mask coefficient，不能只修改 family 名称。

随后可在仓库外 clean package consumer 中重复同一真实输入，建立 package-consumer-runtime 证据；source-tree `real-model-runtime` 仍不能替代发布包 proof。
