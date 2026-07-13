# YoloVision 总览：一个样例覆盖 YOLO 多系列多任务

早期检测样例命名太窄，只能让人想到 detection。现在项目已经切换到 `samples/YoloVision`：它的定位是完整覆盖 YOLO 系列常见任务，让一个样例成为后续技术文章、模型教程和真实 proof overlay 的共同入口。

## 适合谁阅读

- 希望在 .NET 中统一跑 YOLO 多任务模型的视觉工程师。
- 准备为 YOLOv8 det/seg/pose/obb/cls/sem 回填真实模型资产的维护者。
- 需要理解 YoloVision 与 TensorRtExec、OnnxToEngine、package-consumer proof 边界的发布负责人。

## 目标范围

YoloVision 的长期目标是支持：

- YOLOv5、v6、v7、v8、v9、v10、v11、v26。
- detection、classification、segmentation、OBB、pose、semantic segmentation。
- ONNX 输入、TensorRT engine 构建、外部预处理 tensor、任务专属后处理 metadata。

当前已有 YOLOv8 六任务候选资产模板：

```text
samples/assets/yolovision-yolov8-det-candidate.template.json
samples/assets/yolovision-yolov8-seg-candidate.template.json
samples/assets/yolovision-yolov8-pose-candidate.template.json
samples/assets/yolovision-yolov8-obb-candidate.template.json
samples/assets/yolovision-yolov8-cls-candidate.template.json
samples/assets/yolovision-yolov8-sem-candidate.template.json
```

这些模板是 owner-action-required，不是 runtime proof。它们用于告诉维护者真实模型资产需要补什么。

## 为什么要做成统一样例

YOLO 系列模型的问题不是“能不能加载 ONNX”这么简单。不同任务的输出 tensor 语义差异很大：

- det 需要 box、class score、NMS。
- seg 需要 detection output 和 mask prototype。
- pose 需要 keypoint count、layout、visibility/confidence。
- obb 需要 rotated box 和 angle metadata。
- cls 需要 topK class score。
- sem 需要 class map layout、palette 和 resize-back rule。

统一样例可以把这些差异放在明确 metadata 中，而不是散落到每篇文章或每个 demo 的临时代码里。

## 典型命令

YOLOv8 detection 的模板命令形态：

```powershell
dotnet run --project .\samples\YoloVision -- `
  --model .\models\yolov8-det.onnx `
  --labels .\models\coco.names `
  --input-data .\models\yolov8-det-fp32.bin `
  --input-shape 1x3x640x640 `
  --tensor-rt-line 10 `
  --family v8 `
  --task det `
  --layout auto `
  --has-objectness auto `
  --nms-mode class-aware
```

pose、obb、cls、sem 会继续追加任务专属参数，例如 keypoint layout、rotated box layout、topK、semantic map shape。

## Proof 边界

YoloVision 可以支持 real-model-runtime proof，但不能直接支持 package-consumer-runtime proof。原因很简单：样例运行证明的是某个模型资产在某个 host 上可以走通样例路径；package proof 必须来自干净外部 consumer 使用公开 NuGet 包执行。

真实模型晋级至少需要：

- 模型、labels、图片、预处理 tensor、运行日志的 64 位 SHA256。
- stdout/stderr 摘要。
- owner 提供的 YoloVision 成功标记日志行。
- 任务专属 output metadata。
- owner 审核。

## 配图建议

- YOLO 六任务输出 tensor 到结果对象的流程图。
- YoloVision CLI 参数到任务 decoder 的映射图。
- det/seg/pose/obb/cls/sem 六格结果示意图，后续用真实 owner 图片替换。

## 下一步

下一步应把 YOLOv8 六任务模板扩展到 YOLOv5/v6/v7/v9/v10/v11/v26，并为每个系列标注输出差异、导出命令、许可证和真实资产回填要求。
