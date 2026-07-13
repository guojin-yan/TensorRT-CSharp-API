# YoloVision Asset Candidates

本文为 `samples/YoloVision` 选择候选 detector 资产。当前状态是候选清单，不是已实跑检测 demo，也不是项目随包分发模型。

## 选择原则

- 明确模型、代码、权重、labels 和测试图片的许可证。
- 能导出单输入 float ONNX。
- 输出 layout 可解释，例如 `[1, 84, 8400]` 或 `[1, 8400, 84]`。
- postprocess 规则可写入文档，包括 confidence、IoU、objectness 和 NMS。

## 候选 1：YOLOX

- 来源：[YOLOX GitHub 仓库](https://github.com/Megvii-BaseDetection/YOLOX)
- 许可证：[Apache-2.0](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE)
- 推荐用途：许可证相对宽松，适合作为优先检测候选。
- 导出方向：使用 YOLOX 官方导出流程生成 ONNX，再用 `samples/YoloVision` 验证输出 layout。

待验证：

- 官方权重许可是否与仓库许可证一致。
- ONNX opset 和 TensorRT parser 支持。
- 输出是否需要 application-side NMS。
- COCO labels 来源和测试图片许可。

对应 manifest：`samples/assets/yolovision-assets.template.json`。

对应 acquisition plan：`artifacts/user-acceptance/sample-asset-acquisition-plan.md`。

## 候选 2：Ultralytics YOLOv8/YOLO11

- 来源：[Ultralytics 文档](https://docs.ultralytics.com/)
- 许可证边界：[YOLOv8 文档说明 AGPL-3.0/Enterprise](https://docs.ultralytics.com/models/yolov8)，[YOLO11 文档说明 AGPL-3.0/Enterprise](https://docs.ultralytics.com/models/yolo11)
- 推荐用途：用户自带模型路径或内部验证。
- 风险：AGPL/Enterprise 边界必须由 release owner 确认，不能默认作为可再分发宣传资产。

待验证：

- 导出的 ONNX 输出 layout。
- 是否包含 NMS。
- AGPL/Enterprise 对发布文章、模型下载和商业 demo 的影响。

## COCO Labels 和图片

- COCO 官方站点：[cocodataset.org](https://cocodataset.org/)
- COCO labels 可作为类别顺序参考，但测试图片必须逐项确认来源和授权。
- 不建议在仓库中直接分发未复核授权的互联网图片。

## 不允许的写法

- 不要把 YOLOX 或 Ultralytics 候选写成 `YoloVision Passed=True`。
- 不要把 COCO 测试图片默认写成可再分发。
- 不要把 AGPL/Enterprise 模型写成无条件可商用可分发。
- 不要把 synthetic input 的 pipeline evidence 写成检测质量证明。
