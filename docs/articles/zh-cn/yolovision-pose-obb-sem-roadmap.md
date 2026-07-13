# YoloVision Pose / OBB / Semantic Segmentation 路线图

YoloVision 的长期目标不是只覆盖 detection，而是统一承载 det、cls、seg、obb、pose、sem 等任务。本文把 pose、OBB 和 semantic segmentation 的接入路线拆开，说明哪些是样例能力规划，哪些需要真实模型 proof。

## 适用读者

适合计划扩展 YoloVision 后处理的维护者，也适合准备为不同视觉任务撰写系列文章的作者。

## 解决问题

pose、OBB 和 semantic segmentation 的输出语义差异很大。pose 需要 keypoint 和 skeleton，OBB 需要 angle 和 rotated box，semantic segmentation 需要 per-pixel class map。本文解决任务边界、资产字段和 proof 要求不清的问题。

## 背景与场景

这些任务很适合作为宣传文章系列：pose 可展示人体或关键点，OBB 可展示遥感/工业旋转框，semantic segmentation 可展示逐像素分类。但任何图文展示都必须来自真实模型和真实输入，不能从 template 或 matrix 推导。

## 操作路径

1. 为每个任务选择一个候选模型，并记录来源、license、ONNX 导出方式。
2. 在 `yolo-model-matrix.json` 中补 family/task/profile/postprocess 字段。
3. 为每个任务定义输出 metadata：keypoint count、angle convention、class map shape 等。
4. 用 TensorRtExec 或 OnnxToEngine 生成 build-only report。
5. 用 YoloVision runner 和 validator 采集真实 runtime proof。

## 代码与文件入口

- `samples/YoloVision/yolo-model-matrix.json`
- `docs/articles/zh-cn/yolovision-pose-tutorial.md`
- `docs/articles/zh-cn/yolovision-obb-tutorial.md`
- `docs/articles/zh-cn/yolovision-classification-semantic-tutorial.md`
- `artifacts/user-acceptance/sample-run-evidence-record.yolovision.template.json`

## 边界说明

build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 都不是 runtime proof。路线图、任务矩阵和图示建议只能描述计划，不代表真实模型已经验证。

## 下一步

下一步按任务拆文章：pose keypoint 输出解释、OBB angle convention、semantic segmentation color map，并为每篇准备真实模型资产字段。
