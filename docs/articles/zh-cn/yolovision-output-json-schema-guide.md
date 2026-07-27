# YoloVision 输出 JSON Schema 指南

## 适用读者

本文面向需要把 `samples/YoloVision` 输出接入自动化测试、可视化工具或业务系统的开发者，重点说明 detection、segmentation、pose、OBB 四类任务的输出字段如何稳定记录。

## 解决问题

YoloVision 可以覆盖 YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom 与 det、cls、seg、obb、pose、sem，但如果输出 JSON 没有 schema，用户很难比较不同模型、不同 TensorRT 版本或不同后处理参数的结果。本文给出统一字段建议，同时明确 build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 都不是 runtime proof。

## 背景与场景

输出 JSON 的价值是让样例从“肉眼看截图”升级为“机器可复查记录”。它应记录输入资产、engine、任务类型、输出张量 metadata、后处理参数和最终预测结果。它可以帮助形成 runtime proof candidate，但只有在真实兼容主机、真实资产、真实日志和 validator 同时存在时，才可能进入 runtime proof 链。

## 实现路径

1. 顶层字段建议包含 `schemaVersion`、`task`、`modelFamily`、`input`、`engine`、`runtime`、`outputs`、`postprocess` 和 `predictions`。
2. detection prediction 应包含 `box`、`classId`、`className`、`score` 和 `sourceTensor`。
3. segmentation prediction 应额外包含 `maskShape`、active `maskPixelCount`、`maskTotalPixelCount`、`maskThreshold`、`maskValueKind`、`maskPixelCountScope=prototype-grid-before-crop-resize`；显式启用 `--mask-spatial-transform` 时，还包含 source-image shape/count、coordinate space、bilinear interpolation、source box 与 exporter-validation boundary 自洽的 `spatialTransform`。
4. pose prediction 应额外包含 `keypoints`，每个 keypoint 包含 `index`、`x`、`y`、`score` 和可选 `visibility`。
5. OBB prediction 应包含 `center`、`size`、`angle`、`angleUnit`、`angleRange` 和可选四点坐标。

## 代码与文件入口

- `samples/YoloVision`：统一输出生成入口。
- `samples/YoloVision/yolo-model-matrix.json`：模型族、任务和输出 metadata 矩阵。
- `docs/articles/zh-cn/yolovision-detection-yolov8n-download-export-run.md`：detection 运行路径。
- `docs/articles/zh-cn/yolovision-segmentation-mask-postprocess-guide.md`：mask 后处理路径。
- `docs/articles/zh-cn/yolovision-pose-keypoint-output-guide.md` 与 `docs/articles/zh-cn/yolovision-obb-angle-output-guide.md`：pose/OBB 输出说明。

## 图示建议

建议画一张 JSON 树状图：顶层 `run` 记录输入与 runtime，下面按 `predictions[]` 展开 det、seg、pose、obb 的差异字段，并用颜色标出哪些字段可用于 validator。

## 边界说明

输出 JSON schema 是可复查性的基础，但不是 runtime proof。TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics、build-only、dry-run、template、local feed、ProjectReference 和 direct `.nupkg` 都只能作为上下文或候选输入，不能替代真实 clean consumer 或真实样例运行 proof。

## 下一步

下一轮应把该 schema 固化为示例 JSON 和质量测试，至少覆盖 det、seg、pose、OBB 的必填字段，并记录 schemaVersion 以避免后续字段漂移。
