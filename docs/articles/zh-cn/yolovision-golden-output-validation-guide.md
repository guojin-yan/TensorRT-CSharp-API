# YoloVision Golden Output 校验指南

## 适用读者

本文面向需要验证 YoloVision 输出稳定性的测试人员和维护者，重点说明如何用 golden output 检查 det、seg、pose、OBB 的后处理结果。

## 解决问题

仅看截图无法判断模型输出是否可复现。本文给出 golden output 校验思路：记录输入 hash、engine hash、输出 JSON、容差、后处理参数和比较结果，同时避免把 build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 当作 runtime proof。

## 背景与场景

TensorRT 版本、精度、shape profile 和后处理阈值都可能导致输出差异。Golden output 的目标不是强制所有环境逐像素一致，而是为每个任务定义可解释的容差和关键字段，例如 detection 的框数量与最高分、segmentation 的 mask 面积范围、pose 的关键点数量、OBB 的 angle 约定。

## 操作路径

1. 为每个真实资产记录输入图片 hash、engine hash、runtime package 和后处理参数。
2. 运行 YoloVision 生成输出 JSON，并将关键字段保存为 golden baseline。
3. 对 detection 比较 class、score tolerance、IoU tolerance 和 box count。
4. 对 segmentation 比较 mask shape、maskPixelCount tolerance 和 box IoU。
5. 对 pose/OBB 比较 keypoint count、keypoint score、angleUnit、angleRange 和 rotated box IoU。

## 代码与文件入口

- `samples/YoloVision`：输出 JSON 生成入口。
- `samples/YoloVision/yolo-model-matrix.json`：任务类型和输出 metadata。
- `docs/articles/zh-cn/yolovision-output-json-schema-guide.md`：输出 schema。
- `docs/articles/zh-cn/yolovision-real-asset-record-template-guide.md`：真实资产记录模板。
- `tests/JYPPX.ProjectQuality.Tests`：适合承载 schema/golden 字段质量测试。

## 图示建议

建议画一张比较流程图：`input + engine -> current output -> compare with golden -> tolerance result -> proof candidate boundary`，并在最后节点标注仍需真实 runtime proof。

## 边界说明

Golden output validation 能证明输出结构和后处理更稳定，但不自动等于 runtime proof。TensorRtExec report、OnnxToEngine report、YoloVision matrix、readonly diagnostics、build-only、dry-run、template、local feed、ProjectReference 和 direct `.nupkg` 仍必须被视为非替代项。

## 下一步

下一轮应为每种任务定义最小 golden 示例和比较容差字段，并把 schema 字段纳入质量测试，确保输出记录能被机器解析。
