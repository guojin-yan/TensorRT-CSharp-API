# YoloVision Segmentation 真实模型教程

本文说明 YOLO segmentation 模型在 YoloVision 中应如何准备资产、解释输出和采集 proof。segmentation 不只输出 boxes，还包含 mask coefficients、prototype 或 per-pixel logits，因此 evidence 必须比 detection 更细。

## 适用读者

适合准备接入 YOLOv8-seg、YOLOv11-seg 或 custom segmentation ONNX 模型的用户，也适合维护 YoloVision 多任务教程的人。

## 解决问题

segmentation 的常见问题是 engine 构建成功但 mask 语义不清：prototype shape、mask coefficient layout、resize 回原图规则和阈值都可能不同。本文把 build-only report、YoloVision matrix 和真实 mask 输出 proof 分开记录。

## 背景与场景

真实业务中的分割模型常用于缺陷检测、医学图像、实例分割或工业视觉。它对输入 resize、letterbox、mask crop 和后处理阈值更敏感，所以必须把图像资产、输出 metadata 和可视化建议一起记录。

## 操作路径

1. 下载 segmentation 权重并记录 license、source URL 和 SHA256。
2. 导出 ONNX，记录输出 tensor 名称、prototype shape 和 coefficient count。
3. 用 TensorRtExec build-only 模式生成 engine 构建报告。
4. 用 YoloVision segmentation runner 执行真实图片，保存 mask summary 和可选 overlay 图。
5. 用 validator 校验模型 hash、图片 hash、输出摘要和 host metadata。

## 代码与文件入口

- `samples/YoloVision/Program.cs`
- `samples/YoloVision/yolo-model-matrix.json`
- `docs/articles/zh-cn/yolovision-segmentation-tutorial.md`
- `artifacts/user-acceptance/sample-run-evidence-record.yolovision.template.json`
- `applications/TensorRtExec/tensor-rt-exec-feature-matrix.json`

## 边界说明

build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 都不是 runtime proof。segmentation runtime proof 必须证明真实 mask 输出、图像资产和后处理参数，不只是 engine 文件存在。

## 下一步

下一步为 YOLOv8 segmentation 准备一篇图文教程：包含模型获取、图片来源、导出命令、mask overlay 示例和 proof validator 输出。
