# YoloVision Detection 真实模型教程

本文面向想用 `applications/YoloVision` 跑 YOLO detection 模型的用户。它给出从模型选择、ONNX 导出、TensorRtExec build-only 预检，到 YoloVision runner 采集真实输出证据的完整路径。当前仓库没有内置真实模型资产，因此本文是可执行教程和 evidence contract，不把文档本身写成 runtime proof。

## 适用读者

适合准备接入 YOLOv5、YOLOv6、YOLOv7、YOLOv8、YOLOv9、YOLOv10、YOLOv11、YOLOv26 或 custom detection 模型的用户，也适合需要为公众号、博客或 release evidence 准备真实 detection 案例的维护者。

## 解决问题

Detection 案例最容易把“engine 能构建”和“框输出正确”混在一起。本文解决三件事：模型和图片从哪里来，如何记录 shape/layout/labels/NMS 规则，以及如何把 TensorRtExec report、YoloVision matrix 和真实 YoloVision run log 分成不同证据层。

## 背景与场景

YOLO detection 模型的输出 layout 因 family 和导出脚本不同而变化。v5/v7 常见 objectness + classes 输出，v8/v11 常见 decoupled head，v10 可能走 NMS-free 输出。用户必须记录输出 shape、class count、confidence rule、NMS mode 和 labels，否则同一个 engine 即使能构建，也不能证明检测框正确。

## 操作路径

1. 从官方仓库或模型发布页下载权重，记录 URL、license、commit/tag 和 SHA256。
2. 使用模型项目推荐脚本导出 ONNX，保留导出命令和 opset。
3. 用 `applications/OnnxToEngine` 或 `applications/TensorRtExec` 先生成 build-only 报告。
4. 准备一张授权可发布的测试图片和 labels 文件。
5. 运行 `applications/YoloVision` 的 detection 模式，保存 stdout/stderr summary、输出 JSON、输入图片 hash、模型 hash 和 validator 结果。

## 代码与文件入口

- `applications/YoloVision/Program.cs`
- `applications/YoloVision/yolo-model-matrix.json`
- `samples/assets/yolovision-assets.template.json`
- `artifacts/user-acceptance/sample-run-evidence-record.yolovision.template.json`
- `applications/TensorRtExec/README.md`

## 边界说明

build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 都不是 runtime proof。真实 detection runtime proof 必须来自真实模型、真实输入图片、真实兼容主机、运行日志、hash、stdout/stderr summary 和 validator。

## 下一步

下一步为一个许可证清晰的 YOLOv8n detection 模型补真实资产包：模型 URL、ONNX 导出命令、labels、测试图片、YoloVision 输出和 sample-run evidence validator。
