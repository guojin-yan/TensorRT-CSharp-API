# TensorRT CSharp API v4.0 完整应用

本模块面向需要完整工作流的使用者，覆盖模型转换、桌面工具和多任务视觉应用。`applications/` 当前提供三个源码应用；它们不是独立 NuGet 包，也不在 `v4.0.0` Release 中作为应用二进制发布。

## 1. 首批正式文章

| 编号 | 应用 | 主要用途 | 正式文章 | 状态 |
| --- | --- | --- | --- | --- |
| `APP-YV-001` | `YoloVision` | 检测、分类、实例分割、OBB、姿态和语义分割 | [全任务总览](yolovision/app-yv-001-yolovision-overview.md) | `ready` |
| `APP-YV-002` | `YoloVision` | YOLOv8n 目标检测真实模型闭环 | [YOLOv8n 目标检测](yolovision/app-yv-002-yolov8n-detection.md) | `ready` |
| `APP-YV-003` | `YoloVision` | YOLOv8n ImageNet 图像分类与 Top-5 复核 | [YOLOv8n 图像分类](yolovision/app-yv-003-yolov8n-classification.md) | `ready` |
| `APP-YV-004` | `YoloVision` | YOLOv8n 双输出实例分割与 mask 验证 | [YOLOv8n 实例分割](yolovision/app-yv-004-yolov8n-instance-segmentation.md) | `ready` |
| `APP-YV-005` | `YoloVision` | YOLOv8n 17 关键点人体姿态估计 | [YOLOv8n 姿态估计](yolovision/app-yv-005-yolov8n-pose.md) | `ready` |
| `APP-YV-006` | `YoloVision` | YOLOv8n DOTA 旋转目标检测 | [YOLOv8n OBB](yolovision/app-yv-006-yolov8n-obb.md) | `ready` |
| `APP-YV-007` | `YoloVision` | LRASPP VOC 21 类语义分割 | [LRASPP 语义分割](yolovision/app-yv-007-lraspp-semantic-segmentation.md) | `ready` |
| `APP-YV-008` | `YoloVision` | YOLOv10n End-to-End 六列输出检测 | [YOLOv10n 检测](yolovision/app-yv-008-yolov10n-end-to-end.md) | `ready` |
| `APP-YV-009` | `YoloVision` | YOLOX-S grid/stride 目标检测 | [YOLOX-S 检测](yolovision/app-yv-009-yolox-s-detection.md) | `ready` |
| `APP-ONNX-001` | `OnnxToEngine` | ONNX 解析、Engine 构建、MNIST 推理与报告 | [OnnxToEngine 入门](onnxtoengine/app-onnx-001-onnx-to-engine-getting-started.md) | `ready` |
| `APP-ONNX-002` | `OnnxToEngine` | MNIST 项目自有输入、TensorRT 与 ORT 双重验证 | [MNIST 完整流程](onnxtoengine/app-onnx-002-mnist-runtime-validation.md) | `ready` |
| `APP-ONNX-003` | `OnnxToEngine` | Dynamic Shape、精度、Workspace、Timing Cache 与报告 | [高级构建选项](onnxtoengine/app-onnx-003-advanced-build-options.md) | `ready` |
| `APP-EXEC-001` | `TensorRtExec` | trtexec 风格 CLI 与 WinForms 工作流 | [TensorRtExec 入门](tensorrtexec/app-exec-001-tensorrtexec-getting-started.md) | `ready` |
| `APP-EXEC-002` | `TensorRtExec` | 使用 WinForms 配置 ONNX 并生成 Engine 与报告 | [GUI 构建指南](tensorrtexec/app-exec-002-gui-onnx-build.md) | `ready` |
| `APP-EXEC-003` | `TensorRtExec` | trtexec 参数迁移、分层状态和严格输出校验 | [CLI 参数指南](tensorrtexec/app-exec-003-cli-parameter-guide.md) | `ready` |
| `APP-EXEC-004` | `TensorRtExec` | 多流 Benchmark、CUDA Graph、精度策略与 Reference | [性能与输出校验](tensorrtexec/app-exec-004-performance-and-output-validation.md) | `ready` |
| `APP-EXEC-005` | `TensorRtExec` | Stripped Plan、ONNX Refit、完整权重持久化与独立 Reload | [Refit 与 Engine 持久化](tensorrtexec/app-exec-005-refit-and-engine-persistence.md) | `ready` |

旧文章没有删除或批量移动，原路径继续兼容既有链接和证据合同。新文章使用 `APP-YV-###`、`APP-ONNX-###` 和 `APP-EXEC-###` 标识应用系列，并在 [`article-index.json`](../article-index.json) 登记 canonical 路径。

`APP-ONNX-001` 至 `APP-ONNX-003` 已于 2026-08-13 完成 MNIST、ORT、动态 Profile 和高级构建批次验证；`APP-EXEC-001` 至 `APP-EXEC-005` 也已在同一 TensorRT 10.11、CUDA 12.9、RTX 3060 Laptop 环境完成 CLI bounded runtime、GUI build-only、benchmark、独立 loadEngine 与 Refit 持久化复跑，批次证据见 [`tensorrtexec-runtime-evidence-20260813.json`](tensorrtexec/tensorrtexec-runtime-evidence-20260813.json)。当前总计为 `42 ready / 8 review`；这些状态不等于 public package、post-publish 或 release-close 证明。

## 2. 新增规则

1. 使用 `APP-<系列>-###` 作为稳定 ID，并在 [`article-index.json`](../article-index.json) 登记 canonical 路径。
2. 每篇文章必须区分命令解析、Engine 构建、真实运行、输出语义校验和公开包消费者验证。
3. 桌面应用文章需要真实软件页面；模型应用还需要可复核模型来源与任务结果图。
4. 未实现、只解析或仅生成报告的能力必须按实际状态表述，不使用“完整支持”等扩大性措辞。
5. 应用源码继续保持 source-only，除非未来存在单独、明确的发布决策。
