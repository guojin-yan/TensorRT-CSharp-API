# Classification Asset Candidates

本文为 `samples/ComputerVision/01.Classification` 选择候选模型资产。当前状态是候选清单，属于未实跑 sample smoke，不是项目内置模型。

## 选择原则

- 许可证清晰，来源可追溯。
- 输入 shape 稳定，优先 `1x3x224x224`。
- 可通过 PyTorch 或 ONNX 工具导出到 ONNX。
- labels 和测试图片不随仓库默认分发，必须单独记录来源和许可。

## 候选 1：TorchVision ResNet18

- 来源：[TorchVision ResNet18 文档](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
- 代码许可证：[TorchVision BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE)
- 推荐用途：分类入门教程，模型结构经典，输入 `1x3x224x224`。
- 导出方式：用 `torchvision.models.resnet18(weights=...)` 加 `torch.onnx.export` 导出。
- labels：建议使用 ImageNet 1K labels，必须记录来源和许可证。
- 测试图片：使用自有或许可清晰图片，不直接假定互联网图片可再分发。

待验证：

- ONNX opset。
- TensorRT parser 支持。
- Top-K 输出。
- 模型 SHA256。

对应 manifest：`samples/assets/classification-assets.template.json`。

对应 acquisition plan：`artifacts/user-acceptance/sample-asset-acquisition-plan.md`。

## 候选 2：TorchVision MobileNetV2

- 来源：[TorchVision model 文档](https://docs.pytorch.org/vision/0.8/models.html)
- 代码许可证：[TorchVision BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE)
- 推荐用途：小体积分类 demo，适合发布文章和快速下载。
- 输入：通常使用 ImageNet 风格 `1x3x224x224`。
- 导出方式：用 TorchVision 权重导出 ONNX。

待验证：

- 预处理与权重 metadata 是否一致。
- TensorRT line 10/11 parser 行为。
- synthetic input 与真实图片路径分开记录。

## 候选 3：自训练或内部授权分类模型

- 来源：项目 owner 自有模型或明确授权模型。
- 推荐用途：微信公众号、博客和商业演示。
- 优点：可控许可证，可随文章附下载方式。
- 要求：必须附训练/导出说明、labels、测试图片和许可证。

## 不允许的写法

- 不要写“Classification sample 已完成真实图片分类效果”，除非已经保存真实模型、labels、图片和 Top-K 输出。
- 不要把 synthetic input 的 `Classification Passed=True` 写成模型精度证明。
- 不要把候选模型下载链接写成仓库已内置资产。
