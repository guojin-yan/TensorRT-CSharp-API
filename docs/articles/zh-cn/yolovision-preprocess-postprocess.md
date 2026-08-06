# YoloVision 预处理与后处理指南

YoloVision 故意不把某一个 YOLO 导出脚本的图像预处理写死为全系列默认。不同 family、opset、export 参数和任务类型会影响输入归一化、letterbox、输出 layout、objectness、NMS 和多输出张量解释。本文给出推荐的 metadata 写法和托管后处理边界。

## 预处理输入

`applications/YoloVision` 当前支持两类明确输入：

| 参数 | 输入内容 | 适用场景 |
| --- | --- | --- |
| `--input <path>` | raw byte tensor | 快速验证 byte 输入路径 |
| `--input-data <path>` | 预处理后的 float32 tensor | 推荐的真实模型样例输入 |

`--input-data` 应与 `--input-shape` 元素数量完全一致。例如 `1x3x640x640` 需要 `1*3*640*640` 个 float32。

```powershell
dotnet run --project .\applications\YoloVision -- `
  --model .\models\yolo.onnx `
  --labels .\models\labels.txt `
  --input-data .\models\input-fp32.bin `
  --input-shape 1x3x640x640 `
  --family v8 `
  --task det
```

图片 decode、resize、letterbox、BGR/RGB、归一化、padding 和 HWC/NCHW 转换建议由 owner 的预处理脚本完成，并把脚本 hash、输入图 hash 和输出 tensor hash 写进 asset manifest。

## 输出 layout

常见 detection 输出包括：

| Layout | 示例 shape | 说明 |
| --- | --- | --- |
| channel-first | `[1,84,8400]` | 通道维包含 box 和 class score |
| box-first | `[1,8400,84]` | 每个候选框一行 |
| end-to-end | `[1,N,6]` 或多输出 | NMS 已在图内或 plugin 内完成 |

如果 `--layout auto` 无法无歧义判断，应显式指定。`--has-objectness auto` 只在 class count 可推导时使用；否则应根据模型导出说明写明 objectness 规则。

## 托管后处理

YoloVision 的托管后处理覆盖：

- confidence filtering；
- class-aware / class-agnostic NMS；
- segmentation mask compose；
- pose keypoint 解释；
- OBB angle 解释；
- semantic argmax 边界；
- end-to-end 输出校验。

这些能力证明的是样例层 decode/helper 存在，不自动证明真实模型输出质量。真实模型必须用 sample-run-evidence 记录输出摘要、人工或自动比对结果、日志和 hash。

## Metadata 示例

```json
{
  "family": "v8",
  "task": "det",
  "inputShape": "1x3x640x640",
  "outputLayout": "auto",
  "classCount": 80,
  "hasObjectness": "auto",
  "nmsMode": "class-aware",
  "confidenceThreshold": 0.25,
  "iouThreshold": 0.45,
  "preprocessBoundary": "owner-provided",
  "proofClassification": "template-only"
}
```

## 边界说明

- 预处理脚本和 tensor hash 是 real-model-runtime 的必要材料。
- 后处理 helper 通过单元测试或样例运行证明代码路径，但不替代真实模型质量评估。
- build-only、parse-only、sidecar-only 都不是 inference proof。
- sample evidence 不是 public package proof，也不是 post-publish proof。
- Boundary keywords: not public package proof, not post-publish proof, not package push, not release close approval.
