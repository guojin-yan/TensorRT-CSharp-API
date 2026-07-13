# YoloVision Segmentation 教程

YOLO segmentation 比 detection 多了 mask prototype 和 mask coefficients。本文说明如何把多输出 metadata 写清楚，并把 build-only、sidecar-only 和 real-model-runtime 的边界分开。

## 资产要求

owner 需要准备：

- segmentation ONNX。
- labels。
- input image。
- mask output role 说明。
- prototype tensor shape。
- coefficient tensor 与 detection tensor 的对应关系。
- license 和 SHA256。

## 多输出 metadata

seg 模型通常至少有两类输出：

| role | 说明 |
| --- | --- |
| boxes | box、score、class、mask coefficient |
| mask-prototypes | mask basis/prototype tensor |

不要只写“第二个输出是 mask”。应记录 tensor name、shape、dtype、role、layout 和 resize/裁剪策略。

## 构建与运行

TensorRtExec build-only：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\yolo-seg.onnx `
  --saveEngine .\models\yolo-seg.engine `
  --fp16 `
  --buildOnly `
  --exportProfile .\models\yolo-seg-build-report.json
```

YoloVision 运行时需要 task=seg，并提供多输出 metadata。具体字段应以样例 README 和 asset manifest template 为准，不要让程序猜测模型语义。

## Evidence 要求

- build report 是 `build-only`，不证明 mask 正确。
- sidecar 是 `sidecar-only`，用于桥接 build report 与资产信息。
- sample-run-evidence 需要真实 runner log、mask 输出摘要和 SHA256。
- 只有真实模型、真实输入、真实日志和 validator 通过后，才可能成为 `real-model-runtime`。
- 它仍不是 `package-consumer-runtime`。

## 常见问题

| 问题 | 检查点 |
| --- | --- |
| mask 全黑 | prototype shape、coefficient 顺序、resize/crop |
| 框正确但 mask 偏移 | letterbox 还原与 mask resize 策略 |
| class 错位 | label count 与 class count |
| 多输出缺失 | tensor role 是否写入 metadata |

## 边界词

本文涉及的证据边界包括 `build-only`、`parse-only`、`sidecar-only`、`blocked-by-cuda-driver`、`real-model-runtime`、`package-consumer-runtime`、`owner action`。这些词不是装饰，而是避免把样例运行、模型资产和 release proof 混在一起。
