# YoloVision Engine 构建与运行

本文串联 YoloVision 的 engine 构建、样例运行和证据回填。推荐流程是先用 `TensorRtExec` 做 build-only，再用 `applications/YoloVision` 跑真实模型输入，最后把 sample-run-evidence 与 release proof record 分开保存。

## 1. 生成 build-only 报告

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\yolovision\model.onnx `
  --saveEngine .\models\yolovision\model.plan `
  --minShapes images:1x3x640x640 `
  --optShapes images:1x3x640x640 `
  --maxShapes images:4x3x640x640 `
  --fp16 `
  --workspace 1GiB `
  --buildOnly `
  --exportReport .\models\yolovision\build-report.json
```

这一步输出 build/report evidence。它可以说明 ONNX、shape profile、precision、workspace 和 engine 输出路径，但不证明 YoloVision 后处理结果正确。

## 2. 运行 YoloVision

```powershell
dotnet run --project .\applications\YoloVision -- `
  --model .\models\yolovision\model.onnx `
  --labels .\models\yolovision\labels.txt `
  --input-data .\models\yolovision\input-fp32.bin `
  --input-shape 1x3x640x640 `
  --tensor-rt-line 11 `
  --family v8 `
  --task det `
  --layout auto `
  --confidence 0.25 `
  --iou-threshold 0.45
```

如果当前机器缺少兼容 CUDA driver/runtime，应记录 `blocked-by-cuda-driver`，不要把阻塞状态写成 smoke passed。

## 3. 回填 sample evidence

真实样例 evidence 应包含：

- 命令行；
- TensorRT line、CUDA runtime、driver；
- ONNX/labels/input/log SHA256；
- build-only report 路径和 hash；
- stdout/stderr 摘要；
- 关键输出数量，例如 detection count、Top-K、mask count、keypoint count；
- owner 对模型 license 和资产来源的确认；
- `proofClassification=real-model-runtime` 仅在真实运行和验证完成后才允许设置。

## 4. 何时可以晋级

| 状态 | 可以说明 | 不能说明 |
| --- | --- | --- |
| `template-only` | 资产清单结构存在 | 模型可运行 |
| `build-only` | engine 构建报告存在 | 推理输出正确 |
| `synthetic-input-runtime` | 管线可执行 | 真实图片质量 |
| `real-model-runtime` | 该样例真实模型运行 | public package 已发布 |
| `package-consumer-runtime` | 外部包消费者验证 | 不属于样例直接产生 |

## 边界说明

YoloVision engine build/run 文章只服务 sample adoption。它不是 publish approval，不是 release close approval，不是 package push，也不是 post-publish verification。真实公开包验证必须由独立 clean consumer 从公开包源 restore/build/run 后回填。Boundary keywords: not public package proof, not post-publish proof, not package push, not release close approval.
