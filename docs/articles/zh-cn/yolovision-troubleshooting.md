# YoloVision 排障指南

YoloVision 涉及模型导出、TensorRT 构建、输入预处理、输出 layout 和后处理。失败时不要先改 wrapper；先按本文把问题归类，避免把资产问题、驱动问题或 metadata 问题误判为发布阻断。

## 常见问题

| 现象 | 优先检查 | 处理方式 |
| --- | --- | --- |
| 找不到 native bridge | `JYPPX_ENABLE_DEVELOPMENT_PROBING`、runtime assets | 先运行 package/runtime 说明中的本地探测步骤 |
| CUDA error 35 | driver/runtime 不兼容 | 记录 `blocked-by-cuda-driver`，换兼容环境 |
| ONNX parser 失败 | opset、unsupported op、plugin | 用 TensorRtExec 导出 build report |
| 输入元素数量不匹配 | `--input-shape` 与 `.bin` 元素数 | 重新生成预处理 tensor |
| 输出 shape 无法识别 | `--layout auto` 歧义 | 显式指定 layout、class count、objectness |
| detection 过多或过少 | confidence / IoU / NMS mode | 调整 `--confidence`、`--iou-threshold`、`--nms-mode` |
| segmentation mask 异常 | prototype shape、mask coefficient | 补齐多输出 metadata |
| pose keypoint 异常 | keypoint count、visibility | 检查 task/profile 是否匹配 |
| OBB 角度异常 | angle unit、angle channel | 明确 radians/degrees 和 channel index |

## 排查命令顺序

1. 离线能力矩阵：

```powershell
dotnet run --project .\samples\YoloVision -- --list-capabilities
```

2. TensorRtExec dry-run：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model.plan `
  --previewOnly `
  --exportReport .\models\precheck.md
```

3. TensorRtExec build-only：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model.plan `
  --buildOnly `
  --exportReport .\models\build-report.json
```

4. YoloVision sample run：

```powershell
dotnet run --project .\samples\YoloVision -- `
  --model .\models\model.onnx `
  --labels .\models\labels.txt `
  --input-data .\models\input-fp32.bin `
  --input-shape 1x3x640x640 `
  --family v8 `
  --task det
```

## 证据记录

排障记录至少保留：

- 命令；
- stdout/stderr；
- ONNX、labels、input、report、log 的 SHA256；
- driver/runtime/TensorRT line；
- failure classification；
- 是否为 `blocked-by-cuda-driver`；
- 是否仍是 `template-only`、`build-only`、`synthetic-input-runtime` 或 `real-model-runtime`。

## 边界说明

排障文章不是 proof 生成器。它不能批准公开发布，不能关闭 release issue，也不能把 build-only、parse-only、sidecar-only、local feed 或 ProjectReference 推广为 public package proof。Boundary keywords: not public package proof, not post-publish proof, not package push, not release close approval.
