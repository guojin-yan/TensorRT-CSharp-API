# ONNX 到 Engine FP16 / INT8 边界

TensorRtSharp4.0 的转换工具支持常见 precision 参数，但不同 precision 的完成度不同。本文说明 FP16、BF16、TF32、INT8 在 `TensorRtExec` 和 `OnnxToEngine` 中的边界，避免把 parse/report-only 参数写成真实 runtime proof。

## 参数概览

| 参数 | 当前用途 | 边界 |
| --- | --- | --- |
| `--fp16` | 进入 builder config | 可作为 build/report evidence |
| `--bf16` | 进入 precision intent 或支持线检查 | 依赖 TensorRT line 和硬件 |
| `--noTF32` | 记录 TF32 策略 | 不等于逐层 precision proof |
| `--int8` | 记录 INT8 intent | calibrator callback 仍需单独证明 |
| `--calib <path>` | 记录 calibration cache 路径 | 当前不是完整 calibrator proof |
| `--precisionConstraints` | parse/report-only | 不声明 layer precision routing 已完成 |
| `--layerPrecisions` | parse/report-only | 不声明每层 precision 已真实执行 |
| `--layerOutputTypes` | parse/report-only | 不声明输出类型已真实路由 |

## FP16 build-only 示例

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model-fp16.plan `
  --minShapes input:1x3x224x224 `
  --optShapes input:4x3x224x224 `
  --maxShapes input:8x3x224x224 `
  --fp16 `
  --buildOnly `
  --exportReport .\models\model-fp16-report.json
```

## INT8 当前建议

INT8 需要校准数据、calibrator 生命周期、cache 读取/写入和模型输出验证。若这些链路未全部完成，报告必须保持 build/report 或 parse/report 边界：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model-int8.plan `
  --int8 `
  --calib .\models\calibration.cache `
  --buildOnly `
  --exportReport .\models\model-int8-boundary.md
```

## 边界说明

Precision 参数不是发布证明。FP16 build 成功不代表输出数值正确；INT8 参数解析不代表 calibrator 已安全执行；layer precision policy 当前仍需模型专属验证。任何 precision 报告都不能替代 public package proof、post-publish proof 或 release close approval。
