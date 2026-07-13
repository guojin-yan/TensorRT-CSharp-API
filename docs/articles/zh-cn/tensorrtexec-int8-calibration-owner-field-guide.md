# TensorRtExec INT8 Calibration Owner Field Guide

本文说明 `TensorRtExec` 的 INT8 calibration 字段如何进入 owner 审核。核心边界是：`--int8` 和 `--calib` 可以表达 INT8 intent、cache 路径和报告字段，但在 calibrator ownership、callback ownership、数据集来源和模型精度结果齐备前，不能声明为完整 INT8 runtime proof。

## Owner 需要确认什么

| 字段 | Owner 输入 | 可接受证据 | 不能替代 |
| --- | --- | --- | --- |
| INT8 开关 | `--int8` | normalized command、build report、precision intent | model-specific INT8 accuracy evidence |
| calibration cache | `--calib <path>` | cache 路径、SHA256、大小、更新时间 | calibrator callback runtime proof |
| calibration dataset provenance | 数据集名称、版本、来源、样本数 | owner review、数据集清单、hash 或下载记录 | package-consumer-runtime |
| calibrator ownership | cache 由谁生成、何时生成、适用模型 | owner field record、生成日志 | callback ownership |
| 精度结果 | FP32/FP16/INT8 对照指标 | 模型专属评估日志、输出样本、阈值说明 | parse/report-only |

## 推荐命令

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --saveEngine .\artifacts\models\model-int8.plan `
  --minShapes input:1x3x640x640 `
  --optShapes input:1x3x640x640 `
  --maxShapes input:1x3x640x640 `
  --int8 `
  --calib .\artifacts\models\model-int8.calib.cache `
  --buildOnly `
  --exportProfile .\artifacts\models\model-int8-report.json
```

这条命令只能形成 INT8 build/report 证据。若没有真实 calibrator 生命周期、calibration dataset provenance、cache content hash、模型专属精度验证和 owner review，它仍然是 `not runtime proof`。

## 记录模板

| 字段 | 示例 | 必填 |
| --- | --- | --- |
| `modelId` | `yolov8n-det-640` | 是 |
| `onnxSha256` | 64 位小写 SHA256 | 是 |
| `calibrationCachePath` | `artifacts/models/model-int8.calib.cache` | 是 |
| `calibrationCacheSha256` | 64 位小写 SHA256 | 是 |
| `calibrationDatasetName` | `coco-val2017-calib-subset` | 是 |
| `calibrationDatasetProvenance` | 下载源、内部资产编号或 owner 记录 | 是 |
| `calibrationSampleCount` | `500` | 是 |
| `calibratorOwnership` | `owner-provided-cache` 或 `native-calibrator-run` | 是 |
| `callbackOwnership` | `not-used`、`managed-callback-owned` 或 `native-owned` | 是 |
| `accuracyEvidencePath` | 模型指标日志或对照报告 | 是 |
| `proofClassification` | `parse/report-only` 或 `build-only` | 是 |

## 提升到真实证据前的检查

- 必须同时记录 `--int8`、`--calib`、cache SHA256 和 calibration dataset provenance。
- 必须说明 calibration cache 是 owner 提供、TensorRT 生成，还是 callback calibrator 生成。
- 若涉及 callback calibrator，必须先完成 callback ownership、异常边界、释放顺序和跨 ABI no-throw 设计验证。
- 必须提供 model-specific INT8 accuracy evidence，至少包含 FP32/FP16/INT8 对照、阈值和失败处理。
- 必须明确 `parse/report-only` 与 `build-only` 不等于 `package-consumer-runtime`。

## 禁止晋级规则

以下材料不能单独晋级为 proof：

- `--int8` 参数解析成功。
- `--calib` 路径进入 report。
- calibration cache 文件存在但没有来源、hash 或适配模型记录。
- build report、sidecar-only report、dry-run、template 或 local feed。
- 没有 callback ownership 的 calibrator 路径。
- 没有模型专属精度结果的 INT8 engine。

INT8 的完成标准不是“参数能传进去”，而是 calibration 数据、cache 生命周期、calibrator ownership、callback ownership、模型输出精度和 release proof 全链路闭合。
