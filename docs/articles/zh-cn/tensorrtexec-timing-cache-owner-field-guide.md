# TensorRtExec Timing Cache Owner Field Guide

本文面向 release owner 和模型负责人，说明 `TensorRtExec` 中 timing cache 相关参数如何收集、校验和进入发布证据链。结论很简单：`--timingCacheFile`、`--timingCache`、`--exportTimingCache` 当前可以形成 parse/report-only 或 build-only 证据，但不能单独成为 `package-consumer-runtime`、`real-model-runtime` 或 post publish proof。

## Owner 需要确认什么

| 字段 | Owner 输入 | 可接受证据 | 不能替代 |
| --- | --- | --- | --- |
| cache 输入路径 | `--timingCacheFile <path>` 或 `--timingCache <path>` | 文件存在、路径进入 normalized command、build report 记录导入意图 | native import/export smoke |
| cache 输出路径 | `--exportTimingCache <path>` | report 记录输出路径、构建后可计算 hash | runtime inference proof |
| cache content hash | SHA256 | owner 提供构建前后 hash、大小和更新时间 | package-consumer-runtime |
| TensorRT/CUDA 环境 | TensorRT version、CUDA version、driver、GPU | 真实主机日志、diagnostics、owner review | sidecar-only report |
| 结果等级 | `parse/report-only` 或 `build-only` | `OptionImplementationStatus`、build report、owner field record | clean consumer proof |

## 推荐命令

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --saveEngine .\artifacts\models\model.plan `
  --minShapes input:1x3x640x640 `
  --optShapes input:1x3x640x640 `
  --maxShapes input:1x3x640x640 `
  --timingCacheFile .\artifacts\models\model.timing.cache `
  --exportTimingCache .\artifacts\models\model.exported.timing.cache `
  --buildOnly `
  --exportProfile .\artifacts\models\model-timing-cache-report.json
```

这条命令最多证明 TensorRtExec 已接收 timing cache 参数并生成构建报告。只有在 native import/export smoke、真实 engine 构建日志、cache content hash 和模型级验证都齐备时，owner 才能把它作为更高等级证据的组成部分。

## 记录模板

| 字段 | 示例 | 必填 |
| --- | --- | --- |
| `modelId` | `yolov8n-det-640` | 是 |
| `onnxSha256` | 64 位小写 SHA256 | 是 |
| `timingCacheInputPath` | `artifacts/models/model.timing.cache` | 否 |
| `timingCacheInputSha256` | 64 位小写 SHA256 | 有输入 cache 时必填 |
| `timingCacheOutputPath` | `artifacts/models/model.exported.timing.cache` | 否 |
| `timingCacheOutputSha256` | 64 位小写 SHA256 | 有输出 cache 时必填 |
| `tensorRtVersion` | `11.x` | 是 |
| `cudaVersion` | `13.x` | 是 |
| `gpuName` | 真实 GPU 名称 | 是 |
| `ownerReviewer` | 审核人 | 是 |
| `proofClassification` | `parse/report-only` 或 `build-only` | 是 |

## 提升到真实证据前的检查

- 必须有 `--timingCacheFile` 或 `--exportTimingCache` 的 normalized command 记录。
- 必须有 cache content hash、文件大小、最后修改时间和 owner review。
- 必须能区分 import cache、export cache、复用 cache 三个场景。
- 必须有 native import/export smoke 或真实 TensorRT 构建日志，不能只依赖 sidecar-only report。
- 必须保留 `parse/report-only` 边界，直到模型级 run log 和输出校验完成。

## 禁止晋级规则

以下材料不能单独晋级为 proof：

- `parse/report-only`
- `build-only`
- `dry-run`
- `sidecar-only`
- `blocked-by-cuda-driver`
- `direct .nupkg`
- `ProjectReference`
- `local feed`
- `not runtime proof`

Timing cache 是性能和构建复用能力，不是推理正确性的证明。发布前仍需要 `package-consumer-runtime` 记录、真实 consumer smoke、模型输出校验或 post publish verification。
