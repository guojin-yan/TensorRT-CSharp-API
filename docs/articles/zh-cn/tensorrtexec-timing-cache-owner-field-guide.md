# TensorRtExec Timing Cache Owner Field Guide

本文面向 release owner 和模型负责人，说明 `TensorRtExec` 中 timing cache 相关参数如何收集、校验和进入发布证据链。成功构建时，`--timingCacheFile`、`--timingCache` 和 `--exportTimingCache` 会通过 typed `TensorRtTimingCache` owner 完成导入/导出，并在报告中记录大小与 SHA256；dry-run、load-engine 和依赖不可用路径会明确记录为未应用。它们仍不能单独成为 `package-consumer-runtime`、`real-model-runtime` 或 post publish proof。

## Owner 需要确认什么

| 字段 | Owner 输入 | 可接受证据 | 不能替代 |
| --- | --- | --- | --- |
| cache 输入路径 | `--timingCacheFile <path>` 或 `--timingCache <path>` | 成功构建时导入 cache，报告记录路径、大小和 SHA256 | runtime inference proof |
| cache 输出路径 | `--exportTimingCache <path>` | 成功构建后序列化并写出 cache，报告记录路径、大小和 SHA256 | runtime inference proof |
| cache content hash | SHA256 | owner 提供构建前后 hash、大小和更新时间 | package-consumer-runtime |
| TensorRT/CUDA 环境 | TensorRT version、CUDA version、driver、GPU | 真实主机日志、diagnostics、owner review | sidecar-only report |
| 结果等级 | `applied-build-cache-lifecycle` 或 `build-only` | `TimingCacheArtifact`、`OptionImplementationStatus`、build report、owner field record | clean consumer proof |

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

这条命令最多证明 TensorRtExec 在兼容 TensorRT 环境中完成 build-cache lifecycle 并生成构建报告。只有在真实 engine 构建日志、cache content hash 和模型级验证都齐备时，owner 才能把它作为更高等级证据的组成部分。

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

- 必须有 `--timingCacheFile` 或 `--exportTimingCache` 的 normalized command 记录和 `TimingCacheArtifact` 状态。
- 必须有 cache content hash、文件大小、最后修改时间和 owner review。
- 必须能区分 import cache、export cache、复用 cache 三个场景。
- 必须有 native import/export smoke 或真实 TensorRT 构建日志，不能只依赖 dry-run 或 sidecar-only report。
- 必须保留 build-cache lifecycle 不等于 runtime proof 的边界，直到模型级 run log 和输出校验完成。

## 禁止晋级规则

以下材料不能单独晋级为 proof：

- `parse/report-only`（未执行的路径）
- `build-only`
- `dry-run`
- `sidecar-only`
- `blocked-by-cuda-driver`
- `direct .nupkg`
- `ProjectReference`
- `local feed`
- `not runtime proof`

Timing cache 是性能和构建复用能力，不是推理正确性的证明。发布前仍需要 `package-consumer-runtime` 记录、真实 consumer smoke、模型输出校验或 post publish verification。
