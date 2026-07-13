# ONNX 到 Engine 输出产物说明

`TensorRtExec` 和 `OnnxToEngine` 会生成 engine、report、sidecar、timing/output/profile 等不同产物。本文说明每类产物的用途和 proof 边界。

## 常见产物

| 产物 | 参数 | 用途 | 边界 |
| --- | --- | --- | --- |
| serialized engine | `--saveEngine` | TensorRT engine 输出 | build artifact |
| build report | `--exportReport` | JSON/Markdown 诊断 | build-only/precheck |
| evidence sidecar | `--evidenceSidecar` | 资产桥接记录 | sidecar-only |
| layer info | `--exportLayerInfo` | layer 诊断 | diagnostic |
| timing cache | `--timingCacheFile` / `--exportTimingCache` | cache 路径记录 | parse/report-only |
| output summary | `--exportOutput` | runtime 输出摘要 | 仅在真实 runtime 时有效 |
| raw bindings | `--dumpRawBindingsToFile` | binding dump | 需要输入输出语义 |
| profile/times | `--exportProfile` / `--exportTimes` | timing/profile 数据 | 不自动等于 release proof |

## 报告字段

报告应包含：

- `ProofClassification`；
- `BuildEvidenceOnly`；
- `DryRun`；
- `IsRuntimeExecutionProof`；
- `IsRealModelRuntimeProof`；
- `IsPackageConsumerRuntimeProof`；
- `NormalizedCommandSha256`；
- `ArtifactProofBoundary`；
- `PreflightMetadata`；
- `OptionImplementationStatus`。

## 推荐归档方式

```text
models/
  model.onnx
  model.plan
  model-build-report.json
  model-evidence.sidecar.json
  model-run.log
  model-output.json
```

每个文件都应在 owner evidence 中记录 SHA256。没有 hash 的报告只能作为人工参考，不应作为可关闭 release issue 的证据。

## 边界说明

本地 engine、报告和 sidecar 都不是 public package proof。只有从公开包源安装到 clean consumer，再 restore/build/run 并由 validator 接受的记录，才能进入 post-publish proof 或 package-consumer-runtime proof。输出产物不能绕过 release evidence classification audit。Boundary keywords: not public package proof, not post-publish proof, not package push, not release close approval.
