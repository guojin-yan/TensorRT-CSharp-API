# ONNX 到 Engine 输出产物说明

`TensorRtExec` 和 `OnnxToEngine` 会生成 engine、report、sidecar、timing/output/profile 等不同产物。本文说明每类产物的用途和 proof 边界。

## 常见产物

| 产物 | 参数 | 用途 | 边界 |
| --- | --- | --- | --- |
| serialized engine | `--saveEngine` | TensorRT engine 输出 | build artifact |
| build report | `--exportReport` | JSON/Markdown 诊断 | build-only/precheck |
| evidence sidecar | `--evidenceSidecar` | 资产桥接记录 | sidecar-only |
| layer info | `--exportLayerInfo` | layer 诊断 | diagnostic |
| timing cache | `--timingCacheFile` / `--exportTimingCache` | 成功构建时导入/导出 cache，并记录 `TimingCacheArtifact` 大小与 SHA256 | build-cache lifecycle evidence，不是 runtime proof |
| output summary | `--exportOutput` | runtime 输出摘要 | 仅在真实 runtime 时有效 |
| raw bindings | `--dumpRawBindingsToFile` | binding dump | 需要输入输出语义 |
| reference validation | `--referenceOutputs` + tolerance/policy | 全部 output 的 name/shape/count/value 校验 | structured reference 全通过才设置 `OutputValidated` |
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

多输入与逐输出 reference JSON 的完整格式、tolerance 公式和 NaN/Infinity 策略见
[TensorRtExec 多输入与 Reference Output 校验](tensorrtexec-multi-input-reference-validation.md)。reference hash 只证明文件 identity，
不能替代实际数值比较。

## 边界说明

本地 engine、报告和 sidecar 都不是 public package proof。只有从公开包源安装到 clean consumer，再 restore/build/run 并由 validator 接受的记录，才能进入 post-publish proof 或 package-consumer-runtime proof。输出产物不能绕过 release evidence classification audit。Boundary keywords: not public package proof, not post-publish proof, not package push, not release close approval.
