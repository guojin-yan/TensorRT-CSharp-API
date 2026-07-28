# TensorRtExec 多输入与 Reference Output 校验

## 运行合同

generic bounded runtime 支持一个或多个 float inputs 和一个或多个 float outputs。输入、输出与工件都保持
`TensorRtEngineBindingReport` 的 engine 顺序。使用 `--loadInputs` 时必须提供完整的 `tensor:path` 映射；缺失、
重复或未知 tensor name 会直接失败。未提供该参数时，每个 input 独立生成确定性 float 数据。

```powershell
TensorRtExec `
  --loadEngine .\model.plan `
  --shapes "left:2x4,right:2x4" `
  --loadInputs "left:.\left.bin,right:.\right.bin" `
  --referenceOutputs "sum:.\sum.reference.json,difference:.\difference.reference.json" `
  --referenceAbsTolerance 1e-5 `
  --referenceRelTolerance 1e-4 `
  --referenceNaNPolicy reject `
  --referenceInfinityPolicy exact `
  --exportOutput .\output.json
```

`InputTensors` 为每个输入保存 name、shape、element/byte count、最多 8 个 preview values、SHA256、
`SourceClassification` 与 `SourcePath`。这些字段都是 managed copy，不包含 device pointer 或 borrowed handle。

## Reference JSON

每个 output 使用独立、可追溯的 JSON：

```json
{
  "schemaVersion": 1,
  "tensorName": "sum",
  "shape": [2, 4],
  "values": [2.25, 5.25, 8.25, 11.25, 14.25, 17.25, 20.25, 23.25],
  "sourceClassification": "synthetic-generated"
}
```

校验顺序为 mapping、文件读取、tensor name、shape、element count、逐值比较。有限值满足以下任一条件即通过：

- `abs(actual - expected) <= absoluteTolerance`
- `abs(actual - expected) <= relativeTolerance * max(abs(actual), abs(expected))`

NaN 默认全部拒绝；`equal` 只允许两侧同时为 NaN。Infinity 的 `exact` 要求符号完全一致，`reject` 则拒绝任意
Infinity。每个 tensor 都记录 reference path/hash/source、actual/reference shape/count、mismatch count、first mismatch、
最大绝对/相对误差和 diagnostic。

## 结果边界

只有 mapping 覆盖全部 engine outputs、所有 reference 可读且所有 tensor 都通过，`OutputValidated` 才为 true。
identity input/output 快捷比较仍保留为 `IdentityOutputMatch`，不会单独提升 `OutputValidated`。output capture、raw bytes、
reference SHA256 或单个 tensor 通过都不能替代全输出数值校验。

本仓库的 TRT10/CUDA12.9 smoke 使用生成的 Add/Sub ONNX 完成双输入、双输出真实 enqueue/readback；随后把
`difference[7]` reference 修改 `0.25`，结果正确进入 `load-engine-reference-validation-failed`。该证据仍是 synthetic
runtime，不是 real-model、package-consumer、public package、post-publish 或 release proof。
