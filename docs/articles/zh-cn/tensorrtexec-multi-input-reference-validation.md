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

## Classification / YoloVision 共享样例合同

`samples/_shared/JYPPX.SampleSupport` 现在使用同样的严格名称绑定，不再把 Classification/YoloVision 限制为一个 ONNX
input。样例命令采用 kebab-case 参数：

- `--input-shapes name:dims,...`：必须覆盖全部 model inputs；
- `--min-shapes`、`--opt-shapes`、`--max-shapes`：任意一个出现时，三份 map 必须同时完整；
- `--load-inputs`、`--load-byte-inputs`、`--input-patterns`：每个 input 必须且只能命中一种来源；
- `--reference-outputs name:path,...`：必须覆盖全部 captured outputs；
- `--reference-abs-tolerance`、`--reference-rel-tolerance` 与显式 NaN/Infinity policy。

旧的 `--input-name`、`--input-shape`、`--input-data` 等单输入参数继续兼容，但不能与复数 map 混用。运行结果按
engine 顺序保存 `Inputs`/`Outputs`，YoloVision JSON 写入 `inputTensors` 和 `referenceValidation`，Classification JSON
同时区分任务级 `referenceValidation` 与 raw runtime `runtimeReferenceValidation`。两种 Classification 校验同时请求时
必须全部通过。

同一条 TRT10/CUDA12.9 Add/Sub smoke 现在也执行共享样例层：成功分支为 2 inputs、2 outputs、2 comparisons 全通过；
负向分支固定得到 `MismatchCount=1`、`FirstMismatchIndex=7`、`MaximumAbsoluteError=0.25`。这证明实际 build、enqueue、
readback 与 fail-closed 行为，不改变 synthetic runtime 的证据分类。

smoke summary 把总体与单 tensor 完成状态分开命名为 `ReferenceValidationCompleted`、
`ReferenceValidationPassed` 和 `ComparisonCompleted`，避免匿名字段重名，也避免把 metadata 不可比较误写成数值比较完成。

## Reference JSON

每个 output 使用独立、可追溯的 JSON：

共享样例对应 schema 为 `samples/_shared/JYPPX.SampleSupport/onnx-sample-reference.schema.json`。

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

## MNIST Reference 候选

仓库本机另以 TensorRT 10.11/CUDA 12.9 的既有 MNIST digit-7 资产验证了同一合同。模型随 TensorRT
`data/mnist` 提供，其 README 标记来源为 ONNX Model Zoo；输入为 `7.pgm` 经 `1-pixel/255` 预处理后的
`Input3` float32 tensor。reference 仅包含 `Plus214_Output_0` 的 `[1,10]` 十个 logits，文件的
`sourceClassification` 固定为 `repository-mnist-runtime-output-derived-unreviewed`。

同运行时 reference 从先前 TensorRT 输出复制而来，因此它只用于回归一致性。source-tree build 和独立
`--loadEngine` 都得到 `OutputValidated=true`、10/10 比较、0 mismatch，最大绝对/相对误差分别为
`9.536743e-07` / `1.3443339e-06`，使用 `1e-4` 的 absolute/relative tolerance。隔离的本地
`PackageReference` consumer 同样比较该 reference，并由 strict evidence validator 记录 53 项通过。该层记录位于
`artifacts/interface-coverage/tensorrtexec-mnist-reference-validation-evidence.json`。

## 独立 ONNX Runtime CPU 候选

隔离 producer 另使用 ONNX Runtime `1.23.2` 的 `CPUExecutionProvider` 执行相同 ONNX 和输入。runner 只从本机已有
NuGet cache 复制四个 `.nupkg` 到 E 盘临时 feed，`NuGet.Config` 清空全部远程源，restore cache 与 `DOTNET_CLI_HOME`
也位于 E 盘隔离 workspace；主 solution 不增加 ONNX Runtime 依赖。两次 ORT 输出的 float32 bytes 完全一致，raw SHA256
为 `a20932857fb2d51f5f0b79daa211140fce631b3f67787b69e0a23f33c8817d75`，预测 digit 7。profiling trace 中只出现
`CPUExecutionProvider`，因此执行路径独立于 TensorRT。

ORT reference 使用 `onnxruntime-cpu-1.23.2-derived-unreviewed` 分类，SHA256 为
`1babfa81c0d277a3d483fdc6288285b26008ba5b382868272f1e2187338bd571`。它与保留的 TensorRT logits 进行 10/10 比较，
0 mismatch，最大绝对/相对误差为 `5.722046e-06` / `5.7323444e-07`，均在 `1e-4` tolerance 内。compact evidence 与
strict validator 分别位于 `tensorrtexec-mnist-onnxruntime-reference-evidence.json` 和对应 validation 文件；要求本机
raw/profile/log 时为 51/51。clean clone 只保留 reference、sidecar、compact evidence 和 validation summary，不要求被
忽略的重运行工件。

## 受控负向运行

独立 ORT reference 还派生五个受控畸形副本：tensor name mismatch、shape mismatch、value count mismatch、NaN/reject
和 Infinity/reject。每个副本都在 source-tree CLI 与隔离 local-feed `PackageReference` consumer 上运行，共 10 次真实
TensorRT enqueue/readback。两条路径都先得到相同 raw output SHA256，再以非零退出码结束且
`OutputValidated=false`；consumer 还记录 `OwnerScopeExited=true`。元数据三例为 `Completed=false`，两个特殊值例为
`Completed=true`、1 mismatch、first mismatch 0。严格 validator 为 72/72。

这三层记录必须分开解释：同运行时 reference 是回归基线；ORT CPU 是独立框架候选；受控负例只证明 fail-closed。
TensorRT sample 条款仍只是许可审查输入，Owner 尚未批准模型/输入/reference 的仓库再分发，也未接受 ORT 候选为
golden reference。本地 feed、CPU profiling、真实 GPU enqueue 与 reference hash 均不是 public-package、post-publish、
Owner accepted real-model 或 release proof。
