# 样例证据分层：precheck/build/runtime/proof

TensorRtSharp4.0 的样例已经不只是“能不能编译”的演示工程。它们同时承担用户入门、模型转换、真实资产接入和发布证据回填的职责。为了避免把不同层级的证据混在一起，本项目把样例和发布证据分成几个固定等级：`precheck`、`build-only`、`synthetic-input-runtime`、`real-model-runtime` 和 `package-consumer-runtime`。

这套分层的目的很简单：让用户知道自己手里的报告到底证明了什么，也让发布前检查能够拒绝过度宣传。一个 ONNX build report 很有价值，但它不是推理正确性证明；一个 YoloVision 真实模型日志也很有价值，但它仍不是 NuGet package consumer runtime proof。

## 证据等级

| 等级 | 证明内容 | 不能证明 |
| --- | --- | --- |
| `precheck` | 命令、shape、报告路径或 manifest 模板可解析 | TensorRT runtime 可用、ONNX 可构建、推理可运行 |
| `build-only` | ONNX parser/builder 走到构建或报告边界 | 输出正确、真实模型质量、package consumer runtime |
| `dependency-probe-only` | 依赖探测、DLL 路径或 runtime probing 结果 | TensorRT enqueue 已执行 |
| `synthetic-input-runtime` | 使用 synthetic input 的最小管线执行 | 真实图片、真实 labels 或真实模型质量 |
| `real-model-runtime` | 真实模型、labels、输入资产、hash、许可证、runner log 和 sample-run-evidence 对齐 | NuGet 包消费端 proof |
| `package-consumer-runtime` | 外部消费项目从包源 restore/build/run，并由 release proof record 验证 | 不能由 sample report、sidecar 或 manifest 单独声明 |

`blocked-by-cuda-driver` 不是 smoke passed。它表示当前 host 的 CUDA driver/runtime 不兼容目标运行线，应该作为 owner action 或兼容主机待执行项保留，而不是改写成 API 缺口或通过证据。

## TensorRtExec 的证据位置

`applications/TensorRtExec` 适合生成 build/precheck 证据。常见命令：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model.plan `
  --minShapes input:1x3x640x640 `
  --optShapes input:1x3x640x640 `
  --maxShapes input:4x3x640x640 `
  --fp16 `
  --workspace 512 `
  --buildOnly `
  --exportReport .\models\model-build-report.json
```

如果报告中出现：

```text
ProofClassification=build-only
BuildEvidenceOnly=True
InferenceRan=False
IsRuntimeExecutionProof=False
```

它说明外部 ONNX 的 parser/builder/build report 路径有记录，但不说明这个模型已经完成端到端推理。外部模型的输入名、输出名、shape、后处理、NMS、labels 和图片预处理都需要具体 sample runner 或应用补齐。

`--dryRun` 或 `--previewOnly` 更早一层，只证明参数归一化和报告生成边界，不读取 ONNX，不构建 engine，不探测 TensorRT runtime。

## OnnxToEngine 的证据位置

`samples/OnnxToEngine` 是最小 identity ONNX round-trip 样例。它可以生成内置 dynamic identity ONNX，并在兼容环境下验证 parser、optimization profile、serialized engine、deserialize、binding 和 readback。

这条路径证明项目自己的最小 ONNX 构建与执行链路，不证明任意外部 ONNX 的语义都已经被项目理解。外部模型仍应先通过 TensorRtExec 形成 build report，再由 Classification、YoloVision 或用户自己的 binding 代码补真实输出语义。

## Classification 与 YoloVision 的晋级路径

Classification 和 YoloVision 是 asset-dependent samples。它们可以晋级到 `real-model-runtime`，但前提是 owner 提供并记录：

- ONNX 模型来源、许可证、SHA256、opset 和导出命令。
- labels 文件来源、许可证、SHA256 和类别数。
- 输入图片或输入数据来源、许可证、SHA256 和预处理规则。
- Tensor 名称、shape、layout、dtype 和输出解释方式。
- TensorRtExec build-only report 和 evidence sidecar。
- sample runner 的真实日志、日志 SHA256、stdout/stderr 摘要。
- `sample-run-evidence-record` validator 通过结果。

YoloVision 还需要记录 family/task/profile、输出 layout、objectness 规则、NMS 模式，以及 seg/pose/obb/sem 等多输出 metadata。只有真实 `YoloVision Passed=True` 日志与 manifest、sidecar、sample-run-evidence 一致时，才可以把样例层证据写成 `real-model-runtime`。

## sample-run-evidence 的边界

`sample-run-evidence` 文件连接真实 runner log 和 asset manifest。它最多把 Classification 或 YoloVision 晋级到 sample-level `real-model-runtime`。它不能声明 `package-consumer-runtime`，也不能替代 release proof record。

校验命令：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -InputPath .\models\yolovision-sample-run-evidence.json -RequireExistingLog
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleAssetManifest.ps1
```

如果缺少真实日志、hash 或许可证，应该保持 owner-action-required，而不是把 `isSmokePassed` 改成 `true`。

## release proof 的边界

`package-consumer-runtime` belongs to release proof records。它需要外部消费项目从包源安装管理包和 runtime 包，执行 clean smoke，并由 `external-runtime-proof-record.json` 通过严格校验。

当前主机如果仍是 `blocked-by-cuda-driver`，正确做法是保留 blocker，并在兼容 CUDA host 上执行 owner action。runbook、collection bundle、draft、template、example、dependency probe、build-only report 都不能提升为 runtime proof。

## 发布前自查

发布前建议搜索：

```powershell
rg -n "package-consumer-runtime|blocked-by-cuda-driver|Passed=True|build-only|dependency-probe-only|sample-run-evidence" .\docs .\samples .\applications .\eng
```

看到这些词时要确认上下文：它们是在解释边界，还是在过度宣称通过。正确的发布材料应该让用户清楚地知道当前证据处在哪一层。

## 小结

样例证据分层不是保守措辞，而是让项目可发布、可维护、可审计的基础。TensorRtExec build report 负责构建证据，OnnxToEngine 负责最小 round-trip，Classification/YoloVision 负责真实模型样例证据，package consumer runtime proof 则由 release proof record 负责。把边界讲清楚，项目才能既可用又可信。
