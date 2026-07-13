# TensorRtExec 工具入门

`applications/TensorRtExec` 是 TensorRtSharp4.0 面向最终用户的 ONNX 到 TensorRT engine 工具。它提供两个入口：命令行入口用于自动化构建和日志采集，WinForms 入口用于在 Windows 桌面上选择 ONNX、engine、shape profile 和报告路径。

这一阶段的重点是把应用从“参数预检骨架”提升到“复用真实工具库构建服务”。也就是说，CLI 和 WinForms 不再各自维护一套影子逻辑，而是通过 `JYPPX.TensorRtSharp.Tools` 中的 `OnnxEngineBuildService` 走同一条 parser、builder config、serialized engine、report 输出路径。

## 适用场景

TensorRtExec 当前适合以下工作：

- 快速把外部 ONNX 模型构建成 TensorRT serialized engine。
- 记录 TensorRT line、workspace、precision、shape profile、engine path 和日志。
- 生成 JSON 或 Markdown 构建报告，方便回填到发布证据或问题排查记录。
- 用内置 identity ONNX 路径做最小 round-trip 验证。
- 在桌面 UI 中给非命令行用户提供同一套构建入口。

它当前不承担以下证明：

- 不证明 NuGet 包已经公开发布。
- 不证明任意外部 ONNX 模型已经完成端到端推理。
- 不主动加载 plugin library，也不注册或注销 TensorRT plugin。
- 不把 build-only 报告写成 runtime execution proof。

## CLI 示例

外部 ONNX 的常见 build-only 命令如下：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model.plan `
  --minShapes input:1x3x640x640 `
  --optShapes input:1x3x640x640 `
  --maxShapes input:4x3x640x640 `
  --fp16 `
  --workspace 512 `
  --exportReport .\models\model-build-report.json `
  --evidenceSidecar .\models\model-evidence.sidecar.json `
  --buildOnly
```

如果模型没有动态输入，也可以省略 shape profile；如果模型包含动态维度，应显式提供 `--minShapes`、`--optShapes` 和 `--maxShapes`。这一阶段把外部 ONNX 默认视作 build/report 证据，除非后续补齐模型绑定名、输出语义、后处理和真实输入资产。

加载已有 engine 的预检命令：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --loadEngine .\models\model.plan `
  --exportReport .\models\load-engine-report.md
```

这条路径会先记录 engine 文件存在并进入 load-engine preflight。如果 engine 只有一个 float input、float outputs，且 shape 可以由 engine/profile 或 `--optShapes` 推断，工具会再执行 bounded enqueue/readback。identity engine 可以得到 `synthetic-input-runtime`；其他模型如果没有 reference output，会保留 `runtime-output-captured-unverified`，不能伪造成真实模型 proof。

## WinForms 入口

WinForms 界面提供以下字段：

- ONNX 路径。
- Engine 输出路径。
- TensorRT line：8、10 或 11。
- FP16、INT8、BF16、TF32 precision 选项。
- advanced timing：`--minTiming`、`--avgTiming`、`--infStreams`。
- precision policy：`--precisionConstraints`、`--layerPrecisions`、`--layerOutputTypes`。
- engine packaging/refit：`--versionCompatible`、`--excludeLeanRuntime`、`--stripWeights`、`--refit`、`--weightStreamingBudget`。
- timing cache export：`--exportTimingCache`。
- Workspace MiB。
- Min/Opt/Max shape profile。
- Report 输出路径。
- Build only 和 Skip inference 模式。

界面默认启用 build-only/skip-inference，避免用户在没有模型绑定语义时误把任意外部 ONNX 当作已经完成推理验证。UI 运行后输出的日志和 CLI 来自同一个 `TensorRtExecService`。

其中 advanced timing、precision policy、engine packaging/refit、weight-streaming 和 timing cache export 当前是 parse/report-only 能力。它们会进入命令预览、报告和 diagnostics，但不会被写成已经真实应用的 TensorRT 行为；报告会用 `TrtexecAlignmentStatus=parse-only` 明确标注边界。

## 报告语义

`--exportReport` 支持 `.json` 和 `.md`。JSON 报告包含 `ProofClassification`、`EvidenceClassifications`、`BuildEvidenceOnly`、`IsRuntimeExecutionProof`、`IsRealModelRuntimeProof`、`IsPackageConsumerRuntimeProof`、`StdoutSummary`、`StderrSummary` 和 `ModelEvidence` 字段。`IsRuntimeExecutionProof` 只有在真实执行并且输出匹配时才为 `true`；外部 ONNX build-only、skip-inference、load-engine preflight 和 `runtime-output-captured-unverified` 都不是可发布 runtime proof。

典型 build-only 报告片段如下：

```json
{
  "ProofClassification": "build-only",
  "EvidenceClassifications": [
    "build-only",
    "dependency-probe-only",
    "synthetic-input-runtime",
    "real-model-runtime",
    "package-consumer-runtime"
  ],
  "BuildEvidenceOnly": true,
  "IsRuntimeExecutionProof": false,
  "IsRealModelRuntimeProof": false,
  "IsPackageConsumerRuntimeProof": false,
  "StdoutSummary": "OnnxToEngine BuildOnly=True",
  "StderrSummary": "",
  "ModelEvidence": {
    "ModelSource": ".\\models\\model.onnx",
    "ModelSha256": "",
    "ModelLicense": "",
    "InputAssetName": "",
    "InputAssetSha256": ""
  }
}
```

`--evidenceSidecar` 是可选参数，用来读取一份 JSON sidecar，把模型 SHA256、许可证、输入资产 SHA256、stdout/stderr summary 等信息写进报告。它不能把 build-only 报告提升成 `package-consumer-runtime`；真实模型样例晋级仍要由 Classification / YoloVision 等 sample runner 和 asset manifest 共同证明。

Markdown 报告会明确写入：

```text
This report is build/sample evidence only. build-only and dependency-probe-only are not runtime proof; synthetic-input-runtime is not real model proof; package-consumer-runtime is tracked by release proof records.
```

这句边界很重要。它允许项目沉淀构建证据，同时避免把工具可用、样例可编译、报告可生成混同为公开发布完成。

## 参数边界

`--plugins` 和 `--timingCacheFile` 已被应用层解析并写入诊断记录，但当前安全阶段不会加载 plugin library，也不会导入或导出 timing cache。原因是 plugin registry 的 register/deregister/load library 仍属于生命周期和 ABI 风险更高的边界；下一阶段应先补安全设计和 smoke，再决定是否开放。

`--int8` 可以被解析并记录，但 calibrator 与 calibration cache 尚未在这个工具阶段实现。用户需要把 INT8 视为构建参数探索，而不是完整量化工作流。

## 与 OnnxToEngine 样例的关系

`samples/OnnxToEngine` 仍然是最小可验证样例，负责证明 parser、optimization profile、serialized engine 和 identity round-trip 可以在兼容环境下跑通。`applications/TensorRtExec` 直接引用 `src/JYPPX.TensorRtSharp.Tools`，不再依赖 sample executable project，因此应用发布边界更清楚。

## 排障建议

如果构建失败，先检查以下项目：

- `JYPPX_ENABLE_DEVELOPMENT_PROBING=1` 是否启用。
- TensorRT/CUDA/cuDNN runtime DLL 是否在预期探测路径。
- `--tensor-rt-line` 是否与本机 bridge/runtime 匹配。
- 动态模型是否提供完整 min/opt/max shape profile。
- 外部 ONNX 是否需要 TensorRT 不支持的算子或自定义 plugin。
- 报告中 `Skipped`、`SkipReason`、`Diagnostics` 和 `LogLines` 的内容。

当错误来自驱动或 runtime 兼容性时，应记录为环境阻塞，例如 `blocked-by-cuda-driver`，不要改写成 API 未完成或发布 proof。
