# TensorRtExec 外部 ONNX 构建报告指南

`TensorRtExec` 已经从参数预检工具提升为复用 `JYPPX.TensorRtSharp.Tools` 的 build/report 工具。它可以把外部 ONNX 模型交给 TensorRT parser 和 builder，并输出 JSON 或 Markdown 报告。本文说明如何使用报告，以及哪些结论不能从报告中推出。

## 基本命令

先做 dry-run 预检时，可以使用 `--dryRun` 或 `--previewOnly`：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model.plan `
  --minShapes input:1x3x640x640 `
  --optShapes input:1x3x640x640 `
  --maxShapes input:1x3x640x640 `
  --builderOptimizationLevel 4 `
  --maxAuxStreams 2 `
  --exportReport .\models\model-precheck-report.json `
  --previewOnly
```

dry-run 允许模型路径暂时不存在，报告固定为 `State=dry-run-precheck`、`DryRun=true`、`ProofClassification=precheck` 和 `BuildEvidenceOnly=true`。它不会触发 CUDA/TensorRT runtime probe、ONNX parser、engine build、plugin loading 或 inference，只适合命令预检、参数归一化和交接记录。

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model.plan `
  --minShapes input:1x3x640x640 `
  --optShapes input:1x3x640x640 `
  --maxShapes input:1x3x640x640 `
  --fp16 `
  --workspace 512 `
  --buildOnly `
  --evidenceSidecar .\models\model-evidence.sidecar.json `
  --exportReport .\models\model-build-report.json
```

如果模型是静态 shape，可以省略 shape profile。动态 shape 模型必须补齐 min/opt/max，否则 TensorRT 可能在 build 阶段失败。

## 报告字段

JSON 报告包含这些关键字段：

- `Success`：工具流程是否成功返回。
- `Skipped`：是否因为环境或能力探测跳过。
- `State`：`identity-roundtrip`、`external-onnx-build-only`、`load-engine-preflight` 等状态。
- `TensorRtLine`：8、10 或 11。
- `ModelSource`：模型来源路径或内置 identity。
- `EnginePath`：engine 输出路径。
- `Parsed`：ONNX parser 是否成功。
- `EngineSaved`：serialized engine 是否写出。
- `DryRun`：是否只执行参数预检。为 true 时不会探测 runtime、解析 ONNX、构建 engine 或执行推理。
- `InferenceRan`：是否真的执行了推理。
- `OutputMatch`：输出是否与预期匹配。
- `IsRuntimeExecutionProof`：只有 `InferenceRan && OutputMatch` 才为 true。
- `NormalizedCommandSha256`：对归一化命令行计算的 SHA256，用于 issue、日志和发布证据对账。
- `ProofClassification`：当前报告的证据分类，dry-run 是 `precheck`，外部 ONNX 构建通常是 `build-only`，加载预检或环境跳过通常是 `dependency-probe-only`。
- `EvidenceClassifications`：项目当前识别的证据分类集合，包括 `build-only`、`dependency-probe-only`、`precheck`、`synthetic-input-runtime`、`real-model-runtime` 和 `package-consumer-runtime`。
- `BuildEvidenceOnly`：是否只能作为构建/样例证据。
- `IsRealModelRuntimeProof`：是否可作为真实模型 runtime proof。单独的 build report 通常为 false。
- `IsPackageConsumerRuntimeProof`：是否可作为 NuGet/runtime package consumer proof。TensorRtExec 报告不会声明该级别，它属于 release proof record。
- `OptionImplementationStatus`：把当前报告中的参数拆为 `ParsedOptions`、`AppliedOptions` 和 `ParseOnlyOptions`，用于说明哪些参数只是被 parser/report 接住，哪些已经在 build/runtime 服务中实际应用。
- `StdoutSummary` / `StderrSummary`：用于发布证据或问题单的短摘要。
- `ModelEvidence`：模型来源、模型 SHA256、许可证、输入资产名和输入资产 SHA256 的承载结构。build-only report 可以先留空，真实模型案例必须回填。

外部 ONNX 默认会进入 build-only 或 skip-inference 语义。原因是工具并不知道任意模型的输入 tensor 名称、输出 tensor 名称、输出 layout、后处理、labels 和真实图片。

典型 JSON 片段：

```json
{
  "ProofClassification": "build-only",
  "BuildEvidenceOnly": true,
  "DryRun": false,
  "NormalizedCommandSha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "IsRuntimeExecutionProof": false,
  "IsRealModelRuntimeProof": false,
  "IsPackageConsumerRuntimeProof": false,
  "OptionImplementationStatus": {
    "ParsedOptions": [
      "--tensor-rt-line",
      "--workspace",
      "--saveEngine",
      "--builderOptimizationLevel",
      "--minTiming",
      "--precisionConstraints"
    ],
    "AppliedOptions": [
      "--tensor-rt-line",
      "--workspace",
      "--builderOptimizationLevel",
      "--saveEngine"
    ],
    "ParseOnlyOptions": [
      "--minTiming",
      "--precisionConstraints"
    ],
    "EvidenceBoundary": "build reports distinguish parsed, applied, and parse-only options; parse-only/build-only evidence cannot promote real-model-runtime or package-consumer-runtime proof."
  },
  "StdoutSummary": "Parsed=True EngineSaved=True",
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

## 证据边界

可以从 build-only 报告得出的结论：

- TensorRT runtime/builder 在当前环境可创建，或报告记录了跳过原因。
- ONNX parser 至少尝试解析了指定模型。
- TensorRT builder 尝试生成 serialized engine。
- precision、workspace、shape profile 参数被记录。
- advanced timing、precision constraints、engine packaging/refit、weight-streaming 和 timing cache 参数会进入 `DeploymentOptions` / `RuntimeOptions` / `NormalizedCommandLine`；成功构建时 timing cache 导入/导出还会进入 `TimingCacheArtifact`，记录文件大小和 SHA256。
- `OptionImplementationStatus` 会把 `ParsedOptions`、`AppliedOptions`、`ParseOnlyOptions` 写进 JSON 和 Markdown，便于 review 时直接看出 parse-only 边界。
- `--inputIOFormats`、`--outputIOFormats` 与 TRT8/10 的 `--precisionConstraints`、`--layerPrecisions`、`--layerOutputTypes` 只有真实 set/readback match 才进入 `AppliedOptions`；TRT11 type 不匹配的 I/O 或已移除的 precision setters、dry-run、load-engine、依赖不可用路径继续进入 `ParseOnlyOptions`。TRT10/11 的 `--minTiming`、`--infStreams`、engine packaging/refit 与 weight streaming 仍按各自执行结果或版本 guard 分类。所有 builder readback 都只是 build evidence，不是模型正确性或 package-consumer proof。
- `ProofClassification=precheck`、`build-only` 或 `dependency-probe-only` 明确了这份证据不能直接提升为 runtime proof。

不能从 build-only 报告得出的结论：

- 模型检测或分类质量正确。
- 任意外部模型都能通用推理。
- NuGet 包已经公开发布。
- callback、allocator、debug listener runtime proof 已完成。
- dry-run 已经读取真实模型或验证 TensorRT runtime。
- plugin library load/register 已完成。
- `package-consumer-runtime` 已完成；它只由 release proof record 校验和提升。

## plugin 和 timing cache

`--plugins` 与 `--timingCacheFile` 当前会进入诊断记录：

```powershell
--plugins .\plugins\custom.dll --timingCacheFile .\models\model.cache
```

现阶段不会加载 plugin library。成功 TensorRT build 会导入/导出 timing cache，并在 `TimingCacheArtifact` 记录 lifecycle 状态、文件大小和 SHA256；dry-run、load-engine 和依赖不可用路径不会执行 cache 操作。plugin registry mutation、外部资源加载和 plugin instance 生命周期仍属于高风险边界，需要单独设计和 smoke 验证。

## Markdown 报告

把后缀改成 `.md` 即可生成 Markdown：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model.plan `
  --buildOnly `
  --exportReport .\models\model-build-report.md
```

Markdown 报告会明确写入 build/sample evidence 边界，适合附在 issue、发布候选记录或模型资产清单旁边。

## 推荐归档结构

```text
models/
  model.onnx
  model.plan
  model-build-report.json
  model.assets.json
  model-run.log
```

`model-build-report.json` 证明 build 尝试；`model.assets.json` 证明来源和许可证；`model-run.log` 证明样例实际运行。三者合在一起才接近可复现案例证据。

## Evidence sidecar

sidecar 是连接 build report 与真实模型资产记录的轻量 JSON：

```json
{
  "proofClassification": "build-only",
  "stdoutSummary": "Parsed=True EngineSaved=True",
  "stderrSummary": "",
  "modelEvidence": {
    "modelSource": ".\\models\\model.onnx",
    "modelSha256": "",
    "modelLicense": "",
    "inputAssetName": "",
    "inputAssetSha256": ""
  }
}
```

`TensorRtExec` 会把这些字段写入报告的 `ModelEvidence`、`StdoutSummary`、`StderrSummary` 和 `EvidenceSidecarDiagnostics`。即使 sidecar 写了 `real-model-runtime` 或 `package-consumer-runtime`，build report 也不会因此越级；真实模型 runtime 由 sample runner 证明，package consumer runtime 由 release proof record 证明。

## sidecar 模板与校验

发布前建议先生成模板：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OnnxEngineBuildEvidenceSidecarTemplate.ps1
```

该脚本会输出：

- `artifacts/user-acceptance/onnx-engine-build-evidence-sidecar.template.json`
- `artifacts/user-acceptance/onnx-engine-build-evidence-sidecar.classification.template.json`
- `artifacts/user-acceptance/onnx-engine-build-evidence-sidecar.yolovision.template.json`
- `artifacts/user-acceptance/onnx-engine-build-evidence-sidecar.yolox-s.template.json`

回填后运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OnnxEngineBuildEvidenceSidecar.ps1
```

默认模式会扫描 `samples/assets/*.template.json` 和 `samples/assets/*example.json` 中的 `evidence.evidenceSidecar`。模板 manifest 里缺少真实 sidecar 会标记为 `owner-action-required`，不是 error；因为仓库本身不下载模型，也不伪造 hash。传入 `-SidecarPath .\models\model-evidence.sidecar.json` 时可校验单个 sidecar，单个文件不存在才是错误。

校验器会检查 `proofClassification` 枚举、`modelSha256`、`modelLicense`、`inputAssetName`、`inputAssetSha256`、`stdoutSummary` 和 `stderrSummary`。`real-model-runtime` 只有在模型/input hash、许可证和输出摘要齐全时才可作为真实模型证据；`package-consumer-runtime` 在 sidecar 中只能作为诊断记录保留，校验结果必须保持 `canPromotePackageConsumerRuntime=false`，不能由 sidecar 或 build report 晋级。

## sample run evidence record

sidecar 只能说明 build report 旁边有哪些模型资产证据。真实样例是否跑过，需要另一个记录：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-SampleRunEvidenceRecordTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1
```

`sample-run-evidence-record.*.template.json` 会承载真实 runner 命令、`sampleRunLogPath`、`sampleRunLogSha256`、stdout/stderr 摘要、expected evidence lines、`isSmokePassed` 和 `canPromoteRealModelRuntime`。它和 sidecar、asset manifest 一起证明真实模型样例运行；它不替代 NuGet/runtime package consumer proof，也不允许声明 `package-consumer-runtime`。如果 sample record 写入 `proofClassification=package-consumer-runtime`，`Test-SampleRunEvidenceRecord.ps1` 必须输出 `validationState=invalid` 并返回非零退出码。

发布证据聚合时，`Export-ReleaseEvidenceBundle.ps1` 会把 sidecar audit、sample asset manifest audit、sample run evidence validation 和 user acceptance catalog 一起列出。只要 runner evidence 仍是 `owner-action-required` 或 `template-only`，release evidence bundle 就只会显示样例证据尚未晋级，不会把 build-only report 写成真实 Classification/YoloVision runtime。
