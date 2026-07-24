# OnnxToEngine 与 trtexec parity：把模型转换做成可审计流程

`samples/OnnxToEngine` 的目标不是做一个最小 demo，而是把 ONNX 到 TensorRT engine 的转换流程做清楚。对于熟悉 NVIDIA 官方 `trtexec` 的用户来说，它应该尽量贴近模型转换、shape profile、精度、timing cache、engine packaging、profiling 和 runtime artifact 的常用工作流；对于 .NET 用户来说，它又要比直接调用外部命令更容易集成、记录和排障。

但 parity 文章不能写成“已经完全复刻官方 trtexec”。当前项目用 `applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json` 和 `applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json` 把能力分层：implemented、implemented-report、implemented-build-readback、implemented-bounded-runtime、parse-report-only、diagnostic-alias-compatible、checklist-backed-command-preview。任何 report、dry-run、GUI screenshot、sidecar 或 matrix 都不是 package-consumer-runtime proof。

## 适合谁阅读

- 正在把 ONNX 模型转换为 TensorRT engine 的 .NET 开发者。
- 熟悉官方 `trtexec`，希望理解 TensorRtExec / OnnxToEngine 差距的用户。
- 需要审查 build-only、bounded-runtime-output、real-model-runtime、package-consumer-runtime proof 边界的维护者。
- 想把 TensorRtExec CLI、WinForms、YoloVision 和 owner proof input 串成可发布教程的文章作者。

## 两条路径的关系

项目中有两条相关路径：

- `samples/OnnxToEngine`：偏教程和样例，适合新用户理解模型转换。
- `applications/TensorRtExec`：偏正式工具，目标是 CLI + WinForms 复刻官方 `trtexec` 的主要能力。

两者共享 `TrtexecLikeParser`、`TrtexecLikeOptions`、`OnnxEngineBuildOptions.FromTrtexecLikeOptions` 和 `OnnxEngineBuildService`。这意味着 CLI、WinForms 和样例入口应该生成同一类 normalized command、report schema 和 proof boundary。

关键代码路径：

```text
samples/OnnxToEngine/Program.cs
applications/TensorRtExec/Core/TensorRtExecOptions.cs
applications/TensorRtExec/Core/TensorRtExecService.cs
applications/TensorRtExec/Core/TensorRtExecReport.cs
applications/TensorRtExec/Console/TensorRtExecCommand.cs
applications/TensorRtExec/WinForms/MainForm.cs
src/JYPPX.TensorRtSharp.Tools/TrtexecLikeParser.cs
src/JYPPX.TensorRtSharp.Tools/TrtexecLikeOptions.cs
src/JYPPX.TensorRtSharp.Tools/OnnxEngineBuildService.cs
src/JYPPX.TensorRtSharp.Tools/OnnxEngineBuildDiagnostics.cs
```

## 常见转换命令

OnnxToEngine 教程入口：

```powershell
dotnet run --project .\samples\OnnxToEngine\OnnxToEngine.csproj -- `
  --onnx E:\TensorRtSharpAssets\models\model.onnx `
  --saveEngine E:\TensorRtSharpAssets\engines\model.plan `
  --minShapes images:1x3x640x640 `
  --optShapes images:1x3x640x640 `
  --maxShapes images:4x3x640x640 `
  --fp16 `
  --buildOnly `
  --exportReport E:\TensorRtSharpAssets\reports\model-build-report.json
```

TensorRtExec 的 trtexec-like 形态：

```powershell
dotnet run --project .\applications\TensorRtExec\TensorRtExec.csproj -- `
  --onnx E:\TensorRtSharpAssets\models\model.onnx `
  --saveEngine E:\TensorRtSharpAssets\engines\model.plan `
  --minShapes images:1x3x640x640 `
  --optShapes images:1x3x640x640 `
  --maxShapes images:4x3x640x640 `
  --fp16 `
  --workspace 1024 `
  --buildOnly `
  --dumpLayerInfo `
  --exportLayerInfo E:\TensorRtSharpAssets\reports\model.layers.json `
  --exportReport E:\TensorRtSharpAssets\reports\model-tensorrtexec-report.json
```

WinForms 入口：

```powershell
dotnet run --project .\applications\TensorRtExec\TensorRtExec.csproj -- --ui
```

GUI 的 command preview 来自 `TensorRtExecOptions.ToArgumentLine()`，它只是 surface parity evidence。GUI 截图、command preview 和导出的 report 都不能晋级 runtime proof。

## Parity 矩阵怎么读

`tensor-rt-exec-trtexec-parity-matrix.json` 的核心字段包括：

```text
matrixId
application
modes = CLI / WinForms
matrixState = release-readiness-planning
proofBoundary
entries[].id
entries[].trtexecOption
entries[].tensorRtExecStatus
entries[].entryPoints
entries[].proofBoundary
entries[].isRuntimeProof
entries[].gap
entries[].nextStep
```

`tensor-rt-exec-release-candidate-gap-list.json` 则把 parity matrix 压成发布候选工作清单：

```text
gapListId
sourceMatrix
summary.totalItems
summary.implementedOrReportReady
summary.partialOrDiagnostic
summary.runtimeProofItems = 0
summary.packageConsumerRuntimeProofItems = 0
items[].currentStatus
items[].cliSupported
items[].winFormsSupported
items[].nextImplementationPaths
```

这些矩阵是 planning evidence only。它们能告诉维护者下一步做什么，不能授权发布，也不能关闭 release issue。GUI 截图、command preview、parity matrix 和 gap list 都不能证明 runtime output，也不能替代 package-consumer-runtime proof。

## 已覆盖的高价值能力

当前公开文章可以说这些能力已经有明确 surface 或 report/readback 路径：

```text
--onnx / --model / --onnxFile
--saveEngine / --save-engine / --engine / --plan / --engineFile
--loadEngine
--minShapes / --optShapes / --maxShapes
--shapes / --inputShapes / --batch
--fp16
--workspace / --memPoolSize
--avgTiming / --minTiming
--device / --useDLACore / --allowGPUFallback
--tacticSources
--directIO / --sparsity / --stronglyTyped
--inputIOFormats / --outputIOFormats
--precisionConstraints / --layerPrecisions / --layerOutputTypes
--versionCompatible / --excludeLeanRuntime / --stripWeights / --refit
--refitFromOnnx / --saveRefittedEngine
--allowWeightStreaming / --weightStreamingBudget
--timingCacheFile / --exportTimingCache
--profilingVerbosity / --dumpProfile / --exportProfile / --saveProfile
--dumpLayerInfo / --exportLayerInfo
--iterations / --warmUp / --duration / --streams
--useCudaGraph / --noDataTransfers
--loadInputs / --dumpOutput / --dumpRawBindingsToFile / --exportOutput / --exportTimes
```

其中很多能力是 builder/readback/report evidence，不是 runtime proof。例如 workspace/memory pool、timing iterations、deployment policies、IO/layer precision policies 和 timing cache 都能进入 report，但没有真实输入、输出校验、日志 hash 和 owner review 时，不能证明模型正确。

## Conversion Playbook 与 Parity 晋级标准

trtexec parity 不是“参数名字能解析”就结束。建议每个新增或提升的选项都走同一条 playbook：

| 步骤 | 关键代码/证据 | 可以说明什么 | 不能说明什么 |
| --- | --- | --- | --- |
| parse-normalized | TrtexecLikeParser.Parse、TrtexecLikeOptions.ToArgumentLine、NormalizedCommandLine、NormalizedCommandSha256 | 命令行可重复、alias 被归一化、preview/dry-run 可审计 | 不能说明 TensorRT 已应用该选项 |
| profile-normalized | EngineBuildProfile.Parse、EngineBuildShape、shapes/inputShapes fallback、min/opt/max triplet | 动态 shape 和 batch 边界已结构化 | 不能说明 profile 被 builder 接受 |
| builder-applied | TrtexecLikeDeploymentOptions、TrtexecLikeBuildPolicy、OnnxEngineBuildOptions.FromTrtexecLikeOptions | 选项进入 typed wrapper 或 version-guarded branch | 不能说明 readback 匹配 |
| builder-readback | BuilderConfigDeploymentSnapshot、OptionImplementationStatus、ReadbackMatch、VersionGuard = TRT8/TRT10/TRT11 | 已设置项被读回，跨版本差异可解释 | 不能说明 inference 正确 |
| artifact-written | saveEngine、exportTimingCache、exportLayerInfo、exportTimes、exportProfile、ArtifactSha256 | 文件写出、路径/hash 可复核 | 不能替代 runtime output 或 package source proof |
| bounded-runtime-output | loadEngine、loadInputs、dumpOutput、exportOutput、InferenceRan、OutputValidationPerformed | 可以证明 bounded enqueue/readback 曾发生 | 没有 reference output 时仍不是 real-model-runtime proof |
| real-model-runtime candidate | --mnist、--mnistInput、--expectedDigit、--exportPreprocessedInput、MnistOnnxRuntime、MnistOnnxRuntimeResult、OutputMatch | 模型特定输入、预处理、输出和期望值开始闭环 | 仍需要 owner review、日志 hash 和样例证据 validator |
| package-consumer-runtime remains external | clean external consumer、public package source、post-publish verification | 发布证明必须来自仓库外公开包消费 | OnnxToEngine 本仓库样例不能自己证明包发布可用 |

新增 parity 项时，文章和 matrix 至少要写清：AcceptedAlias、ParsedOnlyReason、AppliedByTypedWrapper、VersionGuard、ReadbackMatch、ArtifactWritten、ArtifactSha256、RuntimeExecuted、OutputValidationPerformed、OwnerReviewed 和 ProofClassification。缺少任一 proof-critical 字段时，只能停在 report/readback/candidate lane，不能升级为 package-consumer-runtime。

高风险项要更慢：--int8/--calib 至少需要 calibrator-owner-evidence-required，plugin 相关选项至少需要 plugin-lifecycle-owner-evidence-required；任何 borrowed pointer、外部资源或跨语言 ownership 都应保持 borrowed-pointer-disallowed，直到有 owner-safe wrapper、lifetime smoke 和跨版本质量门。

## 需要谨慎呈现的能力

这些能力特别容易被误写成“已完全支持官方 trtexec”，文章必须保守：

- `--int8 --calib`：当前是 parse-report-only-calibration-boundary；calibrator ownership、校准数据来源、cache hash 和 INT8 精度仍需要独立设计与 owner evidence。
- `--fp8 --best --dumpRefit --markDebug --dumpDebugTensors`：属于 parse-report-only 或 capability probe，不证明 precision support、debug tensor runtime output 或 refit 生命周期。
- `--plugins/--plugin/--dynamicPlugins/--setPluginsToSerialize`：当前只记录 plugin path 和 copied inventory metadata；register/deregister/load library 仍是 ownership 风险边界。
- `--sleepTime`：保持 parse-only，因为仓库没有忠实的 device-side launch-gap primitive。
- `--loadEngine`：可以记录 `PreflightMetadata`、`LoadedEngineDiagnostics`，并在兼容 one-float-input 场景下做 bounded enqueue/readback；没有 reference output 时只是 `runtime-output-captured-unverified`。
- WinForms：`tensor-rt-exec-gui-cli-field-map.json` 和 `tensor-rt-exec-gui-cli-field-map.md` 证明 GUI/CLI field parity，不证明 runtime output。

## Report 与 proof 边界

TensorRtExec report 会输出：

```text
ProofClassification
BuildEvidenceOnly
DryRun
NormalizedCommandLine
NormalizedCommandSha256
DeploymentOptions
BuilderConfigDeploymentSnapshot
ParserPreflightSnapshot
RuntimeOptions
InferenceRan
OutputMatch
IsRuntimeExecutionProof
IsRealModelRuntimeProof
IsPackageConsumerRuntimeProof
PreflightMetadata
LoadedEngineDiagnostics
CapabilityProbe
WorkspaceBytes
OptionImplementationStatus
ReportBoundary
```

这些字段能让文章、排障和 release candidate review 有证据可查。`NormalizedCommandSha256` 可以证明命令线未被替换；`OptionImplementationStatus` 可以说明哪些选项 applied、parse-only 或 blocked；`ReportBoundary.ForbiddenSubstitutes` 会列出不能冒充 proof 的项目。

禁止把以下内容写成 runtime/package proof：

- TensorRtExec report。
- OnnxToEngine report。
- build-only output。
- dry-run / previewOnly。
- dependency-probe-only。
- GUI screenshot。
- command preview。
- parity matrix。
- gap list。
- local feed package consumer。
- ProjectReference consumer。
- direct `.nupkg` install。
- GitHub Actions dry-run。

## 与 YoloVision 的关系

TensorRtExec/OnnxToEngine 负责 build/report 和通用 bounded runtime，YoloVision 负责 model-specific sample run。对于 YOLOv5、YOLOv6、YOLOv7、YOLOv8、YOLOv9、YOLOv10、YOLO11、YOLO26 以及 det/cls/seg/obb/pose/sem，真正能晋级 real-model-runtime 的证据必须来自：

```text
samples/YoloVision/yolovision-task-output-contract.json
samples/assets/yolovision-real-asset-owner-backfill-pack.json
samples/assets/yolovision-article-case-pack.json
eng/Test-YoloVisionRealAssetCandidate.ps1
eng/Test-YoloVisionRealAssetOwnerBackfillPack.ps1
eng/Test-SampleRunEvidenceRecord.ps1
```

推荐路径是：TensorRtExec build-only report -> engine/hash/readback sidecar -> YoloVision real run log -> output JSON/hash -> sample-run evidence validator -> owner review -> clean external consumer package proof -> post-publish verification。

## 配图建议

- 官方 trtexec 参数到 TensorRtExec 参数的对照表。
- ONNX -> engine -> TensorRtExec report -> YoloVision sample run -> proof validator 的流程图。
- parity matrix 截图，展示 implemented、implemented-report、parse-report-only、bounded-runtime-output 的状态颜色。
- build report JSON 示例截图，突出 `ProofClassification`、`BuildEvidenceOnly` 和 `IsPackageConsumerRuntimeProof=false`。

## 下一步

下一阶段应继续把 matrix 中 parse/report-only 或 diagnostic-alias-compatible 的项目变成更强的 typed readback、owner-safe lifecycle 或 model-specific evidence。优先级仍然是只读、查询型、部署关键型 API；calibrator、plugin load/register、callback、allocator、borrowed pointer 和 external resource 不要伪装成低风险实现。
