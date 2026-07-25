# TensorRtExec CLI：用 C# 复刻 trtexec 模型转换体验

TensorRT 用户绕不开 `trtexec`。它是官方最常用的模型转换、engine 构建、profile 配置、timing cache、layer info 和快速诊断工具。问题是，当主项目是 C#/.NET 时，官方 `trtexec` 往往只是外部命令：参数、日志、报告、错误处理、GUI 操作和应用内工作流都要再包一层。

TensorRtSharp4.0 中的 `applications/TensorRtExec` 目标就是做一个 C# 版 trtexec-like 应用：既能在控制台里以命令行方式使用，也能用 WinForms 打开页面；既服务 `samples/OnnxToEngine` 的模型转换，也服务 `samples/YoloVision` 的真实模型文章案例和 release evidence ladder。

这篇文章面向公众号、博客和项目主页读者。它可以介绍 TensorRtExec 的 CLI/WinForms 使用方式，但必须保持 proof boundary：TensorRtExec report、sidecar、GUI screenshot、command preview、dry-run、build-only 和 parity matrix 都不是 package-consumer-runtime proof。

## 适合谁阅读

- 已经熟悉官方 `trtexec`，希望在 .NET 工具链中复用类似能力的用户。
- 需要把 ONNX -> TensorRT engine 构建流程做进内部平台、桌面软件或 CI 的团队。
- 想用 WinForms 页面选择 ONNX、engine、shape profile、精度、timing cache 和 report 的 Windows 用户。
- 想理解 TensorRtSharp4.0 如何区分 build report、bounded runtime output、real-model-runtime 和 package-consumer-runtime proof 的维护者。

## 应用入口

TensorRtExec 的关键路径：

```text
applications/TensorRtExec/README.md
applications/TensorRtExec/TensorRtExec.csproj
applications/TensorRtExec/Console/TensorRtExecCommand.cs
applications/TensorRtExec/Core/TensorRtExecOptions.cs
applications/TensorRtExec/Core/TensorRtExecService.cs
applications/TensorRtExec/Core/TensorRtExecReport.cs
applications/TensorRtExec/WinForms/MainForm.cs
applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json
applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.md
applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json
applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.md
applications/TensorRtExec/tensor-rt-exec-gui-cli-field-map.json
applications/TensorRtExec/tensor-rt-exec-gui-cli-field-map.md
applications/TensorRtExec/tensor-rt-exec-report.schema.json
```

共享工具层：

```text
src/JYPPX.TensorRtSharp.Tools/TrtexecLikeParser.cs
src/JYPPX.TensorRtSharp.Tools/TrtexecLikeOptions.cs
src/JYPPX.TensorRtSharp.Tools/OnnxEngineBuildOptions.cs
src/JYPPX.TensorRtSharp.Tools/OnnxEngineBuildService.cs
src/JYPPX.TensorRtSharp.Tools/OnnxEngineBuildDiagnostics.cs
src/JYPPX.TensorRtSharp.Tools/OnnxEngineRuntimeArtifactWriter.cs
```

`TensorRtExecCommand` 负责 CLI parse、执行和 console summary；`TensorRtExecOptions` 负责把 trtexec-like 参数归一化为应用配置；`TensorRtExecService` 调用共享 build/report 服务；`TensorRtExecReport` 是面向 CLI/WinForms 的报告摘要；`MainForm` 使用同一套 options/service，避免 GUI 和 CLI 语义漂移。

## CLI 的定位

TensorRtExec 不是简单 demo runner。它承担三件事：

1. 把常用 `trtexec` 参数映射到 C# CLI 和 WinForms。
2. 输出机器可读 report、runtime/output artifact 和 evidence sidecar，便于后续审计。
3. 为 OnnxToEngine、YoloVision、文章案例和 release proof record 提供统一的构建/诊断后端。

parity matrix 的意义是说明“哪些官方 trtexec 能力已经映射，哪些只是 parse/report-only，哪些需要 owner proof”。它不是 runtime proof，也不是 release close approval。

## 常见命令

最小 ONNX build-only 转换：

```powershell
dotnet run --project .\applications\TensorRtExec\TensorRtExec.csproj -- `
  --onnx E:\TensorRtSharpAssets\models\model.onnx `
  --saveEngine E:\TensorRtSharpAssets\engines\model.plan `
  --buildOnly `
  --exportReport E:\TensorRtSharpAssets\reports\model-build-report.json
```

动态 shape、FP16、workspace、memory pool、timing cache 和 layer info：

```powershell
dotnet run --project .\applications\TensorRtExec\TensorRtExec.csproj -- `
  --onnx E:\TensorRtSharpAssets\models\yolov8-det.onnx `
  --save-engine E:\TensorRtSharpAssets\engines\yolov8-det.plan `
  --minShapes images:1x3x640x640 `
  --optShapes images:1x3x640x640 `
  --maxShapes images:4x3x640x640 `
  --fp16 `
  --workspace 1GiB `
  --memPoolSize workspace:512MiB,tacticDram:1GiB `
  --timingCacheFile E:\TensorRtSharpAssets\cache\yolov8-det.cache `
  --exportTimingCache E:\TensorRtSharpAssets\cache\yolov8-det-export.cache `
  --profilingVerbosity detailed `
  --dumpLayerInfo `
  --exportLayerInfo E:\TensorRtSharpAssets\reports\yolov8-det-layer-info.txt `
  --buildOnly `
  --exportReport E:\TensorRtSharpAssets\reports\yolov8-det-build-report.json
```

dry-run / previewOnly 只做参数解析和命令归一化，不读取 ONNX、不构建 engine、不探测 TensorRT runtime：

```powershell
dotnet run --project .\applications\TensorRtExec\TensorRtExec.csproj -- `
  --onnx E:\TensorRtSharpAssets\models\model.onnx `
  --saveEngine E:\TensorRtSharpAssets\engines\model.plan `
  --minShapes images:1x3x640x640 `
  --optShapes images:1x3x640x640 `
  --maxShapes images:4x3x640x640 `
  --dryRun `
  --exportReport E:\TensorRtSharpAssets\reports\model-precheck-report.md
```

加载已有 engine 做 readonly diagnostics 和 bounded runtime output：

```powershell
dotnet run --project .\applications\TensorRtExec\TensorRtExec.csproj -- `
  --loadEngine E:\TensorRtSharpAssets\engines\identity.plan `
  --optShapes input:1x1x1x1 `
  --iterations 10 `
  --warmUp 50 `
  --duration 1 `
  --streams 1 `
  --useCudaGraph `
  --exportTimes E:\TensorRtSharpAssets\reports\identity-times.json `
  --exportOutput E:\TensorRtSharpAssets\reports\identity-output.json `
  --dumpRawBindingsToFile E:\TensorRtSharpAssets\reports\identity-bindings `
  --exportReport E:\TensorRtSharpAssets\reports\identity-load-report.json
```

启动 WinForms：

```powershell
dotnet run --project .\applications\TensorRtExec\TensorRtExec.csproj -- --ui
```

这些命令适合做模型转换教程、开发调试、文章案例和内部验证。若要作为 package-consumer-runtime proof，还需要仓库外 clean consumer 使用公开包执行，不能用本仓库 ProjectReference、local feed 或 direct `.nupkg` 替代。

## trtexec-like 参数覆盖

常用模型与 engine 参数：

```text
--onnx / --model / --onnxFile
--saveEngine / --save-engine / --engine / --plan / --engineFile
--loadEngine / --load-engine
--buildOnly
--dryRun / --previewOnly
--exportReport / --report
--evidenceSidecar
```

shape profile：

```text
--minShapes
--optShapes
--maxShapes
--shapes
--inputShapes
--batch
```

precision、deployment 和 builder config：

```text
--fp16
--bf16
--noTF32
--int8
--calib
--fp8
--best
--workspace
--memPoolSize
--avgTiming
--minTiming
--device
--useDLACore
--allowGPUFallback
--tacticSources
--directIO
--sparsity
--stronglyTyped
--inputIOFormats
--outputIOFormats
--precisionConstraints
--layerPrecisions
--layerOutputTypes
```

engine packaging、refit 和 weight streaming：

```text
--versionCompatible
--excludeLeanRuntime
--stripWeights
--refit
--refitFromOnnx
--saveRefittedEngine
--allowWeightStreaming
--weightStreamingBudget
```

runtime timing 与 output artifacts：

```text
--iterations
--warmUp
--duration
--streams
--infStreams
--avgRuns
--percentile
--threads
--useSpinWait
--useCudaGraph
--noDataTransfers
--loadInputs
--dumpOutput
--dumpRawBindingsToFile
--exportOutput
--exportTimes
--sleepTime
--idleTime
```

diagnostics 与 profile：

```text
--timingCacheFile
--timingCache
--exportTimingCache
--profilingVerbosity
--verbose
--dumpProfile
--separateProfileRun
--exportProfile
--saveProfile
--dumpLayerInfo
--exportLayerInfo
--plugins / --plugin / --dynamicPlugins / --setPluginsToSerialize
--safe
--consistency
--builderCache
--noBuilderCache
```

`OptionImplementationStatus` 会把这些参数拆成 `ParsedOptions`、`AppliedOptions` 和 `ParseOnlyOptions`。能 parse 不等于真实 TensorRT 行为已经执行；真实 applied 需要 native call、readback、version guard 和 report 一致。

## 对齐状态怎么看

`applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json` 使用以下状态表达当前边界：

```text
implemented
implemented-report
implemented-build-readback
implemented-bounded-runtime
parse-report-only
diagnostic-alias-compatible
checklist-backed-command-preview
```

`tensor-rt-exec-release-candidate-gap-list.json` 会把 matrix 转成发布候选 gap list，并保留：

```text
matrixState = release-readiness-planning
runtimeProofItems = 0
packageConsumerRuntimeProofItems = 0
```

这两个文件是 planning/release-readiness evidence only。它们不能证明 runtime output，不能授权发布，也不能关闭 release issue。

## 报告字段

`--exportReport` 输出 JSON 或 Markdown，JSON schema 位于 `applications/TensorRtExec/tensor-rt-exec-report.schema.json`。核心字段包括：

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
ReportBoundary.ForbiddenSubstitutes
ReportBoundary.CopiedDiagnosticsBoundary
```

`NormalizedCommandSha256` 用来证明命令线未被替换；`BuilderConfigDeploymentSnapshot` 记录 pointer-free copied readback；`ParserPreflightSnapshot` 记录 parser diagnostics 和 model support；`LoadedEngineDiagnostics` 记录 engine readonly metadata；`CapabilityProbe` 记录 host/tool capability visibility；`ReportBoundary.ForbiddenSubstitutes` 明确列出哪些材料不能冒充 proof。

验证 report：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-TensorRtExecReport.ps1 `
  -InputPath E:\TensorRtSharpAssets\reports\model-build-report.json `
  -OutputPath E:\TensorRtSharpAssets\reports\model-build-report-validation.json `
  -Strict
```

脚本路径：`eng/Test-TensorRtExecReport.ps1`。

validator 通过只能说明 report 可被机器审查，不能把 TensorRtExec report、OnnxToEngine report、ONNX Parser diagnostic snapshot、ONNX ParserRefitter diagnostic snapshot 或 readonly diagnostics 提升成 `real-model-runtime` 或 `package-consumer-runtime` proof。

## Runtime/output artifact 边界

`--exportTimes`、`--exportOutput`、`--exportProfile`、`--saveProfile` 和 `--dumpRawBindingsToFile` 会输出 runtime/output artifacts。JSON artifact 会写入：

```text
ArtifactProofBoundary
RuntimeProofClass
HasTensorOutputProof
HasRawBindingProof
IsBuildOnlyEvidence
IsDependencyProbeOnly
IsSyntheticRuntime
ModelSource
EnginePath
PreflightMetadata
```

当 runtime artifact 存在时，工具还会写相邻 `*.engine-readback.json`，记录：

```text
LoadedEngineDiagnostics
ReadbackFingerprint
ReadbackSha256
ReadbackAvailable
skipped reason
```

`runtime-output-captured-unverified` 只代表 bounded enqueue/readback 已完成但没有 reference output。`synthetic-input-runtime is not real-model-runtime`，也不是 `package-consumer-runtime`。

## WinForms 与 CLI 一致性

WinForms 页面应围绕真实使用流设计，而不是只做参数堆叠：

- 左侧选择 ONNX、engine 输出路径、TensorRT line、精度和 profile。
- 中间提供参数分组：构建、shape、precision、deployment、timing cache、runtime timing、layer/profile diagnostics、output artifacts。
- 右侧显示 command preview、执行日志、report 摘要、error diagnostics 和 `NormalizedCommandSha256`。
- 底部保留可复制命令，让 GUI 操作可以回到 CLI 自动化。

GUI/CLI field map 由：

```text
eng/Export-TensorRtExecGuiCliParityChecklist.ps1
eng/Test-TensorRtExecGuiCliParityChecklist.ps1
applications/TensorRtExec/tensor-rt-exec-gui-cli-field-map.json
applications/TensorRtExec/tensor-rt-exec-gui-cli-field-map.md
```

维护。它记录 CLI token、WinForms 字段、command preview、状态和下一步，但只是 surface parity evidence；GUI screenshot、dry-run、build report、timing cache 路径、INT8 calibration cache 路径和 command preview 都不是 runtime proof，也不是 package-consumer-runtime proof。

## CLI/WinForms 操作闭环与 DLL 排障

一条可复用的 TensorRtExec 操作闭环应当有四个阶段：

1. **准备阶段**：在 E 盘 workspace 中确认 ONNX、输出目录、TensorRT line、CUDA/TensorRT DLL 搜索路径和 shape profile；先执行 --dryRun，保存 normalized command 和 precheck report。
2. **构建阶段**：执行 --buildOnly --exportReport --evidenceSidecar，确认 OptionImplementationStatus、BuilderConfigDeploymentSnapshot、EngineFileRoundTrip 和 engine SHA256；失败时保留 stdout/stderr，不要只截 GUI 红色提示。
3. **诊断/运行阶段**：对已有 engine 先做 readonly diagnostics；只有输入 binding、输出语义和 reference output 都明确时，才执行 bounded runtime 或模型特定 runner。
4. **归档阶段**：把 command、report、sidecar、engine/readback、日志和 host metadata 放入同一个 case workspace，并记录每个文件的 SHA256；这些材料仍然要经过 owner review 才能进入 real-case evidence。

CLI 和 WinForms 应保持同一条数据流：WinForms 控件先写入 TensorRtExecOptions，再调用 ToArgumentLine() 生成 command preview，CLI 与 GUI 最终都进入 TensorRtExecService，由同一份 TensorRtExecReport 输出状态。GUI 不能自己拼接另一套参数，也不能用截图替代 report、日志或 validator。

常见 DLL 问题可以按下面顺序排查：

| 症状 | 先检查 | 文章中的证据分类 |
| --- | --- | --- |
| 启动时找不到 jyppxtrt*.dll 或 jyppxcudabridge.dll | 应用输出目录、native bridge 架构、PATH、TensorRT/CUDA bin 目录 | loader/preflight diagnostic，不是 runtime proof |
| 能加载 bridge 但找不到 nvinfer.dll、nvinfer_plugin.dll 或 nvonnxparser.dll | TensorRT line、x64 架构、对应 bin 目录和版本匹配 | dependency-probe-only 或 blocked-by-cuda-driver |
| CUDA runtime 找不到 cudart64_*.dll | CUDA toolkit/runtime 版本、PATH 顺序、进程位数 | environment diagnostic |
| CLI 能运行但 WinForms 失败 | WinForms 输出目录是否复制同一 native assets、GUI 使用的 normalized command 是否一致 | surface parity issue |
| build report 成功但 engine runtime 失败 | engine 与 host TensorRT line、driver、binding shape、plugin 依赖和 readback metadata | build-only 不自动升级为 runtime proof |

Windows 下建议先用 where.exe 和进程实际工作目录确认 loader 看到的路径，再检查 TensorRtExecReport.PreflightMetadata、CapabilityProbe 和 LoadedEngineDiagnostics。不要通过把多个版本 DLL 混到 PATH 中“碰运气”；不同 TRT8/TRT10/TRT11 目录必须与对应 manifest、native bridge 和托管 version guard 保持一致。若必须临时复制 DLL 做隔离实验，应把复制目录放在 E 盘 case workspace，并在报告中记录 source path、版本和 SHA256。

一个 DLL 加载成功的报告只说明 loader 和依赖探测阶段完成。它不能证明 ONNX 输出正确；不能证明真实模型 runtime 成功；不能证明公开 NuGet 包可消费，也不能关闭 release issue。

## 与 OnnxToEngine / YoloVision 的关系

`samples/OnnxToEngine` 更像“面向样例读者的模型转换路径”，重点是清晰、易懂、适合教程；`applications/TensorRtExec` 更像正式工具，目标是覆盖官方 `trtexec` 的主要模型转换能力，并同时支持 CLI 和 WinForms。

`samples/YoloVision` 则负责 model-specific real run：YOLOv5、YOLOv6、YOLOv7、YOLOv8、YOLOv9、YOLOv10、YOLO11、YOLO26、YOLOX 和 custom，det/cls/seg/obb/pose/sem 六任务都需要真实模型、labels、输入资产、output JSON、log SHA256 和 owner review。

推荐证据路径：

```text
TensorRtExec build-only report
  -> engine path/hash/readback sidecar
  -> YoloVision real run log
  -> yolovision-output.v1 JSON/SVG
  -> sample-run-evidence record
  -> owner review
  -> clean external package consumer
  -> post-publish verification
```

`samples/assets/yolovision-real-asset-owner-backfill-pack.json` 和 `samples/YoloVision/yolovision-task-output-contract.json` 会把 TensorRtExec shape profile、task/output metadata 和 sample-run evidence 字段对齐。它们是 owner backfill scaffold，不是 package-consumer-runtime proof。

## Real case proof pack

TensorRtExec 在 `artifacts/final-release/real-case-proof-execution-pack.json` 中有两个 release-facing case：

```text
tensorrtexec-build-report
tensorrtexec-gui-workflow
```

相关脚本：

```text
eng/Export-RealCaseProofExecutionPack.ps1
eng/Export-RealCaseEvidenceRecordTemplate.ps1
eng/Test-RealCaseEvidenceRecord.ps1
```

`Test-RealCaseEvidenceRecord.ps1` 只有在 owner 回填模型来源、license、ONNX/engine/input/output SHA256、stdout/stderr log、截图、host OS、GPU、driver、CUDA、TensorRT、runtime package metadata 和 owner review 后才允许形成候选 real-case evidence。即便 CLI/GUI report、sidecar 和截图齐全，也必须保持 `canPublishPublicly=false`，不能替代 `real-model-runtime`、`package-consumer-runtime`、Linux runner proof、owner authorization 或 post-publish verification。

## Proof 边界

必须保持以下边界：

- `precheck` / `dryRun` 只证明参数可解析、命令可归一化，不读取模型，不构建 engine。
- `build-only` 只证明 ONNX parser/builder 走到构建报告边界，不证明推理输出正确。
- `dependency-probe-only` 只证明依赖探测或 load-engine preflight 结果，不是 runtime execution proof。
- `capability-probe-only` 只证明只读 API/host capability 可见性，不是 runtime execution proof、real-model-runtime proof 或 package-consumer-runtime proof。
- `runtime-output-captured-unverified` 只证明 bounded output 被捕获，不能证明模型正确。
- `synthetic-input-runtime` 可证明最小管线执行，不是用户真实模型质量证明。
- `real-model-runtime` 需要 Classification / YoloVision 等 sample runner 的真实模型、labels、输入资产、hash、许可证和 sample run evidence。
- `package-consumer-runtime` belongs to release proof records；TensorRtExec build report、sidecar、样例 manifest 都不能直接声明它。
- `sidecar-only` 是模型来源、hash、license、stdout/stderr 摘要和 handoff metadata，不是 runtime proof，也不是 release proof。
- `blocked-by-cuda-driver` 是环境兼容性阻塞，不是 smoke passed，也不是 API 缺口。

一句话边界：TensorRtExec 能把模型转换、参数归一化、builder/readback、diagnostics 和 bounded runtime output 变成可审计材料，但不能单独授权发布、不能关闭 release issue、不能证明 package-consumer-runtime。

## 本地质量检查

推荐本地检查：

```powershell
dotnet build .\TensorRtSharp.sln -c Debug --no-restore /p:UseSharedCompilation=false /nr:false

dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj `
  -c Debug --no-build `
  --filter "FullyQualifiedName~TensorRtExec|FullyQualifiedName~OnnxToEngine|FullyQualifiedName~YoloVision"
```

文章或 report boundary 改动后，至少跑：

```powershell
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj `
  -c Debug `
  --filter "FullyQualifiedName~PublishingPublicArticleTests"
```

## 配图建议

- CLI 命令到 `TensorRtExecReport` 的流程图。
- WinForms 参数页、command preview、report 摘要和错误诊断截图。
- trtexec parity matrix 状态表：implemented、implemented-report、implemented-build-readback、parse-report-only。
- Evidence ladder：TensorRtExec build report -> YoloVision sample run -> package-consumer-runtime -> post-publish verification。

## 下一步

TensorRtExec 的下一阶段应该继续补齐官方 `trtexec` conversion parity：更多 profile/precision/serialization 参数、错误诊断、report 字段和 GUI 参数同步。新增高级参数时，先进入 parser/report/`OptionImplementationStatus.ParseOnlyOptions`，再用真实 native readback 或 bounded runtime smoke 提升状态；不要把 callback、allocator、plugin lifecycle、borrowed pointer、external resource 或 runtime deserialization ownership 伪装成低风险实现，也不要把 TensorRtExec report 写成 package-consumer-runtime proof。
