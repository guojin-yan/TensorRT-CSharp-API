# ONNX 到 Engine：把模型转换做成可审计流程

ONNX 到 TensorRT engine 的转换是 TensorRtSharp4.0 最容易被用户感知的能力。`samples/OnnxToEngine/Program.cs` 提供了面向样例和文章教程的转换入口，`applications/TensorRtExec` 则承担更完整的 trtexec-like 参数、CLI/WinForms 双入口、报告和 evidence sidecar。

这条链路要说清楚两件事：第一，ONNX parse、builder config、engine serialization、load-engine readonly diagnostics 和 report JSON 都可以做成可审计证据；第二，转换成功不是 package-consumer-runtime proof，也不等于模型语义正确。build-only、dry-run、dependency-probe-only、local feed、ProjectReference 和 direct `.nupkg` install 只能放在证据梯子的低层。

## 适合

- 想把 ONNX 模型转成 TensorRT engine 的 C# 用户。
- 需要比较 OnnxToEngine 与官方 `trtexec` 转换参数覆盖的人。
- 准备为 YoloVision 或自有模型生成真实资产 proof 的维护者。
- 想把 TensorRtExec report、OnnxToEngine report 和 owner proof input 串起来的发布负责人。

## 关键路径

- 样例入口：`samples/OnnxToEngine/Program.cs`。
- 共享参数解析：`src/JYPPX.TensorRtSharp.Tools/Trtexec/TrtexecLikeParser.cs`、`TrtexecLikeParser.Arguments.cs`、
  `TrtexecLikeParser.ScalarParsing.cs`、`TrtexecLikeParser.BuildOptionValues.cs`、`TrtexecLikeParser.MemoryUnits.cs`。
- 参数模型：`src/JYPPX.TensorRtSharp.Tools/Trtexec/TrtexecLikeOptions.cs`、`OnnxEngineBuildOptions.cs`。
- 构建服务：`src/JYPPX.TensorRtSharp.Tools/Build/OnnxEngineBuildService.cs`。
- 构建结果：`src/JYPPX.TensorRtSharp.Tools/Build/OnnxEngineBuildResult.cs`。
- JSON/Markdown 诊断投影：`src/JYPPX.TensorRtSharp.Tools/Build/OnnxEngineBuildDiagnostics.Json.cs`、`OnnxEngineBuildDiagnostics.Markdown.cs`。
- report writer：`src/JYPPX.TensorRtSharp.Tools/OnnxEngineBuildReportWriter.cs`。
- evidence sidecar：`src/JYPPX.TensorRtSharp.Tools/Artifacts/OnnxEngineBuildEvidenceSidecar.cs`。
- MNIST 模型 runtime：`src/JYPPX.TensorRtSharp.Tools/Runtime/MnistOnnxRuntimeService.cs`、
  `MnistOnnxRuntimeService.Tensors.cs`、`MnistOnnxRuntimeResult.cs`、`MnistOnnxRuntimeDiagnostics.cs`。
- TensorRtExec CLI/GUI：`applications/TensorRtExec/Console/TensorRtExecCommand.cs`、`applications/TensorRtExec/WinForms/MainForm.cs`。
- parity/gap 证据：`applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json`、`applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json`。
- YoloVision 案例包：`samples/assets/yolovision-article-case-pack.json`、`samples/assets/yolovision-family-task-real-asset-roadmap.json`。

## 基本流程

推荐把模型、engine、report 和 sidecar 都放在 E 盘固定 workspace，例如：

```powershell
$assetRoot = "..\downloads\onnx-to-engine"

dotnet run --project .\samples\OnnxToEngine\OnnxToEngine.csproj -- `
  --onnx "$assetRoot\models\model.onnx" `
  --saveEngine "$assetRoot\engines\model.plan" `
  --minShapes images:1x3x640x640 `
  --optShapes images:1x3x640x640 `
  --maxShapes images:4x3x640x640 `
  --fp16 `
  --workspace 1024 `
  --buildOnly `
  --exportReport "$assetRoot\reports\model-build-report.json" `
  --evidenceSidecar "$assetRoot\reports\model-evidence.sidecar.json"
```

同一套参数也能交给 TensorRtExec：

```powershell
dotnet run --project .\applications\TensorRtExec\TensorRtExec.csproj -- `
  --onnx "$assetRoot\models\model.onnx" `
  --saveEngine "$assetRoot\engines\model.plan" `
  --minShapes images:1x3x640x640 `
  --optShapes images:1x3x640x640 `
  --maxShapes images:4x3x640x640 `
  --fp16 `
  --buildOnly `
  --dumpLayerInfo `
  --exportLayerInfo "$assetRoot\reports\model.layers.json" `
  --exportReport "$assetRoot\reports\model-tensorrtexec-report.json"
```

这两个入口共享 `TrtexecLikeParser` 和 `OnnxEngineBuildService`，所以文章里不要把它们写成两套互相矛盾的转换语义。OnnxToEngine 更像教程入口；TensorRtExec 更像官方 `trtexec` 的扩展复刻入口，适合展示 CLI/GUI parity、report schema 和参数覆盖。

## trtexec-like 参数面

当前 parser 已覆盖或记录大量官方 `trtexec` 风格参数：

```text
--onnx / --model / --onnxFile
--saveEngine / --save-engine / --engine / --plan / --engineFile
--loadEngine / --load-engine
--minShapes / --optShapes / --maxShapes / --shapes / --inputShapes
--fp16 / --int8 / --bf16 / --noTF32
--workspace / --memPoolSize
--timingCache / --timingCacheFile / --exportTimingCache
--profilingVerbosity / --verbose
--dumpLayerInfo / --exportLayerInfo
--device / --useDLACore / --allowGPUFallback
--tacticSources
--inputIOFormats / --outputIOFormats / --directIO
--sparsity / --stronglyTyped
--precisionConstraints / --layerPrecisions / --layerOutputTypes
--versionCompatible / --excludeLeanRuntime
--stripWeights / --refit / --refitFromOnnx / --saveRefittedEngine
--allowWeightStreaming / --weightStreamingBudget
--iterations / --warmUp / --duration / --streams / --useCudaGraph
--loadInputs / --dumpOutput / --dumpRawBindingsToFile
--exportOutput / --exportTimes / --exportProfile / --saveProfile
--safe / --consistency / --builderCache / --noBuilderCache
--dryRun / --previewOnly / --buildOnly / --skipInference
--exportReport / --report / --evidenceSidecar
```

文章要把“implemented”和“parsed/diagnostic only”分开说。比如 `--avgTiming` 可以走 builder config readback，`--minTiming` 在 TRT8 有 legacy compatibility，TRT10/11 则保留 parse-only 诊断。`--int8` flag 可以被解析，但 calibrator 与 calibration cache 并不因此自动拥有真实 INT8 calibration proof。`--plugins` 也只记录 plugin library arguments，不在安全阶段加载 plugin library。

## report 能证明什么

`OnnxEngineBuildResult` 和 `OnnxEngineBuildDiagnostics` 会把转换链路写成机器可读字段：

```text
Success
Skipped
State
TensorRtLine
ModelSource
EnginePath
Parsed
EngineSaved
EngineFileRoundTrip
InferenceRan
OutputMatch
ProofClassification
BuildEvidenceOnly
IsRuntimeExecutionProof
IsRealModelRuntimeProof
IsPackageConsumerRuntimeProof
NormalizedCommandLine
NormalizedCommandSha256
WorkspaceBytes
PreflightMetadata
LoadedEngineDiagnostics
BuilderConfigDeploymentSnapshot
ParserPreflightSnapshot
TimingCacheArtifact
CapabilityProbe
EvidenceSidecar
```

这些字段适合进入文章截图、排障记录和候选矩阵。`Parsed=true`、`EngineSaved=true`、`EngineFileRoundTrip=true` 可以证明 parser/build/serialization 路径走通；`NormalizedCommandSha256` 可以证明命令没有被悄悄换掉；`LoadedEngineDiagnostics` 和 `.engine-readback.json` 可以证明 engine metadata 可读取。它们仍然不能单独证明模型输出正确。

## MNIST 路径和真实模型边界

`samples/OnnxToEngine/Program.cs` 还有一个 `--mnist` 路径，走 `MnistOnnxRuntimeService`：

```powershell
dotnet run --project .\samples\OnnxToEngine\OnnxToEngine.csproj -- `
  --mnist `
  --tensor-rt-line 10 `
  --onnx "$assetRoot\models\mnist.onnx" `
  --mnistInput "$assetRoot\inputs\7.pgm" `
  --expectedDigit 7 `
  --saveEngine "$assetRoot\engines\mnist.plan" `
  --exportReport "$assetRoot\reports\mnist-report.json" `
  --exportOutput "$assetRoot\reports\mnist-output.json" `
  --exportPreprocessedInput "$assetRoot\reports\mnist-input.bin"
```

这个路径会打印 `MnistOnnxRuntime ProofClassification`、`RealModelRuntime`、`PackageConsumerRuntime`、`Expected`、`Predicted`、`Confidence`、`OutputMatch` 和 `ProofBoundary`。只有当 owner 提供真实 ONNX、输入、expected label、输出日志、SHA256、host metadata，并且 validator 接受时，它才可能接近 real-model-runtime proof。它仍不是 package-consumer-runtime proof，因为它不证明公开包在 clean external consumer 中安装、restore、build 和 smoke 成功。

## 与 YoloVision 的关系

YoloVision 负责把真实模型资产、输入样例、输出 schema 和任务 metadata 组织起来。OnnxToEngine 负责转换，TensorRtExec 负责更接近 `trtexec` 的参数和报告，YoloVision 则把 detection、classification、segmentation、OBB、pose、semantic segmentation 等任务串成可复核案例。

公开文章应使用 `samples/YoloVision` 和 `samples/assets/yolovision-article-case-pack.json` 的口径，覆盖 YOLOv5、YOLOv6、YOLOv7、YOLOv8、YOLOv9、YOLOv10、YOLO11、YOLO26 等系列的候选路线，不得退回早期过窄的 detection-only 样例命名。模型获取、license、ONNX export、engine build、YoloVision run、output schema 和 SHA256 都要由 owner evidence 补齐。

## proof 边界

ONNX 转换成功是必要条件，但不是 package-consumer-runtime proof。build-only、dry-run、previewOnly、dependency-probe-only、template、local feed、ProjectReference 和 direct `.nupkg` install 不能证明公开包可被用户消费。真实 package proof 需要 owner 提供外部 clean consumer 或公开包消费日志、SHA256、host metadata、exitCode=0、passed=true 和严格 validator 输出。

禁止把以下内容写成 runtime proof：

- TensorRtExec build-only report。
- OnnxToEngine report。
- `ParserPreflightSnapshot`。
- `BuilderConfigDeploymentSnapshot`。
- `LoadedEngineDiagnostics`。
- `.engine-readback.json`。
- `NormalizedCommandSha256`。
- `Skipped=True` / `DependencyProbeOnly`。
- GitHub Actions dry-run。
- GUI screenshot。
- owner input template。
- local feed package consumer、ProjectReference consumer 或 direct `.nupkg` install。

正确的证据梯子应该是：ONNX/source/license -> command normalization -> parser/build report -> engine serialization/hash -> optional load-engine readonly diagnostics -> sample runner real output -> clean external consumer package proof -> post-publish verification -> release close。

## 常见排障

如果 `--onnx` 文件不存在，parser 会在非 dry-run 下抛出 `FileNotFoundException`；dry-run 则只做参数预检。文章示例不要把 dry-run 写成已经构建 engine。

如果出现 CUDA/TensorRT DLL 加载失败，优先检查 `PATH`、CUDA runtime、TensorRT bin/lib、cuDNN 和 adapter line。不要把模型、engine、runtime package 或 NuGet 临时包下载到 C 盘；本项目约定大型资产放 E 盘 workspace。

如果 shape profile 报错，先确认 `--minShapes`、`--optShapes`、`--maxShapes` 的 input name 是否和 ONNX 一致。YOLO 系列常见 input name 是 `images`，但不同 exporter 可能使用 `input` 或其他名字。

如果 `--loadEngine` 只能输出 readonly diagnostics，不要在文章里说它已经验证模型语义。只有执行 enqueue、读取输出、匹配 reference output 并记录 owner evidence，才能往 real-model-runtime proof 走。

## 从模型获取到可复核案例

一篇面向微信公众号或博客的完整案例，不能只给出一条 dotnet run 命令。建议按下面顺序准备资产和证据：

1. **确认模型来源与许可**：记录模型项目主页、版本、下载地址、许可证、导出工具版本和原始文件名；不要把未经许可的模型或权重提交进仓库。
2. **把大文件放到固定外部 workspace**：例如 ..\downloads\cases\<case-id>，分成 models、inputs、engines、reports、logs 和 packages，不要把 ONNX、engine、模型权重或临时 nupkg 放到 %USERPROFILE%\Downloads 或 Temp。
3. **计算来源 hash**：对原始 ONNX、输入样例和必要的模型配置执行 Get-FileHash -Algorithm SHA256，把 modelSourceUrl、license、onnxSha256、inputSha256 和 downloadedAtUtc 写入案例记录。
4. **先做 parser dry-run**：使用 --previewOnly --exportReport 检查 alias、shape profile、precision、输出路径和 NormalizedCommandSha256；此阶段不能创建 engine，也不能写成 build proof。
5. **再做 build-only**：使用 --buildOnly --saveEngine --exportReport --evidenceSidecar，核对 Parsed、EngineSaved、EngineFileRoundTrip、BuilderConfigDeploymentSnapshot、ReadbackMatch 和 engine SHA256。
6. **最后做模型特定 runtime**：只有案例能够定义输入预处理、输出 tensor、reference output、容差和失败诊断时，才执行 enqueue/readback；MNIST 使用 --expectedDigit，YOLO 使用 YoloVision 的 task output contract。
7. **保存可复核日志**：记录完整命令、stdout/stderr、exitCode、host OS、GPU、driver、CUDA、TensorRT line、runtime package key、report SHA256、engine SHA256 和 owner review。

推荐的案例记录字段如下：

caseId
modelName
modelVersion
modelSourceUrl
license
downloadedAtUtc
onnxSha256
inputSha256
exporterVersion
normalizedCommandLine
normalizedCommandSha256
buildReportSha256
engineSha256
runtimeLogSha256
hostOs
gpuName
cudaDriverVersion
cudaRuntimeVersion
tensorRtLine
outputValidationPerformed
ownerReviewed
proofClassification

这套记录把“模型能下载”“engine 能生成”“输出符合预期”和“公开包能被外部用户消费”分成四个问题。缺少 license、来源 hash、reference output、host metadata 或 owner review 时，文章只能展示教程路径，不能宣称 real-model-runtime proof；即使全部齐全，也仍需 clean external consumer 和 post-publish verification 才能进入发布闭环。

## 配图建议

- 一张 ONNX -> parser/build -> engine -> readback diagnostics -> sample runtime -> package proof 的证据梯子图。
- 一张 TensorRtExec report JSON 摘要截图，突出 `ProofClassification`、`BuildEvidenceOnly`、`NormalizedCommandSha256` 和 `LoadedEngineDiagnostics`。
- 一张 YoloVision matrix 截图，展示 YOLO 系列和 det/cls/seg/obb/pose/sem 任务类型。
- 一张路径布局图，展示 E 盘 models、engines、reports、logs 的推荐目录。

## 下一步

继续补齐 TensorRtExec 参数 parity、YoloVision 真实资产候选和 owner result input。转换链路稳定后，把真实运行结果写入 `artifacts/final-release/owner-external-proof-execution-result.input.json`，再通过 import、candidate、strict validator、post-publish verification 和 release close bridge 逐级推进。
