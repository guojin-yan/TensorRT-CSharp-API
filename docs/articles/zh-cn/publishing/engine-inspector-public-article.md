# Engine Inspector：如何读取 engine 信息而不伪造 runtime proof

Engine Inspector 是 TensorRtSharp4.0 里非常适合排障和文章展示的一类能力。它可以在 engine build 或 `--loadEngine` readonly diagnostics 阶段读取 engine 名称、IO tensor、layer count、profile count、device memory、auxiliary streams、profiling verbosity、inspector text 和 readback hash，帮助用户判断 engine 结构是否符合预期。

但它的边界也必须说清楚：Engine Inspector 不创建 execution bindings，不绑定输入输出，不 enqueue，不读取真实输出 tensor，也不验证模型语义。它是 readonly diagnostics，不是 real-model-runtime proof，也不是 package-consumer-runtime proof。

## 适合

- 想调试 serialized engine 层信息、IO tensor 和 profiling verbosity 的使用者。
- 需要理解 `TensorRtEngineInspector` 与 `applications/TensorRtExec` load-engine readonly diagnostics 边界的维护者。
- 正在区分 build/read-only diagnostics、runtime-output-unverified、real-model-runtime proof 与 package-consumer-runtime proof 的发布负责人。
- 想把 TensorRtExec 的 `--dumpLayerInfo`、`--exportLayerInfo`、`--profilingVerbosity` 写成公开教程的文章作者。

## 关键路径

- 高层入口：`src/JYPPX.TensorRtSharp/Engine/TensorRtEngineInspector.Trt11Diagnostics.cs`。
- loaded engine 诊断模型：`src/JYPPX.TensorRtSharp.Tools/Build/OnnxEngineBuildResult.cs`。
- engine readback artifact：`src/JYPPX.TensorRtSharp.Tools/Artifacts/OnnxEngineRuntimeArtifactWriter.cs`。
- 工具投影：`src/JYPPX.TensorRtSharp.Tools/Build/OnnxEngineBuildDiagnostics.cs`。
- TensorRtExec 报告：`applications/TensorRtExec/Core/TensorRtExecReport.cs`。
- CLI/GUI 输出：`applications/TensorRtExec/Console/TensorRtExecCommand.cs` 和 `applications/TensorRtExec/WinForms/MainForm.cs`。
- parity matrix：`applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json`。

## Inspector 能读什么

`TensorRtEngineInspector` 的安全读取面是复制型字符串和只读诊断：

```text
GetLayerInformation(layerIndex, format)
HasExecutionContext
ClearExecutionContext()
HasErrorRecorder
TryGetErrorRecorderSnapshot(out snapshot)
ClearErrorRecorder()
```

其中 `GetLayerInformation` 返回 copied UTF-8 string；error recorder 走 copied snapshot，不暴露 borrowed recorder pointer，也不改变 recorder ownership。`HasExecutionContext` 只说明 inspector 是否关联了 execution context；它本身不代表 TensorRT 已执行 inference。

在工具层，`OnnxLoadedEngineDiagnostics` 记录这些字段：

```text
Attempted
Succeeded
DiagnosticsState
FailureReason
EngineName
IOTensorCount
LayerCount
OptimizationProfileCount
DeviceMemorySizeInBytes
AuxiliaryStreamCount
Capability
ProfilingVerbosity
InspectorInformationLength
IOTensorSummaries
ReadbackFingerprint
ReadbackSha256
EvidenceBoundary
```

这些字段适合被写进 report、CLI 输出、WinForms summary 和文章截图。它们解决的是“engine 元数据能不能读出来”，不是“模型输出是否正确”。

## Inspector 边界分层

Engine Inspector 的读取面可以拆成三层，每一层都应该保持 copied/pointer-free：

| 层级 | 典型 API / 字段 | 能证明 | 不能证明 |
| --- | --- | --- | --- |
| Engine-level metadata | `GetEngineInformation`、`EngineName`、`IOTensorCount`、`LayerCount`、`OptimizationProfileCount` | engine 结构摘要可读取 | 输入输出绑定正确、模型输出正确 |
| Layer-level metadata | `GetLayerInformation(layerIndex, format)`、`TensorRtLayerInformationFormat.Oneline`、`TensorRtLayerInformationFormat.Json` | layer text/json 可复制 | per-layer timing 或 layer 执行成功 |
| Association state | `HasExecutionContext`、`ClearExecutionContext()`、`HasErrorRecorder`、`TryGetErrorRecorderSnapshot`、`ClearErrorRecorder()` | inspector 是否挂过 context/error recorder，且 error snapshot 可复制 | context enqueue、callback 生命周期或错误恢复正确 |

`SetEngineInspectorExecutionContext` 可以让 inspector 读取与 context 相关的 layer 信息，但这仍是 metadata readback。如果没有 binding address、input shape、device buffer、stream synchronization、output tensor copy 和 validator，就不能把它写成 inference proof。

错误记录器也同理。`TensorRtErrorRecorderSnapshot`、`TensorRtErrorRecorderSummary` 和 `TensorRtErrorRecord` 都是 copied diagnostics。它们不能把 native error recorder 指针暴露给 C#，也不能证明用户自定义 recorder callback lifecycle 已经安全。

## Readback artifact 字段

`.engine-readback.json` 建议保留这些字段，方便 reviewer 快速判断边界：

```text
ArtifactKind
ArtifactBoundary
EngineInspectorApiAvailable
DiagnosticsState
FailureReason
SkippedReason
InspectorInformationLength
IOTensorSummaries
ReadbackFingerprint
ReadbackSha256
IsRuntimeExecutionProof
IsRealModelRuntimeProof
IsPackageConsumerRuntimeProof
RuntimeOutputCaptured
OutputValidationPerformed
```

其中 `EngineInspectorApiAvailable=True` 只说明 managed/native API surface 可用，可能仍然是 capability-probe-only。`InspectorInformationLength > 0` 只说明 TensorRT 返回了 inspector text。`ReadbackSha256` 可以用来比较两次 metadata，但不是 runtime log hash，也不是 public package hash。

如果 artifact kind 是 `trtexec-like-engine-readback-skipped`，应优先看 `SkippedReason` 和 `DiagnosticsState`。跳过 readback 不等于失败，也不能被改写成成功；它只是说明本次没有足够条件读取 engine metadata。

## TensorRtExec 使用方式

在 TensorRtExec 或 OnnxToEngine 中，常见的 inspect 场景有两种。

第一种是 build 后导出 report：

```powershell
dotnet run --project .\applications\TensorRtExec\TensorRtExec.csproj -- `
  --onnx E:\TensorRtSharpAssets\models\model.onnx `
  --saveEngine E:\TensorRtSharpAssets\engines\model.plan `
  --profilingVerbosity detailed `
  --dumpLayerInfo `
  --exportLayerInfo E:\TensorRtSharpAssets\reports\model.layers.json `
  --exportReport E:\TensorRtSharpAssets\reports\model-build-report.json `
  --buildOnly
```

第二种是 load existing engine 后做 readonly diagnostics：

```powershell
dotnet run --project .\applications\TensorRtExec\TensorRtExec.csproj -- `
  --loadEngine E:\TensorRtSharpAssets\engines\model.plan `
  --profilingVerbosity layer_names_only `
  --exportProfile E:\TensorRtSharpAssets\reports\model-profile.json `
  --dumpLayerInfo `
  --buildOnly
```

如果设置了 `--exportProfile`、`--exportTimes`、`--exportOutput`、`--saveProfile` 或 `--dumpRawBindingsToFile`，工具会派生 `.engine-readback.json`。典型 artifact kind 是：

```text
trtexec-like-engine-readback
trtexec-like-engine-readback-skipped
```

readback 成功时会包含 `ReadbackFingerprint` 和 `ReadbackSha256`。这两个字段能证明本次 readonly metadata 有稳定摘要，方便比较两次 engine 或写入候选记录。

## Report 字段怎么读

`TensorRtExecReport` 把 engine inspector 相关状态投影到应用层：

```text
LoadEngineDiagnosticsState
LoadEngineDiagnosticsAttempted
LoadEngineDiagnosticsSucceeded
LoadEngineDiagnosticsBoundary
ProofClassification
BuildEvidenceOnly
InferenceRan
ReportPath
NormalizedCommandSha256
```

关键是同时看 `LoadEngineDiagnosticsBoundary`、`ProofClassification` 和 `InferenceRan`：

- `LoadEngineDiagnosticsSucceeded=True`：说明 engine metadata 读取成功。
- `InferenceRan=False`：说明没有执行 TensorRT enqueue。
- `BuildEvidenceOnly=True`：说明它仍是构建/诊断证据。
- `ProofClassification=build-only` 或 `load-engine-readonly-diagnostics`：说明不能晋级 runtime proof。
- `NormalizedCommandSha256`：说明命令可追踪，但 command hash 不是 runtime hash。

如果 `InferenceRan=True` 但 `OutputMatch=False`，那也只是 `runtime-output-unverified` 或 bounded runtime evidence，仍不能被写成 real-model 或 package-consumer proof。

## Bounded runtime 与 inspector 的分界

TensorRtExec 有时会同时输出 load-engine readonly diagnostics 和 bounded runtime 信息。两者必须分开读：

```text
LoadEngineReadonlyDiagnostics Attempted=True Succeeded=True
LoadEngineBoundedRuntime Attempted=True Succeeded=True
InferenceRan=True
OutputMatch=True
```

第一行只能证明 inspector/readback 成功；bounded runtime 行才说明这次工具尝试了受控输入、binding 和 enqueue。即便 bounded runtime 成功，也通常只适用于 embedded identity model 或明确的模型专用 runner；它仍不是任意真实模型 proof，也不是 package-consumer-runtime proof。

如果 report 同时包含 `CapabilityProbe State=capability-probe-only` 和 `EngineInspectorApiAvailable=True`，应按更窄的 evidence kind 解读：capability probe 只能证明 host/tool API 可见，不证明 build、load、enqueue 或 output match。

建议 report 中显式保留这些布尔字段：

```text
EngineInspectorReadonlyMetadata = true
EngineInspectorCreatedExecutionBindings = false
EngineInspectorEnqueuedInference = false
EngineInspectorValidatedOutputs = false
EngineInspectorCanPromoteRuntimeProof = false
EngineInspectorCanPromotePackageConsumerProof = false
```

这样 WinForms、CLI 和文章截图都能清楚表达：inspector 是排障能力，不是 proof 升级器。

## Engine Readback Artifact 边界

`OnnxEngineRuntimeArtifactWriter` 的 engine readback artifact 会明确写入：

```text
IsRuntimeExecutionProof = false
IsRealModelRuntimeProof = false
IsPackageConsumerRuntimeProof = false
```

并且 note 会说明：

```text
Engine readback artifact is readonly metadata evidence only; it does not bind tensors, enqueue inference, validate outputs, or prove real-model/package-consumer runtime.
```

loaded engine diagnostics 的 `EvidenceBoundary` 也会保持同样口径：

```text
load-engine readonly diagnostics may deserialize the engine and copy metadata, but it does not create execution bindings, enqueue inference, validate outputs, or prove package-consumer-runtime.
```

这两个边界非常重要。`--loadEngine` 可以反序列化 engine 并复制 metadata，但只要没有真实输入、binding、enqueue、输出读回和 validator，就不能说 runtime proof 已完成。

## 与 trtexec 的关系

官方 trtexec 有 `--dumpLayerInfo`、`--exportLayerInfo`、`--profilingVerbosity`、`--dumpProfile`、`--exportProfile`、`--saveProfile` 等能力。TensorRtExec 的对应目标是：

- CLI 和 WinForms 都能表达这些选项。
- build/load-engine 路径能复制 layer text 和 profile/report metadata。
- dry-run 和 dependency-unavailable 路径保持 parse/report-only。
- 不伪造 per-layer timing，不伪造 output correctness。
- 真实 profile proof 必须来自 enqueue 后的日志和 hash。

因此，`applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json` 中 profiling 和 layer-dump 的状态可以是 implemented-report / implemented-inspector-readback，但 proof 边界仍然是 diagnostic metadata。

## proof 边界

Engine inspector 输出属于 readonly diagnostics。以下材料都不能作为 package-consumer-runtime proof：

- inspector text。
- `.engine-readback.json`。
- `ReadbackFingerprint` / `ReadbackSha256`。
- layer dump。
- profile boundary artifact。
- TensorRtExec GUI screenshot。
- build-only report。
- load-engine readonly diagnostics。
- local feed package consumer。
- ProjectReference consumer。
- direct `.nupkg` install。

真实 package-consumer-runtime proof 至少需要：

- 外部 clean consumer 使用 public package source restore。
- managed/runtime public package identity 和 SHA256。
- 真实 TensorRT runtime smoke 运行成功。
- 非 ProjectReference、非 local feed、非 direct `.nupkg`。
- stdout/stderr/log path 与 SHA256 对齐。
- host GPU、driver、CUDA、TensorRT、cuDNN metadata。
- owner input validator 与 record validator 通过。

特别要避免三种误读：

- `ReadbackSha256` 不是 smoke log SHA256。
- `InspectorInformationLength` 不是 per-layer timing。
- `EngineInspectorApiAvailable` 不是 TensorRT runtime 已经执行。

## 常见排障

- `EngineInspectorApiAvailable=False`：通常表示当前 TensorRT line、native bridge 或依赖不可用；这属于 capability/dependency probe，不是 runtime proof。
- `trtexec-like-engine-readback-skipped`：说明没有足够 readback 信息或 diagnostics 未执行，应看 `SkippedReason`。
- `InspectorInformationLength=0`：可能是 profiling verbosity、engine 内容或 TensorRT line 差异导致，需要结合 layer count 和 IO summaries 判断。
- `ReadbackSha256` 变化：说明 engine metadata 摘要变化，适合追溯构建参数、模型版本或 refit/weight streaming 差异。
- GUI 能显示 report：只说明 WinForms 成功展示结果，不是机器可验证 proof。

## 配图建议

- 一张流程图：load engine -> deserialize metadata -> copy IO/layer/profile/readback hash -> report，只到 readonly diagnostics。
- 一张 `.engine-readback.json` 截图，突出 `ReadbackFingerprint`、`ReadbackSha256`、`IsPackageConsumerRuntimeProof=false`。
- 一张“不发生的步骤”图：没有 create execution bindings、没有 enqueue、没有 output validation。
- 一张 TensorRtExec CLI/WinForms 对照图，展示 `--dumpLayerInfo`、`--exportLayerInfo` 和 `--profilingVerbosity`。

## 下一步

继续扩展 inspector 的安全 readback 字段，优先采用 copied string、copied snapshot、count/copy metadata 和 pointer-free diagnostics。不要把 inspector text、engine-readback artifact、build-only report、GUI screenshot 或 local package consumer 写成 package-consumer-runtime proof；真实 proof 仍必须交给 clean external consumer runtime smoke、owner input validator、record validator 和 post-publish verification。
