# TensorRtExec

[English](README.en.md) | 简体中文

`applications/TensorRtExec` 是面向最终用户的 ONNX 到 TensorRT engine 工具，提供命令行和 WinForms 两个入口。它通过不可打包的应用共享 Tools 项目复用 trtexec-like 参数模型和 build/report 服务；该共享项目使用已发布的 4 系列托管包编译，因此应用不再引用核心 TensorRT/CUDA 源码项目。

## 当前定位

- CLI：适合自动化脚本、CI 预检查、本地构建记录和报告归档。
- WinForms：适合 Windows 桌面用户选择 ONNX、engine、shape profile、报告路径和 evidence sidecar。
- 共享服务：CLI 和 GUI 都调用 `TensorRtExecService`，避免两个入口产生不同的构建语义。
- 证据边界：默认外部 ONNX 路径是 build/report capable；没有显式 binding、输出语义、真实输入和运行日志时，不声明真实模型 runtime proof。

## 常用命令

外部 ONNX build-only 转换：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --save-engine .\models\model.plan `
  --minShapes input:1x3x640x640 `
  --optShapes input:1x3x640x640 `
  --maxShapes input:4x3x640x640 `
  --fp16 `
  --workspace 1GiB `
  --memPoolSize workspace:512MiB,tacticDram:1GiB `
  --timingCache .\models\model.cache `
  --verbose `
  --builderOptimizationLevel 4 `
  --maxAuxStreams 2 `
  --iterations 10 `
  --warmUp 200 `
  --duration 3 `
  --streams 1 `
  --useCudaGraph `
  --buildOnly `
  --exportReport .\models\model-build-report.json
```

预览参数归一化，不读取 ONNX、不探测 TensorRT runtime：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --save-engine .\models\model.plan `
  --minShapes input:1x3x640x640 `
  --optShapes input:1x3x640x640 `
  --maxShapes input:4x3x640x640 `
  --dryRun `
  --exportReport .\models\model-precheck-report.md
```

## trtexec-like 兼容入口

`TensorRtExec` 与 `applications/OnnxToEngine` 共享同一个 `TrtexecLikeParser`，本阶段补齐了更贴近官方 `trtexec` 的常用别名和 runtime/output 诊断参数：

| 功能 | 推荐参数 | 兼容别名 | 当前状态 |
| --- | --- | --- | --- |
| ONNX 输入 | `--onnx` | `--model`、`--onnxFile` | 归一化为 `--onnx`；只是 build input，不是 runtime proof |
| 保存 engine | `--saveEngine` | `--save-engine`、`--engine`、`--plan`、`--engineFile` | 写出 serialized engine 或 dry-run 预检；当同时存在 ONNX/build-only 意图时，engine 文件别名归一化为 `--saveEngine` |
| 加载 engine | `--loadEngine` | `--load-engine`、无 ONNX/build-only 时的 `--engine` / `--plan` / `--engineFile` | load-engine readonly diagnostics；在 one-float-input / float-output / concrete-shape 条件满足时执行 bounded enqueue/readback；report 输出 `PreflightMetadata`、`LoadedEngineDiagnostics`、文件长度、SHA256、engine/tensor metadata、ReadbackFingerprint、ReadbackSha256 和 proof 边界 |
| Shape alias / batch | `--minShapes --optShapes --maxShapes --batch` | `--shapes`、`--inputShapes` | `--shapes` / `--inputShapes` 会复制到 min/opt/max profile；`--batch` 只进入 normalized command 与报告，不替代 explicit shape profile proof |
| Timing cache | `--timingCacheFile` | `--timingCache` | 成功构建时通过 `TensorRtTimingCache` 导入 caller 文件，并在报告中记录输入大小/SHA256 |
| Profiling verbosity | `--profilingVerbosity detailed` | `--verbose` | 归一化为 `none` / `layer_names_only` / `detailed` |
| Plugin libraries | `--plugins` | `--plugin`、`--dynamicPlugins`、`--setPluginsToSerialize` | 路径会去重并归一化到共享 command；不执行 load/register/deregister，也不证明 plugin 运行 |
| Workspace | `--workspace 512MiB` | 无后缀默认 MiB；支持 `GiB/GB`、`MiB/MB`、`KiB/KB`、`B` | active builder workspace limit |
| Memory pools | `--memPoolSize workspace:512MiB,tacticDram:1GiB` | 无后缀默认 MiB；支持 `workspace`、`dlaSRAM`、`dlaLocalDRAM`、`dlaGlobalDRAM`、`tacticDRAM`、`tacticSharedMem` | 真实 build 时设置并 read back；dry-run/load-engine 仍是 parse-only |
| Runtime timing | `--iterations --warmUp --duration --streams --infStreams --sleepTime --idleTime --avgRuns --percentile --threads --useSpinWait --useCudaGraph --noDataTransfers` | 无 | compatible float engine 的 bounded runtime 会创建独立 execution context/stream；`--sleepTime` 通过 bridge-owned host function 和 CUDA event 一次性扇出到全部推理流；`--threads`、spin wait、graph fallback 与 no-transfer 均记录 requested/applied |
| Advanced timing | `--avgTiming --minTiming` | 无 | `--avgTiming` 在真实 build 中调用 builder-config setter 并 read back；`--minTiming` 只在 TRT8 使用 legacy setter，TRT10/11 保持 parse-only |
| Builder scalar controls | `--maxNbTactics --tilingOptimizationLevel --l2LimitForTiling --quantizationFlags` | 无 | max tactics/tiling/L2 在 TRT10/11 build 中应用并 read back；quantization flags 在 TRT8/10 应用并 read back；不兼容版本输出 controlled diagnostics |
| Deployment policies | `--device --useDLACore --allowGPUFallback --tacticSources --directIO --sparsity --stronglyTyped` | 无 | device 在专用 host thread 设置并读回；真实 build 设置/读回 DLA、fallback、tactic、DirectIO 和 sparsity enable/disable；strongly typed 在 TRT10 使用 raw bit、TRT11 依赖 always-strongly-typed 契约，TRT8 与 sparsity force 保持 parse-only |
| IO / precision policies | `--inputIOFormats --outputIOFormats --precisionConstraints --layerPrecisions --layerOutputTypes` | 无 | 按官方 grammar 校验；TRT8/10 对 tensor/layer typed set/readback，TRT11 仅在现有 tensor type 匹配时应用 IO format，已移除的 precision setters 保持 version-guarded parse-only |
| Engine packaging/refit | `--versionCompatible --excludeLeanRuntime --stripWeights --refit --refitFromOnnx --saveRefittedEngine --allowWeightStreaming --weightStreamingBudget` | 无 | TRT10/11 可执行 parser load、engine commit、显式清除 serialization `ExcludeWeights`、refitted plan 持久化、原 owner 释放、独立 reload 和 context gate；TRT8 保持 parse-only guard |
| Safety / builder cache | `--safe --consistency --builderCache --noBuilderCache` | 无 | parse/report-only；记录 safety/consistency 和 builder cache intent，不声明已执行安全 runtime 或 cache lifecycle |
| Output artifacts / validation | `--loadInputs --dumpOutput --dumpRawBindingsToFile --exportOutput --exportTimes --exportProfile --saveProfile --referenceOutputs --referenceAbsTolerance --referenceRelTolerance --referenceNaNPolicy --referenceInfinityPolicy` | 无 | bounded runtime 按 engine 顺序绑定全部 float input 并捕获全部 float output；structured reference 会逐 tensor 校验 name/shape/count/value 与特殊值策略 |
| Diagnostic reports | `--dumpLayerInfo --exportLayerInfo --dumpProfile --separateProfileRun` | 无 | layer-info 在真实 build/load-engine 中复制 inspector 文本并可导出；profile switches 仍是 intent/report evidence；所有诊断都不是 runtime proof |
| Build report export | `--exportReport` | `--report` | JSON/Markdown report 输出；别名会归一化回 `--exportReport`，报告仍是 build/report evidence，不是 runtime proof |

CLI 还提供 `--help-json` / `--capabilities-json`，用于离线输出共享的 `trtexec-like-option-capabilities.v1` 机器可读能力说明。该 JSON 覆盖 option group、alias、implementation class、parse/report-only 或 blocked 状态、`releaseFrozen=true` 和 `canPromoteRuntimeProof=false`；它不会加载 TensorRT、CUDA、ONNX、plugin 或 engine，只能作为 source-quality capability surface，不能替代 runtime proof、package-consumer-runtime proof 或 release proof。

完整的参数状态判定、32 项 capability surface、85 项 GUI/CLI field map、20 项 gap list、dry-run/build-only/readonly/bounded-runtime 命令和 report validator 流程见 `docs/articles/zh-cn/tensorrtexec-option-layering-deep-dive.md`。
| Timing cache export | `--exportTimingCache` | 无 | 成功构建后序列化并写出 cache，报告记录输出大小/SHA256；仍不是 runtime proof |

WinForms 入口现在也暴露上述 runtime timing、advanced timing、precision policy、packaging/refit、safety/consistency、builder cache、weight budget、timing cache export、layer/profile diagnostics 和 output 字段，GUI 与 CLI 都通过 `TensorRtExecOptions` 生成同一条归一化参数线，避免界面入口与命令行入口出现不同语义。`--dumpLayerInfo`、`--dumpProfile` 和 `--separateProfileRun` 已经进入 CLI/GUI 共享参数模型，但仍只代表报告/诊断 intent；没有真实 enqueue、输入资产、输出校验、日志 hash 和 owner review 时，不能晋级为 runtime proof。

GUI/CLI field map 现在由 `tensor-rt-exec-gui-cli-field-map.json` 和 `tensor-rt-exec-gui-cli-field-map.md` 固化。该 field map 记录 WinForms 控件、CLI option、别名和状态，例如 `--onnx`、`--saveEngine`、`--loadEngine`、`--minShapes`、`--fp16`、`--int8`、`--timingCacheFile`、`--dumpLayerInfo`、`--exportReport`、`--buildOnly` 和 `--dryRun`。它是 surface parity 证据，不是 runtime proof；command preview、GUI 截图、report、sidecar、build-only 和 precheck-only 输出都不能晋级 package-consumer-runtime proof。

Precision/debug parse-report-only boundary：`--fp8`、`--best`、`--dumpRefit`、`--markDebug` 和 `--dumpDebugTensors` 已进入共享 parser、TensorRtExec CLI/WinForms 参数模型、normalized command、diagnostics 和 `OptionImplementationStatus.ParseOnlyOptions`。报告同时输出 `CapabilityProbe`，用于记录当前 host/tool 是否能看到 runtime、builder、builder config、engine inspector API 以及 FP8/debug tensor/weight streaming API。该探针本身仍是 `capability-probe-only`；weight streaming 是否 applied 只由真实 builder flag 与 engine budget readback 决定，不能由 capability probe 晋级。

## 官方 trtexec 对齐状态表

| 分组 | 代表参数 | 状态 | 说明 |
| --- | --- | --- | --- |
| Model/build | `--onnx`、`--saveEngine`、`--loadEngine`、`--workspace`、`--minShapes`、`--optShapes`、`--maxShapes` | implemented | 外部 ONNX build-only、load-engine preflight metadata、bounded runtime output、shape profile 和 workspace 已进入共享 build/report 服务 |
| Shape aliases / batch | `--shapes`、`--inputShapes`、`--batch` | implemented-report | alias 可降低 trtexec 迁移成本；batch 仍不能替代 explicit binding/shape/runtime proof |
| Precision | `--fp16`、`--bf16`、`--noTF32`、`--int8` | partially implemented | FP16/BF16/TF32 进入 builder config；INT8 calibrator 仍是边界诊断 |
| Runtime benchmark | `--iterations`、`--warmUp`、`--duration`、`--streams`、`--avgRuns`、`--percentile`、`--threads`、`--useSpinWait`、`--useCudaGraph`、`--noDataTransfers` | bounded-runtime-scheduler | compatible float engine 执行真实 enqueue、driver threads、CUDA event polling、graph launch 或受控 fallback；no-transfer 不读回输出，外部模型没有 expected output 语义时不提升为模型正确性 proof |
| Runtime streams | `--infStreams` | bounded-runtime-scheduler | 每个 inference stream 创建独立 execution context、bindings、CUDA stream 和 timing events；`--infStreams` 覆盖 legacy `--streams` |
| Advanced timing | `--avgTiming`、`--minTiming` | partially applied | `--avgTiming` 在 TRT8/10/11 build 中记录 `RequestedIterations`、`ReadbackIterations` 和 `ReadbackMatch`；`--minTiming` 只在 TRT8 读取 legacy compatibility setter，TRT10/11 仍是 parse-only。该证据只覆盖 builder config，不是 runtime proof |
| Deployment policy | `--device`、`--useDLACore`、`--allowGPUFallback`、`--tacticSources`、`--directIO`、`--sparsity`、`--stronglyTyped` | applied with guards | CUDA device、builder flag、tactic mask 与 DLA config 记录 requested/readback；DLA core 越界 fail closed。TRT8 strongly typed 与 sparsity force 不会伪装成 applied |
| Wait / idle controls | `--sleepTime`、`--idleTime` | applied in bounded runtime | `--sleepTime` 使用 native bridge 持有的 `cudaLaunchHostFunc` state，再用 event 扇出到全部推理 stream；`--idleTime` 在连续测量轮次间执行 host sleep。两者均记录 requested/applied，但都不证明模型正确性 |
| IO / precision policies | `--inputIOFormats`、`--outputIOFormats`、`--precisionConstraints`、`--layerPrecisions`、`--layerOutputTypes` | applied with version guards | TRT8/10 设置并回读 tensor type/format、constraint flags、layer precision/output type；TRT11 只应用 type 已匹配的 IO format，移除的 layer setters 不伪装 applied |
| Engine packaging | `--versionCompatible`、`--excludeLeanRuntime`、`--stripWeights` | version-guarded set/readback | version/exclude 覆盖 TRT8/10/11；strip 在 TRT10/11 使用 `StripPlan`，未同时指定 refit 时自动选择 `RefitIdentical`；TRT8 strip 保持 parse-only |
| Refit / weight streaming | `--refit`、`--refitFromOnnx`、`--saveRefittedEngine`、`--allowWeightStreaming`、`--weightStreamingBudget` | version-guarded build/refit/persist/runtime readback | `--saveRefittedEngine` 要求 refit source，并在 context 前通过 `serializeWithConfig` 清除/回读 `ExcludeWeights`，再完成 distinct artifact、dispose、reload 和 metadata gate；TRT8 保持 parse-only |
| Safety / consistency | `--safe`、`--consistency` | parse-only | 记录安全 runtime / consistency check 意图，不声明安全 runtime 已真实覆盖 |
| Builder cache policy | `--builderCache`、`--noBuilderCache` | parse-only | 两者互斥；当前只记录 builder cache 策略意图，不声明 cache lifecycle 已提升 |
| Timing cache lifecycle | `--timingCacheFile`、`--exportTimingCache` | applied-build-cache-lifecycle | 成功构建时导入/导出并记录 `TimingCacheArtifact` 的大小与 SHA256；cache 证据不等于 runtime 或 package-consumer proof |

启动桌面界面：

```powershell
dotnet run --project .\applications\TensorRtExec -- --ui
```

## 报告与证据边界

报告可导出为 `.json` 或 `.md`，推荐参数是 `--exportReport`，同时兼容 `--report` 并在 normalized command 中归一化为 `--exportReport`。JSON schema 位于 `applications/TensorRtExec/tensor-rt-exec-report.schema.json`，并与 `OnnxEngineBuildDiagnostics.ToJson` 的真实输出对齐。核心字段包括 `ProofClassification`、`BuildEvidenceOnly`、`DryRun`、`NormalizedCommandLine`、`NormalizedCommandSha256`、`DeploymentOptions`、`BuilderConfigDeploymentSnapshot`、`ParserPreflightSnapshot`、`RuntimeOptions`、`InferenceRan`、`OutputMatch`、`IsRuntimeExecutionProof`、`IsRealModelRuntimeProof`、`IsPackageConsumerRuntimeProof`、`PreflightMetadata`、`LoadedEngineDiagnostics`、`CapabilityProbe`、`WorkspaceBytes`、`OptionImplementationStatus` 和 `ReportBoundary`。`DeploymentOptions` 记录请求值，`BuilderConfigDeploymentSnapshot` 记录成功创建 builder config 后的 pointer-free copied readback，`ParserPreflightSnapshot` 记录 parse 后复制的 parser error count、diagnostic summary、Identity 支持和模型/子图支持计数；这些字段都只是 build/deployment/preflight diagnostics。

TRT10.11/CUDA12.9 compatible-host 的 scalar readback 示例见 `artifacts/real-case/trtexec-builder-scalar-trt10-cuda12/`：`--maxNbTactics`、`--tilingOptimizationLevel`、`--quantizationFlags` 成功 read back；`--l2LimitForTiling 256MiB` 被 vendor setter 拒绝并进入 `ParseOnlyOptions`。该目录的报告和 stdout/stderr 带 SHA256，仍属于 build-only evidence，不代表真实模型输出或 package-consumer runtime。

跨版本矩阵见 `artifacts/real-case/trtexec-builder-scalar-multi-version/`。其中 TRT10/11 的 `--l2LimitForTiling 3MiB` 成功 applied/readback match，而 `256MiB` 保留受控拒绝与实际 `3MiB` readback；TRT8 只应用 quantization flags，现代 scalar 明确 unsupported；TRT11 quantization flags 明确 removed-by-vendor。报告分类仍分别是 dependency-probe-only 或 build-only。

报告 JSON 可用下面的 validator 做发布前边界检查：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-TensorRtExecReport.ps1 `
  -InputPath .\models\model-build-report.json `
  -OutputPath .\artifacts\final-release\tensor-rt-exec-report-validation.json `
  -Strict
```

该 validator 检查 required 顶层字段、`NormalizedCommandSha256`、`OptionImplementationStatus`、`PreflightMetadata`、`LoadedEngineDiagnostics`、`CapabilityProbe`、`ReportBoundary.ForbiddenSubstitutes` 和 `ReportBoundary.CopiedDiagnosticsBoundary`。报告边界会显式记录 `ParserDiagnosticsEvidenceKind=copied-parser-diagnostics`、`ParserRefitterDiagnosticsEvidenceKind=copied-parser-refitter-diagnostics`、`CanPromoteCopiedDiagnosticsToRuntimeProof=False` 和 parser diagnostics owner action。校验通过只能说明 report 可被机器审查，不能把 TensorRtExec report、OnnxToEngine report、ONNX Parser diagnostic snapshot、ONNX ParserRefitter diagnostic snapshot 或 readonly diagnostics 提升成 `real-model-runtime` 或 `package-consumer-runtime` proof。

Runtime/output artifact 通过 `--exportTimes`、`--exportOutput`、`--exportProfile`、`--saveProfile` 和 `--dumpRawBindingsToFile` 输出。`--dumpOutput` 会为每个已读回 output 写入名称、shape、element/byte count、SHA256 和最多 8 个 float 的 preview；`--exportOutput` 的 `OutputTensors` 使用同一份有序快照，避免只保留首个 output。raw 文件按 engine output 顺序连续写入 float32 字节，并在 `<raw-path>.manifest.json` 中记录每个 tensor 的 `ByteOffset`、`ByteLength`、shape 和 SHA256。JSON artifact 还会显式写入 `ArtifactProofBoundary`、`RuntimeProofClass`、`HasTensorOutputProof`、`HasRawBindingProof`、`OutputCaptureAvailable`、`OutputValidated`、`IsBuildOnlyEvidence`、`IsDependencyProbeOnly`、`IsSyntheticRuntime`、`ModelSource`、`EnginePath` 和 `PreflightMetadata`。当这些路径存在时，工具还会在相邻位置写出 `*.engine-readback.json`，用于单独保存 `LoadedEngineDiagnostics`、`ReadbackFingerprint`、`ReadbackSha256`、`ReadbackAvailable` 和 skipped reason。

`--dumpOutput`、`--dumpRawBindingsToFile` 或 `--exportOutput` 本身会请求 generic bounded runtime；没有 `--loadInputs` 时使用可复现的 `deterministic-generated` float 输入，并在日志中显式记录 `InputSource`。这些字段是可审计边界，不是 release proof 替代品。`OutputCaptureAvailable=true` 或 raw bytes 已写入只证明 enqueue 后完成了有界复制；没有 reference match 时必须保持 `OutputValidated=false`、`HasTensorOutputProof=false`、`HasRawBindingProof=false` 和 `runtime-output-captured-unverified`。`build-only` artifact 的 tensor/raw proof 必须为 false；`dependency-probe-only` artifact 只能携带 preflight/readback metadata；`synthetic-input-runtime is not real-model-runtime`，也不是 `package-consumer-runtime`。

`--loadInputs` 现在接受覆盖全部 engine inputs 的 `name:path` 映射，缺失、重复或未知 tensor name 会 fail closed。输出 JSON 同步保存 engine-order `InputTensors`，包括 name、shape、element/byte count、最多 8 个 preview values、SHA256、source classification 和 source path。`--referenceOutputs` 使用同样的 name:path 映射，但每个 path 必须是 `schemaVersion=1` 的结构化 JSON，包含 `tensorName`、`shape`、`values` 与 `sourceClassification`。只有所有 engine outputs 的 name/shape/count/value 在绝对/相对 tolerance 及 NaN/Infinity policy 下全部通过，`OutputValidated` 才为 true；reference 文件 hash 本身不等于数值 correctness，synthetic reference 也不晋级 real-model/package-consumer proof。

YoloVision 的真实运行报告还会复制已有 `TensorRtEngineBindingReport` 到 `bindingMetadata`，记录 input/output mode、semantic role、dtype、engine/profile shape、location、format、vectorization、byte-size fallback 和 diagnostics；控制台同步输出 `BindingReport`/`BindingMetadata`。这是 pointer-free deployment metadata，不是 output correctness、real-model-runtime 或 package-consumer-runtime proof。

Layer/profile diagnostic switch parity is now explicit: real build/load-engine paths use a copied `TensorRtEngineInspector` readback for `--dumpLayerInfo` and `--exportLayerInfo` (log lines or UTF-8 text file), while dry-run and unavailable dependencies remain parse/report-only. `--dumpProfile`、`--separateProfileRun`、`--exportProfile` 和 `--saveProfile` still record profiling intent. Layer text and profile artifacts remain diagnostics, not runtime proof, unless a model-specific runner supplies real execution evidence.

YoloVision owner backfill profiles should be copied into TensorRtExec commands rather than inferred later. The current release-facing defaults are:

| Case | Shape profile | Command intent |
| --- | --- | --- |
| YOLOv8n classification | `--minShapes images:1x3x224x224 --optShapes images:1x3x224x224 --maxShapes images:8x3x224x224` | Labels/Top-K article and owner backfill |
| YOLOv8n detection | `--minShapes images:1x3x640x640 --optShapes images:1x3x640x640 --maxShapes images:4x3x640x640` | Detection layout and NMS evidence candidate |
| YOLOv8n segmentation | `--minShapes images:1x3x640x640 --optShapes images:1x3x640x640 --maxShapes images:4x3x640x640` | Mask prototype and coefficient metadata candidate |
| YOLOv8n pose | `--minShapes images:1x3x640x640 --optShapes images:1x3x640x640 --maxShapes images:4x3x640x640` | Keypoint metadata candidate |
| YOLOv8n OBB | `--minShapes images:1x3x1024x1024 --optShapes images:1x3x1024x1024 --maxShapes images:2x3x1024x1024` | Angle/rotated box metadata candidate |

These profiles are deliberately recorded in `samples/assets/yolovision-real-asset-owner-backfill-pack.json` and are cross-checked against `applications/YoloVision/yolovision-task-output-contract.json`. The contract is the source for task names, output roles, required metadata, and TensorRtExec profile hints across det/cls/seg/obb/pose/sem. They are build/report configuration only. `TensorRtExec` can preserve `--dumpLayerInfo`, `--exportLayerInfo`, `--dumpProfile`, `--separateProfileRun`, `--exportProfile` and `--saveProfile`, but profile dumps become proof only when a model-specific sample runner, such as YoloVision, records real input execution, output JSON, log SHA256 and owner review.

For the six YOLOv8n task/article cases, run `eng/Export-YoloVisionRealAssetOwnerBackfillPack.ps1` to generate `artifacts/user-acceptance/yolovision-real-asset-owner-backfill-sample-run-evidence.template.json` and the projection report. That template carries TensorRtExec report/engine hash slots into sample-run evidence, but it remains `template-only` until owner-filled logs and hashes pass `eng/Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog`.

Raw bindings 只有在 embedded identity synthetic runtime 已真实执行、输出匹配且存在 raw bytes 时才写二进制；build-only、dry-run、load-engine preflight、bounded runtime output without reference match 和外部 ONNX 未声明 binding 语义时会写 skipped JSON，并保持 `HasRawBindingProof=false`。

`--loadEngine` 现在分两层执行。第一层是 readonly diagnostics/report：报告会记录 engine path、文件是否存在、length bytes、SHA256、preflight state、proof classification 和 evidence boundary；在 TensorRT runtime 可用时，会反序列化 engine 并复制 engine name、I/O tensor、layer count、profile count、device memory、aux stream、capability、profiling verbosity、inspector 文本长度、ReadbackFingerprint 和 ReadbackSha256 等只读 metadata。第二层是 bounded runtime：当全部 engine input/output 都是 float，且每个 runtime input shape 可由 engine/profile 或 `--optShapes` 推断时，工具会创建 execution context、按 engine 顺序绑定全部输入输出、enqueue 并导出 output/timing summary。只有 legacy identity match 或完整 reference validation 通过时 `OutputMatch` 才为 true；其中 `OutputValidated` 只代表完整 reference comparison，不由 identity 快捷判断代替。

`OptionImplementationStatus` 会把参数拆成 `ParsedOptions`、`AppliedOptions` 和 `ParseOnlyOptions`。`--refitFromOnnx` 要求 parser load、engine commit、missing/error 清零和 context gate 全部成功；`--saveRefittedEngine` 还要求 serialization config 已清除并回读 `ExcludeWeights`、持久化 artifact 与 stripped plan 不同、原 engine 在 reload 前释放、新 engine metadata gate 完整通过，才会进入 AppliedOptions。TensorRT 10 的完整权重 reload engine 可以不再保持 refittable，`ReloadEngineRefittable` 因而只作诊断，不是 context gate。其余 builder/runtime 选项仍按真实 readback 或执行结果分类；本地持久化和 enqueue 不自动成为 package-consumer 或公开发布 proof。

`GUI/CLI field map` 由 `eng/Export-TensorRtExecGuiCliParityChecklist.ps1` 和 `eng/Test-TensorRtExecGuiCliParityChecklist.ps1` 维护。它记录 CLI token、WinForms 字段、command preview、状态和下一步，但只是 surface parity 证据；GUI 截图、dry-run、build report、timing cache 路径、INT8 calibration cache 路径和 command preview 都不是 runtime proof，也不是 package-consumer-runtime proof。

TrtexecAlignmentStatus=parse-only 是当前高级 trtexec-like 参数的默认保守边界：参数可被 CLI、GUI、parser 和 report 接收，但不能被写成真实 TensorRT 行为已经完成。

新增高级参数会进入 `DeploymentOptions` / `RuntimeOptions` / `NormalizedCommandLine`，并在 diagnostics 和 `OptionImplementationStatus.ParseOnlyOptions` 中保留 parse-only 边界。这个状态是故意保守的：它证明参数已被 CLI、GUI、parser 和 report 接住，但不证明官方 `trtexec` 对应行为已经被 native TensorRT 完整执行。

必须保持以下边界：

- `precheck` / `dryRun` 只证明参数可解析、命令可归一化，不读取模型，不构建 engine。
- `build-only` 只证明 ONNX parser/builder 走到构建报告边界，不证明推理输出正确。
- `dependency-probe-only` 只证明依赖探测或 load-engine preflight 结果，不是 runtime execution proof。
- `capability-probe-only` 只证明只读 API/host capability 可见性，不是 runtime execution proof、real-model-runtime proof 或 package-consumer-runtime proof。
- `synthetic-input-runtime` 可证明最小管线执行，不是用户真实模型质量证明。
- `real-model-runtime` 需要 Classification / YoloVision 等 sample runner 的真实模型、labels、输入资产、hash、许可证和 sample run evidence。
- package-consumer-runtime belongs to release proof records；TensorRtExec build report、sidecar、样例 manifest 都不能直接声明它。
- `sidecar-only` 是模型来源、hash、license、stdout/stderr 摘要和 handoff metadata，不是 runtime proof，也不是 release proof。
- evidence sidecar 可以记录 `package-consumer-runtime` 字符串用于诊断，但校验器会保持 `canPromotePackageConsumerRuntime=false`；sample run evidence record 直接声明 `package-consumer-runtime` 会被判定为 invalid。
- `blocked-by-cuda-driver` 是环境兼容性阻塞，不是 smoke passed，也不是 API 缺口。

## 与样例的关系

- `applications/OnnxToEngine`：最小 identity ONNX round-trip 样例，适合证明 parser、profile、serialized engine、deserialize、binding 和 readback。
- `samples/ComputerVision/01.Classification`：用户自备分类 ONNX、labels 和输入图片后，可形成真实分类模型运行证据。
- `applications/YoloVision`：用户自备 YOLO-family ONNX、labels、图片和后处理 metadata 后，可形成检测、分类、分割、OBB、Pose 或语义分割样例证据。

推荐路径是先用 TensorRtExec 做 build-only 报告，再用具体 sample runner 补真实模型运行日志和 sample-run-evidence record。两类证据互相补充，但不能互相替代。

`tests/fixtures/package-consumers/RefittedPlan.PackageConsumer` 提供更窄但更强的本地包验证：它从两个声明的本地 NuGet source restore
managed 与 TRT10 bridge 包，复制 `--saveRefittedEngine` 生成的完整权重 plan 和输入到仓库外 E 盘工作区，独立
deserialize/enqueue/readback，并把 raw output SHA 与 same-process、second-process、baseline 三路证据精确比对。
该结果分类为 `local-package-consumer-refitted-plan-runtime`，仍不是公开 feed 或 post-publish proof。

## Real Case Proof Pack

`TensorRtExec` 在 `artifacts/final-release/real-case-proof-execution-pack.json` 中有两个 release-facing case：`tensorrtexec-build-report` 和 `tensorrtexec-gui-workflow`。前者覆盖 CLI build/report、normalized command、report hash 和外部 ONNX 元数据；后者覆盖 WinForms GUI 截图、导出报告和 owner-reviewed command line。两者都默认保持 `blocked-owner-action-required`，因为报告和截图不是 runtime proof。

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealCaseProofExecutionPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealCaseEvidenceRecordTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealCaseEvidenceRecord.ps1 -RecordPath .\artifacts\final-release\real-case-evidence-record-template.json
```

`Test-RealCaseEvidenceRecord.ps1` 只在真实 owner 回填模型来源、license、ONNX/engine/input/output SHA256、stdout/stderr log、截图、host OS、GPU、driver、CUDA、TensorRT、runtime package metadata 和 owner review 后才允许形成候选 real-case evidence。即便 CLI/GUI report、sidecar 和截图齐全，仍必须保持 `canPublishPublicly=false`，不能替代 `real-model-runtime`、`package-consumer-runtime`、Linux runner、owner authorization 或 post-publish verification proof。

## 本地质量检查

```powershell
dotnet build .\TensorRtSharp.sln -c Debug --no-restore /p:UseSharedCompilation=false /nr:false

dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj `
  -c Debug --no-build `
  --filter "FullyQualifiedName~TensorRtExec|FullyQualifiedName~OnnxToEngine|FullyQualifiedName~YoloVision"
```
