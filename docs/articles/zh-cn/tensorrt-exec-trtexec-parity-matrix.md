# TensorRtExec 与官方 trtexec 能力对齐矩阵

`applications/TensorRtExec` 的目标不是简单包装一个命令行参数表，而是把官方 `trtexec` 中常用的模型转换、engine 构建、诊断导出和证据边界整理成 C# 用户可以审计、复用、扩展的工具链。当前应用同时提供 CLI 和 WinForms 入口，并复用 `src/JYPPX.TensorRtSharp.Tools` 的 `TrtexecLikeParser`、`OnnxEngineBuildOptions`、`OnnxEngineBuildService` 与 report/export 逻辑。

本文是发布前 parity matrix：它说明哪些能力已经可用，哪些能力只是 parse/report-only，哪些能力必须等真实 TensorRT 行为、模型 smoke 或 package consumer proof 证明后才能提升。

> 证据边界：`TensorRtExec` parity 文档、build report、dry-run report、GUI 截图、sidecar 和 `CapabilityProbe` 都不是 runtime proof。`build-only`、`parse-only`、`dry-run`、`dependency-probe-only`、`capability-probe-only` 不能晋级。只有 clean external package consumer smoke 通过，并由 proof validator 判定为 `package-consumer-runtime`，才可能作为发布关闭证据。

## 快速入口

命令行入口：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model.plan `
  --minShapes input:1x3x640x640 `
  --optShapes input:1x3x640x640 `
  --maxShapes input:4x3x640x640 `
  --fp16 `
  --workspace 1024 `
  --exportReport .\models\model-build-report.json `
  --buildOnly
```

WinForms 入口：

```powershell
dotnet run --project .\applications\TensorRtExec -- --ui
```

## Parity Matrix

| # | 能力 | 官方 `trtexec` 对应项 | `TensorRtExec` 当前状态 | 当前入口 | 缺口 | 下一步 | 可作为 runtime proof |
|---|---|---|---|---|---|---|---|
| 1 | ONNX 模型输入 | `--onnx=model.onnx` | implemented | `TensorRtExecOptions.OnnxPath`、`OnnxEngineBuildOptions.FromTrtexecLikeOptions` | 真实模型质量仍依赖 sample runner 和资产 manifest | 继续把真实模型 runner 的 sidecar 与 sample-run-evidence 串起来 | 否，build-only 不是 runtime proof |
| 2 | engine 保存 | `--saveEngine=model.plan` | implemented | `--saveEngine`、`--save-engine`、`--engine` | 需要真实模型 hash、log hash、owner review 才能进入案例 evidence | 与 OnnxToEngine/YoloVision 真实模型运行记录交叉引用 | 否，保存 engine 只证明构建/序列化边界 |
| 3 | engine 加载 | `--loadEngine=model.plan` | readonly-diagnostics / bounded-runtime-output | `--loadEngine`、`--load-engine`、`LoadedEngineDiagnostics`、`TensorRtInferenceBindings`、runtime artifacts | compatible float engine 会创建 typed bindings 并 enqueue/readback；没有 expected output 时仍是 `runtime-output-captured-unverified` | 由真实模型 expected output、runner log 和 owner hash 提升模型正确性 | 否，bounded enqueue 不等于模型或 package proof |
| 4 | dynamic shape | `--minShapes` / `--optShapes` / `--maxShapes` | implemented for build/report | `ShapeProfile`、`TrtexecLikeShapeProfile` | 模型是否真正覆盖所有 profile 仍需要模型级 smoke | 增加 dynamic-shape real model case evidence | 否，profile 配置不是输出正确性 proof |
| 5 | min/opt/max shape profile | `--minShapes=input:...` 等 | implemented | CLI、WinForms、README 示例 | 需要更多真实模型 shape profile 案例 | 在 YoloVision 与 Classification 文章中补真实 shape 采集流程 | 否 |
| 6 | shape alias / batch migration | `--shapes`、`--inputShapes`、`--batch` | implemented-report | `TrtexecLikeParser`、`TrtexecLikeOptions.Batch`、CLI help、README、parity artifact | alias 和 batch 只降低官方 trtexec 迁移成本，不证明 profile 覆盖、binding 语义或输出正确 | 在真实 sample runner 中记录 selected shape、batch、输出 hash 和 owner review | 否 |
| 7 | FP16 | `--fp16` | wrapper-ready / applied when supported | `TensorRtExecOptions.Fp16` | host GPU、TensorRT line、模型 layer 支持需要真实 smoke | 在 compatible host proof 中记录 FP16 host/model 结果 | 仅凭开关否 |
| 8 | INT8 | `--int8`、`--calib` | diagnostic / boundary | `TensorRtExecOptions.Int8`、`CalibrationCacheFile` | calibrator/cache ownership 仍需更严格桥接和真实校准 evidence | 先做 INT8 field guide，再做 owner-provided calibrator proof | 否 |
| 9 | workspace / memory pool | `--workspace`、`--memPoolSize` | implemented-build-readback | `WorkspaceBytes`、`MemoryPoolSizes`、`SetMemoryPoolLimit`、`GetMemoryPoolLimit`、`TrtexecMemoryPool` log | readback 只证明 builder config 接收请求，不证明 runtime 输出、模型准确率或 package consumer | 在 compatible host 记录 requested/readback bytes 与 owner hash | 否 |
| 10 | timing cache | `--timingCacheFile`、`--exportTimingCache` | implemented-build-cache-lifecycle | `TimingCacheFile`、`ExportTimingCachePath`、`OnnxEngineBuildResult.TimingCacheArtifact` | 成功构建时通过 typed timing-cache owner 导入/导出并记录大小与 SHA256；仍不是 runtime proof | 在兼容主机补真实构建日志、owner review 和模型级输出校验 | 否 |
| 10a | timing iterations | `--avgTiming`、`--minTiming` | implemented-builder-config-readback | `SetAverageTimingIterations` / `GetAverageTimingIterations`；TRT8 `SetMinTimingIterationsCompatibility`；`TrtexecTiming` log | `--avgTiming` 的 setter/readback 只证明 builder config；TRT10/11 的 legacy `--minTiming` 保持 parse-only | 收集 TRT8/10/11 compatible-host build log，并保持 tactic quality、benchmark performance 和 runtime proof 分离 | 否 |
| 10b | deployment policies | `--device`、`--useDLACore`、`--allowGPUFallback`、`--tacticSources`、`--directIO`、`--sparsity`、`--stronglyTyped` | implemented-build-readback-with-version-guards | 专用 device thread、typed builder config set/get、network creation flags、`TrtexecDeploymentControl` log | readback 只证明主机上的策略应用；TRT8 strongly typed 与 sparsity force 保持 parse-only，DLA 执行仍需真实模型和 DLA 主机 | 保留 TRT10 build/enqueue evidence；另行采集 DLA owner proof | 否 |
| 10c | I/O 与 layer precision policy | `--inputIOFormats`、`--outputIOFormats`、`--precisionConstraints`、`--layerPrecisions`、`--layerOutputTypes` | implemented-build-readback-with-version-guards | `TrtexecLikeBuildPolicy`、tensor type/format、constraint flags、layer precision/output type、`TrtexecBuildPolicy` log、跨版本 YOLOv8n-cls 证据 | TRT10.11 完成 set/readback；TRT11.0 对 type 已匹配的 FP32 I/O format 报告 applied，移除的 setters 保持 parse-only；两者均通过 1000 值参考校验与 hashed detailed JSON；TRT8 parser 创建已有原生 SEH/status 边界并完成隔离 MNIST parse/build/enqueue，但尚无 TRT8 detailed layer-policy 证据 | 保留版本化证据、caller layout 和 owner hash，不能跨版本投影或靠 synthetic MNIST smoke 晋级模型证明 | 否 |
| 10d | Engine packaging、refit 与 weight streaming | `--versionCompatible`、`--excludeLeanRuntime`、`--stripWeights`、`--refit`、`--allowWeightStreaming`、`--weightStreamingBudget` | implemented-build-runtime-readback-with-version-guards | packaging/refit builder flags、runtime host-code policy、engine refittable 与 V2 budget/scratch readback、YOLOX-S weighted enqueue、TRT10/11 parser-refitter lifecycle | TRT8 strip/streaming 与 version-compatible+refit 冲突保持 parse-only；TRT11 已通过 stripped-plan refit、持久化 reload、loadEngine-only 第二进程和零 mismatch comparison；TRT11 version-compatible+refit、weight streaming 与 external lean runtime 仍未验证，本地结果也不是模型精度或 package proof | 保留 compact evidence 严格验证；另行采集 TRT11 version-compatible、weight-streaming、external lean-runtime 与 public-feed post-publish consumer proof | 否 |
| 11 | plugin library 参数边界 | `--plugins`、`--plugin`、`--dynamicPlugins`、`--setPluginsToSerialize` | diagnostic-alias-compatible / boundary | `TrtexecLikeParser.ParsePluginLibraries`、`Plugins`、Plugin Inventory 只读 API | register/load/deregister library 和 serialized plugin ownership 暂不处理，避免 ownership 风险 | 仅做 plugin path normalization 和 plugin inventory copied metadata appendix，不做 load library | 否 |
| 12 | profiling | `--profilingVerbosity`、`--dumpProfile`、`--exportProfile` | implemented-report / parse-only mixed | `ProfilingVerbosity`、`ExportProfilePath`、`SaveProfilePath` | layer runtime timing 需要真实 enqueue 与 profile log | 先补 profile artifact schema，再做 real model smoke | 否 |
| 12a | bounded benchmark scheduler | `--iterations`、`--warmUp`、`--duration`、`--streams`、`--infStreams`、`--sleepTime`、`--avgRuns`、`--percentile`、`--threads`、`--useSpinWait`、`--useCudaGraph`、`--noDataTransfers` | implemented-bounded-runtime | `RunBoundedBenchmark`、`OnnxEngineBenchmarkSummary`、times artifact | 每个 effective stream 使用独立 context/bindings/CUDA stream/event；可选独立 driver thread、event polling、一次性 start-delay event fan-out、graph capture/launch 或 fallback、零 H2D/D2H；次数和时长均为最低条件 | no-transfer 不声明 output match；继续由模型 runner 提供 expected output，不把 scheduler evidence 当成模型正确性 | 否 |
| 13 | wait / idle benchmark controls | `--sleepTime`、`--idleTime` | implemented-bounded-runtime | `CudaStream.EnqueueDelay`、`jyppx_cuda_stream_enqueue_delay_safe`、`RunBoundedBenchmark`、`OptionImplementationStatus` | `--sleepTime` 使用 bridge-owned native callback state 并把一个 CUDA event 扇出到全部推理 stream；`--idleTime` 在连续测量轮次间应用 host sleep | 保留 TRT10/CUDA12.9 本地 smoke，并在真实模型/clean consumer 中继续记录 requested/applied | 否 |
| 14 | layer dump | `--dumpLayerInfo`、`--exportLayerInfo` | implemented-inspector-readback | `ExportLayerInfoPath`、自描述 JSON、顶层 `LayerInfoArtifact` 长度/SHA256、strict validator | TRT10.11 与 TRT11.0 已有真实 engine readback；external ONNX metadata 仍不等于完整 layer semantic/runtime proof；TRT8 parser SEH 已隔离且 MNIST build/enqueue 通过，但本次未请求 layer artifact | 保留版本化产物 hash，另行采集并严格验证 TRT8 detailed inspector JSON，不能用本次 synthetic runtime 代替 | 否 |
| 15 | precision/debug boundary | `--fp8`、`--best`、`--dumpRefit`、`--allowWeightStreaming`、`--markDebug`、`--dumpDebugTensors` | parse-report-only / capability-probe-only | `TrtexecLikeDeploymentOptions`、`TensorRtExecOptions`、WinForms precision/packaging 字段、`OptionImplementationStatus.ParseOnlyOptions`、`CapabilityProbe` | 只证明参数接入、报告和只读能力可见性，不证明 FP8、best precision、refit、weight streaming 或 debug tensor runtime output | 下一阶段补模型级 smoke、artifact hash、native 行为 readback | 否 |
| 16 | report export alias | `--exportReport`、`--report` | implemented-report | `TrtexecLikeParser`、`ExportReportPath`、`OnnxEngineBuildDiagnostics.WriteReport` | alias parsing 已完成，但 report 仍只是 build/report evidence | 保持 `--exportReport` 为 canonical normalized command，owner-facing 文档可使用 `--report` 别名 | 否 |
| 17 | verbose logging | `--verbose` | implemented-report | `ProfilingVerbosity=detailed`、LogLines | verbose log 不是成功运行证明 | 在 owner proof 中要求 stdout/stderr summary 和 log SHA256 | 否 |
| 18 | input/output binding metadata 与 reference validation | `--loadInputs`、`--dumpOutput`、`--dumpRawBindingsToFile`、`--referenceOutputs`、absolute/relative tolerance、NaN/Infinity policy | implemented-pointer-free-multi-input-binding-multi-output-artifacts-and-reference-validation | `OnnxEngineRuntimeInputArtifact`、`OnnxEngineReferenceTensorData`、`OnnxEngineReferenceValidationArtifact`、raw offset/hash manifest、YoloVision `bindingMetadata` | 全部 float inputs/outputs 按 engine 顺序处理；只有完整 structured reference comparison 才设置 `OutputValidated`；synthetic reference 仍不是 real-model/package proof | 将 reference JSON 与 YoloVision task contract、owner-reviewed expected output 和模型/输入/engine/log hashes 配对 | 仅真实模型 smoke 与 reviewed reference 全通过后可作为 sample evidence，不是 package proof |
| 19 | package consumer / runtime package key proof 边界 | 无直接等价，属于发布 proof | release proof records only | `package-consumer-runtime-proof-*` scripts | 需要公开包源、clean external consumer、真实 host metadata、exitCode=0、hash 对齐 | 由 owner input + validator + post-publish clean consumer proof 完成 | 是，但必须由 `package-consumer-runtime` validator 判定 |

## 参数状态分层

`TensorRtExec` 的参数状态必须区分：

- `implemented`：工具链已经能解析、传递并在 build/report 服务中产生明确效果。
- `implemented-report`：能进入报告或 artifact，但报告不等于 runtime proof。
- `implemented-builder-config-readback`：真实 build 会调用 typed builder-config setter 并回读请求值；这仍是 builder evidence，不是 runtime proof。
- `implemented-bounded-runtime`：compatible float engine 已执行 typed owner、enqueue 和 GPU timing；这只证明调度器执行，不自动证明模型语义或包消费。
- `wrapper-ready`：C# wrapper 已有表达形态，真实硬件/模型效果需要 smoke。
- `diagnostic`：只记录边界和意图，不声明 TensorRT 行为已执行。
- `parse-only`：CLI/GUI/parser/report 接住参数，但不能宣称官方 `trtexec` 对应行为已经完整实现。
- `capability-probe-only`：只读探测 runtime/builder/API 可见性和高级参数 intent，不声明模型构建行为、enqueue、输出校验或发布包消费已完成。
- `planned`：文档化下一步，不作为当前能力。

这些状态是故意保守的。真实 bounded runtime 可以把 `--iterations`、`--warmUp`、`--duration`、有效的 `--streams/--infStreams`、`--sleepTime`、`--idleTime`、`--avgRuns`、`--percentile`、`--threads`、`--useSpinWait` 和 `--noDataTransfers` 标为 applied；`--useCudaGraph` 只有所有 context 捕获并实例化成功时才 applied，否则继续 parse-only 并报告 fallback reason。真实 build 中成功 set/readback 的 device/deployment controls 以及 I/O/layer policies 也可标为 applied，但这仍不是 DLA 执行、binding layout、tactic 选择、模型正确性或 package consumer proof。被 `--infStreams` 覆盖的 `--streams`、`--batch`、TRT10/11 的 `--minTiming`、TRT8 的 `--stronglyTyped`、TRT11 已移除的 precision setters、`--sparsity=force` 以及其余没有真实行为证据的参数不能写成已完成。`--avgTiming`、TRT8 legacy `--minTiming` 与 precision policy readback 只代表 builder evidence，timing-cache lifecycle 也只代表构建缓存证据。

## 与 OnnxToEngine 的关系

`applications/OnnxToEngine` 适合作为最小 ONNX-to-engine 教程和 identity round-trip smoke；`applications/TensorRtExec` 适合作为最终用户工具，覆盖 CLI、WinForms、report、sidecar、parity matrix 和 release-facing 边界说明。两者可以共享 parser 和 build service，但不互相替代：

- OnnxToEngine：更像教程和最小样例。
- TensorRtExec：更像应用程序和发布前诊断工具。
- 两者输出的 build report 都不能替代 `package-consumer-runtime` proof。

## 发布前使用建议

1. 先用 `TensorRtExec --dryRun` 生成参数归一化报告，确认模型路径、shape profile、precision 和 artifact 路径。
2. 再用 `TensorRtExec --buildOnly` 构建 engine 和 report，记录 ONNX/engine/log SHA256。
3. 对真实模型，转入 `samples/ComputerVision/01.Classification` 或 `applications/YoloVision` 运行带输入资产的 sample runner。
4. 对发布关闭，转入 clean external consumer proof，回填 owner input，运行 validator。

## 不能替代 proof 的材料

以下材料可以用于诊断或文章展示，但不能用于关闭 release proof：

- `dry-run`
- `parse-only`
- `capability-probe-only`
- `build-only`
- `dependency-probe-only`
- `sidecar-only`
- 本地 `.nupkg`
- local feed
- ProjectReference
- direct `.nupkg`
- GUI 截图
- README / 技术文章
- `TensorRtExec` build report
- `OnnxToEngine` 教程输出

真实 proof 仍然必须落到 validator 可判定的记录：`real-model-runtime` 用于样例资产证明，`package-consumer-runtime` 用于发布包消费者证明。
