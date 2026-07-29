# Builder Config Readback：发布前如何确认构建参数真的生效

TensorRT engine 的构建参数很多：workspace、memory pool、profiling verbosity、tactic sources、DLA、DirectIO、sparsity、timing iterations、version compatible、refit、weight streaming 等。TensorRtSharp4.0 的 Builder Config Readback 目标，是把这些“用户传入的意图”变成“TensorRT 已接受并能读回的证据”，让 OnnxToEngine 和 TensorRtExec 的构建报告不只写命令行，还能写清楚每个关键选项是否真正应用。

这类证据非常适合发布前排障和技术文章展示，但它仍然是 build/report evidence，不是 runtime proof，也不是 package-consumer-runtime proof。它能回答“构建配置是否生效”，不能单独回答“公开包是否在外部用户机器上完成真实推理”。

## 适合

- 正在使用 `samples/OnnxToEngine` 或 `applications/TensorRtExec` 构建 engine 的用户。
- 需要确认 `--workspace`、`--memPoolSize`、`--avgTiming`、`--tacticSources`、`--profilingVerbosity` 是否真正应用的维护者。
- 想把 TensorRT 8、TensorRT 10、TensorRT 11 的 version guard 和配置差异讲清楚的文章作者。
- 负责 release evidence 审核，但需要区分 build/report evidence 与 package-consumer-runtime proof 的发布负责人。

## 关键路径

- 高层配置：`src/JYPPX.TensorRtSharp/Builder/TensorRtBuilderConfig.cs`。
- TRT11 diagnostics：`src/JYPPX.TensorRtSharp/Builder/TensorRtBuilderConfig.Trt11Diagnostics.cs`。
- trtexec 风格参数：`src/JYPPX.TensorRtSharp.Tools/Trtexec/TrtexecLikeDeploymentOptions.cs`。
- 构建应用点：`src/JYPPX.TensorRtSharp.Tools/Build/OnnxEngineBuildService.DeploymentConfiguration.cs`。
- copied deployment snapshot：`src/JYPPX.TensorRtSharp.Tools/Build/OnnxEngineBuildService.Diagnostics.cs`。
- 构建报告：`src/JYPPX.TensorRtSharp.Tools/Build/OnnxEngineBuildDiagnostics.cs`。
- TensorRtExec gap list：`applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json`。

## 为什么要 Readback

只记录命令行参数有两个问题。第一，用户可能传入了一个 TensorRT 当前版本不支持的选项，例如 TRT8 的 strongly typed network 或 TRT10/11 的 legacy `--minTiming`；第二，有些选项会被 TensorRT 规范化，例如 tactic source mask、memory pool bytes、DLA core、builder flag 或 timing cache 状态。Readback 的价值是把“请求值”和“TensorRT 返回值”放到同一份报告里。

典型日志形态：

```text
TrtexecMemoryPool Applied=True Name=workspace Pool=Workspace RequestedBytes=1073741824 ReadbackBytes=1073741824 ReadbackMatch=True
TrtexecTiming AverageApplied=True RequestedIterations=8 ReadbackIterations=8 ReadbackMatch=True EvidenceBoundary=builder-config-readback-only
TrtexecDeploymentControl Name=TacticSources Applied=True Requested=CuBlas,CuDnn Readback=CuBlas,CuDnn Default=... ReadbackMatch=True
TrtexecDeploymentControl Name=DirectIO Applied=True Requested=True Readback=True ReadbackMatch=True
```

报告里的核心信号不是“某个开关出现过”，而是 `Requested`、`Readback`、`ReadbackMatch` 和 `EvidenceBoundary` 是否同时存在。这样发布负责人可以快速判断问题是参数解析、版本不支持、TensorRT 拒绝，还是后续 runtime proof 缺失。

## Readback 行级 schema

建议每一个可读回的 builder config 选项都形成一条结构化记录，而不是只写一段自然语言日志：

```text
OptionName
OptionGroup
TensorRtLine
RequestedValue
NormalizedRequestedValue
Applied
ReadbackValue
ReadbackMatch
UnsupportedReason
DiagnosticCode
EvidenceBoundary
CanPromoteRuntimeProof
CanPromotePackageConsumerProof
CanDeleteDeferredRecord
```

这些字段的含义要稳定：`RequestedValue` 是用户原始意图，`NormalizedRequestedValue` 是 parser 和 option layering 处理后的值；`Applied=True` 必须意味着 native setter 已被调用且没有被版本 guard 拒绝；`ReadbackMatch=True` 必须来自对应 getter 或等价的 TensorRT readback，不是 CLI echo。`EvidenceBoundary` 对本类记录应保持 `builder-config-readback-only`。

推荐把失败原因也写成枚举或稳定字符串，例如：

```text
unsupported-on-trt-line
removed-in-trt11
legacy-trt8-only
dependency-probe-only
builder-config-unavailable
setter-rejected
readback-mismatch
parse-only
dry-run-only
```

这样 TensorRtExec、OnnxToEngine、WinForms 和 release reviewer 可以用同一套字段判断问题来自解析、版本、依赖还是 TensorRT readback。

## 已覆盖的配置面

当前公开文章和 gap list 里，Builder Config Readback 覆盖这些高价值构建面：

| 能力 | 对应 trtexec 选项 | TensorRtSharp4.0 证据 | 边界 |
| --- | --- | --- | --- |
| workspace 与 memory pool | `--workspace`、`--memPoolSize` | `TensorRtBuilderConfig.SetMemoryPoolLimit` + `GetMemoryPoolLimit`，日志 `TrtexecMemoryPool` | build/report evidence |
| timing iterations | `--avgTiming`、`--minTiming` | `SetAverageTimingIterations` + `GetAverageTimingIterations`；TRT8 legacy `SetMinTimingIterationsCompatibility` | `builder-config-readback-only` |
| profiling verbosity | `--profilingVerbosity`、`--verbose` | `SetProfilingVerbosity`，构建报告记录 profile intent | 需要真实 enqueue 日志后才能谈 runtime |
| tactic sources | `--tacticSources` | 先读默认 mask，再应用 `+/-` 差量并读回 | 只证明 builder config 接受策略 |
| DLA 与 GPU fallback | `--useDLACore`、`--allowGPUFallback` | `DlaCoreCount` 校验、`SetDefaultDeviceType`、`GetDlaCore`、`GpuFallback` flag readback | DLA 模型可运行性仍需单独 proof |
| DirectIO 与 sparsity | `--directIO`、`--sparsity` | builder flag readback；`sparsity=force` 保持 parse/report-only | 不重写模型权重 |
| strongly typed | `--stronglyTyped` | TRT10 creation bit、TRT11 always-strongly-typed contract、TRT8 version guard | 构建语义证据，不是输出正确性 |
| engine packaging | `--versionCompatible`、`--excludeLeanRuntime`、`--stripWeights`、`--refit` | flag readback 与 TensorRT line guard | lean runtime 和公开包 proof 另算 |
| timing cache | `--timingCacheFile`、`--exportTimingCache` | typed timing-cache owner、输入/输出 size、SHA256 | build-cache evidence only |
| scalar controls | `--maxNbTactics`、`--tilingOptimizationLevel`、`--l2LimitForTiling`、`--quantizationFlags` | 支持版本 apply/readback，不支持版本写 controlled diagnostic | 不删除 unsupported guard |

这些字段让报告能解释“为什么选项可用或不可用”。例如 `--minTiming` 在 TRT8 可通过 legacy compatibility API 读回；在 TRT10/11 里则应明确写成不应用，并说明应使用 average timing iterations。这样的差异比默默忽略选项更适合外部用户排障。

## Report promotion flags

Builder config readback 报告应显式带上不能晋级 runtime proof 的布尔字段：

```text
BuilderConfigReadbackEvidence = true
BuilderConfigCreatedEngine = false
BuilderConfigRanInference = false
BuilderConfigValidatedOutputs = false
BuilderConfigIsRealModelRuntimeProof = false
BuilderConfigIsPackageConsumerRuntimeProof = false
BuilderConfigCanPromoteRuntimeProof = false
BuilderConfigCanPromoteReleaseProof = false
```

如果同一个 run 后续又执行了 bounded runtime 或真实模型 runner，应让 runtime section 单独写自己的 `InferenceRan`、`OutputMatch`、stdout/stderr hash 和 validator 结果。不要让 build readback section 继承 runtime section 的 proof 状态。

`ReadbackMatch=True` 是构建配置证据，不是模型输出证据。`BuildEvidenceOnly=True`、`ProofClassification=build-only` 或 `ProofClassification=builder-config-readback-only` 应继续阻止 release close 自动通过。

## OnnxToEngine 使用示例

下面的命令适合做本地构建参数检查。路径请放在 E 盘或项目工作区，不要把大模型、engine、timing cache 写到 C 盘临时目录。

```powershell
dotnet run --project .\samples\OnnxToEngine\OnnxToEngine.csproj -- `
  --onnx E:\TensorRtSharpAssets\models\model.onnx `
  --saveEngine E:\TensorRtSharpAssets\engines\model.plan `
  --workspace 2048 `
  --memPoolSize workspace:2048,tacticDRAM:1024 `
  --avgTiming 8 `
  --tacticSources +cublas,+cudnn `
  --profilingVerbosity detailed `
  --exportReport E:\TensorRtSharpAssets\reports\model-build-report.json `
  --buildOnly
```

重点检查报告中的这些字段或日志行：

- `TrtexecLike BuildOnly=True`：说明本次只做构建，不声明 runtime proof。
- `TrtexecMemoryPool ... ReadbackMatch=True`：说明 memory pool 请求值与 TensorRT 读回值一致。
- `TrtexecTiming ... EvidenceBoundary=builder-config-readback-only`：说明 timing 读回只作为 builder config evidence。
- `TrtexecDeploymentControl Name=TacticSources ... ReadbackMatch=True`：说明 tactic source mask 已按默认值和用户差量计算后读回。
- `OnnxToEngine Passed=True`：说明构建流程完成，但仍需要看 `state` 是否为 build-only、dependency-probe-only、runtime-output-unverified 或真实模型 proof。

如果报告 state 是 `dependency-probe-only`，那说明 native bridge、CUDA、TensorRT 或 cuDNN 依赖不可用。本状态不能被包装成 build 成功或 runtime proof，应回到 DLL 排障文章处理环境。

## TensorRtExec 对齐 trtexec

TensorRtExec 的目标是复刻官方 trtexec 的核心构建和运行体验，同时提供 CLI 和 WinForms 两种入口。Builder Config Readback 是这个目标的关键底座，因为它把 GUI/CLI 的 option layering 固定为：

```text
CLI/WinForms controls
  -> TrtexecLikeParser
  -> TrtexecLikeDeploymentOptions
  -> OnnxEngineBuildOptions
  -> TensorRtBuilderConfig
  -> build log / report readback
```

这条链路能帮助用户定位参数在哪一层失效：

- UI 或 CLI 没生成参数：检查 TensorRtExec command preview 和 WinForms field map。
- parser 拒绝：检查 `TrtexecLikeParser` 的语法错误，例如 memory pool token 或 tactic source token。
- TensorRT line 不支持：检查 version guard 诊断，例如 TRT8 unsupported、TRT11 removed setter。
- TensorRT 接受但读回不一致：这是构建配置问题，应阻断 release candidate。
- 构建成功但没有输出正确性：进入 runtime smoke 或模型任务验证，不在 readback 阶段硬说完成。

## 跨版本边界

Builder config 是跨版本差异最密集的区域之一，文章和测试必须保留这些边界：

- TRT8 保留 `MaxWorkspaceSizeCompatibilityInBytes` 和 `MinTimingIterationsCompatibility`，但跨版本推荐走 memory pool 与 average timing。
- TRT10/11 不应伪造 legacy min timing setter；应记录 `MinimumApplied=False` 和原因。
- TRT11 的 strongly typed 语义不同，应写成 always-strongly-typed contract，而不是复用 TRT10 creation bit。
- `QuantizationFlags` 在 TRT11 被移除，不能为了文章好看把 unsupported 诊断改成 applied。
- progress monitor、calibrator、algorithm selector 这类 callback/borrowed pointer API 不属于低风险 readback，不能用 presence probe 替代生命周期设计。
- plugin load/register/deregister 仍是高风险边界；serialized plugin path snapshot 只能作为 copied readonly inventory。

建议把跨版本失败写成明确分类：

| 分类 | 例子 | 报告建议 |
| --- | --- | --- |
| `trt8-legacy-compatibility` | `MaxWorkspaceSizeCompatibilityInBytes`、`MinTimingIterationsCompatibility` | 可以 apply/readback，但标记 legacy。 |
| `trt10-typed-setter` | memory pool、average timing、tactic sources | setter/readback 都要匹配。 |
| `trt11-removed-setter` | legacy precision constraints、quantization flags | parse/report-only 或 rejected diagnostic。 |
| `always-strongly-typed` | TRT11 network typing | 不复用 TRT10 creation bit。 |
| `callback-lifecycle-required` | calibrator、algorithm selector、progress monitor | presence probe 不能替代 owner lifecycle。 |

这张表能避免把 unsupported 选项写成“暂未测试”，也避免把版本差异伪装成成功 readback。

## proof 边界

Builder Config Readback 是配置确认，不是 runtime proof。以下材料都不能作为 package-consumer-runtime proof：

- build-only report。
- parse-only report。
- dependency-probe-only report。
- timing cache artifact。
- TensorRtExec GUI screenshot。
- command preview。
- load-engine readonly diagnostics。
- local feed package consumer。
- ProjectReference consumer。
- direct `.nupkg` install。
- `ReadbackMatch=True` 的 builder config 行。
- `BuilderConfigReadbackEvidence=true`。
- timing cache SHA256。
- tactic source mask readback。
- profiling verbosity readback。

要证明公开包可用，仍需外部 clean consumer 使用公开包源，执行 restore/build/runtime smoke，记录 stdout/stderr/merged transcript/validator output SHA256、host runtime metadata、package identity 和 owner review。Builder readback 可以作为 proof record 的附属构建证据，但不能单独晋级。

## 配图建议

- 一张 option layering 图：CLI/WinForms controls -> `TrtexecLikeDeploymentOptions` -> `TensorRtBuilderConfig` -> report readback。
- 一张配置读回表截图，突出 `Requested`、`Readback`、`ReadbackMatch`、`EvidenceBoundary`。
- 一张跨版本差异图，对比 TRT8 legacy compatibility、TRT10 builder flags、TRT11 removed/changed APIs。
- 一张证据边界图，把 build/report evidence、runtime-output-unverified、real-model runtime proof、package-consumer-runtime proof 分成四层。

## 下一步

继续增加安全的 readback 字段和报告一致性测试，优先覆盖只读、查询型、部署关键型 API。对 callback、裸指针、外部资源、plugin lifecycle 和 borrowed pointer API，继续走 owner-safe 设计门禁，不为了文章完整度删除 deferred 边界。真实发布 proof 仍交给 clean external consumer runtime smoke、owner input validator 和 post-publish verification。
