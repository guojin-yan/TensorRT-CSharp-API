# Deferred 边界：为什么 manifest 100% 不等于可发布 100%

TensorRtSharp4.0 已经把大量 CUDA / TensorRT 接口纳入 manifest、native source、generated binding 和 C# wrapper 的覆盖体系。这个阶段很容易产生一个误解：只要 manifest/source 匹配、占位 entrypoint 存在、测试没有报 missing，就可以宣布 API 全部可用。实际不是这样。

对 TensorRT/CUDA 这种跨语言、跨版本、强 ownership 的 API 来说，真实完成度必须同时看四件事：native ABI 是否是非 deferred 实现，高层 C# wrapper 是否给出有语义的 typed/copied/owning surface，smoke 或质量门是否覆盖行为边界，以及涉及 runtime 或发布时是否有 package-consumer-runtime proof。manifest/source 100% 只能证明“仓库知道这个接口”，不能证明“用户可以安全调用这个接口”。

## 适合

- 想理解项目从 missing 清零转向 deferred 边界提升的读者。
- 需要审查 `artifacts/interface-coverage/project-completion-review.md` 的维护者。
- 负责决定某个 API 是否能进入 public wrapper 的发布负责人。
- 准备贡献 TensorRT/CUDA binding，但不确定哪些接口能批量提升、哪些必须先做 owner-safe 设计门的人。

## 关键路径

- 总体复审：`artifacts/interface-coverage/project-completion-review.md`。
- 接口矩阵：`artifacts/interface-coverage/tensorrt-interface-comparison.csv`。
- 风险门：`docs/articles/zh-cn/deferred-boundary-risk-tier-gate.md`。
- 设计分组：`docs/articles/zh-cn/deferred-manual-design-groups.md`。
- runtime 反序列化审计：`artifacts/interface-coverage/runtime-deserialization-deferred-boundary-audit.md`。
- runtime 反序列化 JSON 证据：`artifacts/interface-coverage/runtime-deserialization-deferred-boundary-audit.json`。

## 完成度分层

推荐把一个 API 的完成度拆成下面这条 evidence ladder：

```text
manifest/source match
  -> non-deferred native bridge
  -> generated interop/header parity
  -> typed C# wrapper
  -> quality test / compile smoke
  -> runtime smoke
  -> package-consumer-runtime proof
```

每一层都回答不同问题：

| 层级 | 能证明什么 | 不能证明什么 |
| --- | --- | --- |
| manifest/source match | 仓库已追踪接口，version guard 和 source token 可审计 | native entrypoint 已安全实现 |
| non-deferred native bridge | C ABI 有真实参数和返回语义 | C# 用户能正确管理生命周期 |
| typed C# wrapper | 用户不用直接面对裸 `IntPtr` / `nint` | runtime 已在真实模型上通过 |
| quality test / compile smoke | wrapper、文档、manifest、报告字段可回归 | TensorRT 已执行推理 |
| runtime smoke | 当前源码树或本地包在某台兼容主机上执行过 | 公开包已被外部 clean consumer 使用 |
| package-consumer-runtime proof | 公开包源、clean consumer、host metadata、日志 hash 和 validator 闭环 | 自动授权 GitHub/NuGet 发布 |

这也是为什么文章、README、dashboard、precheck 和 candidate inventory 只能作为说明或候选证据，不能替代真实 proof。

## 风险层级

`docs/articles/zh-cn/deferred-boundary-risk-tier-gate.md` 把剩余 deferred 分成 A/B/C/D 四层：

| 层级 | 名称 | 能否直接批量提升 | 典型形态 |
| --- | --- | --- | --- |
| A | `A-tier copied value` | 可以 | count、exists、scalar getter、caller buffer string、copied metadata、immutable snapshot |
| B | `B-tier safe alternative` | 条件允许 | 已有 safe alternative 或 alias wrapper，可做 source-quality proof，但不能删除 deferred history |
| C | `C-tier design-gate-required` | 不可以 | 需要 owner/lifetime/callback/no-throw/pointer-free snapshot 的设计门 |
| D | `D-tier keep-deferred` | 不可以 | plugin instance、allocator callback、enqueue、borrowed pointer、register/deregister、raw ownership API |

当前默认低风险 deferred 候选已经为 `0`。这意味着后续不能再靠“看起来像 getter”机械提升，必须先确认它属于 copied value，还是来自 callback 生命周期、borrowed TensorRT object、external resource 或 runtime ownership 边界。

## 什么可以优先提升

可优先推进的是只读、可复制、无 ownership 争议的 API。它们通常具有这些特征：

- 返回值是 scalar、enum、bool、count 或固定结构。
- 字符串通过 count/copy 或 caller buffer 复制，不返回 TensorRT 内部 char pointer。
- 数组和 metadata 在 native call 内复制到 caller-owned buffer。
- C# wrapper 返回 immutable snapshot、typed value 或 owning wrapper。
- TRT8、TRT10、TRT11 的 version guard 不互相伪装。
- 失败通过状态码、诊断对象或 managed exception 边界处理，不跨 ABI 抛异常。

例如 B-tier proof 批次中，`GetAverageTimingIterations`、`MaxWorkspaceSizeCompatibilityInBytes`、`GetTilingOptimizationLevel`、ONNX parser copied error diagnostics、ParserRefitter copied diagnostics 这类能力，都适合用 wrapper/docs/quality proof 压实。但即使这些 proof 通过，也不代表可以删除 deferred history，更不代表 runtime proof 或公开发布 proof 已完成。

## Uplift 作业单

每一批 deferred uplift 应先从机器可读清单出发，而不是凭直觉判断“像 getter 就能做”。当前推荐入口是：

```text
artifacts/interface-coverage/deferred-readonly-candidate-list.json
artifacts/interface-coverage/deferred-candidate-safety-triage.json
artifacts/interface-coverage/tensorrt-interface-comparison.csv
artifacts/interface-coverage/tensorrt-interface-coverage.json
```

一条候选从 planning 进入 implemented-with-deferred-history，至少要留下这些证据：

```text
candidateId
apiArea
riskLevel
outputMode
nativeLayerRequired
managedWrapperRequired
smokeRequired
implementationStatus
nativeSources
managedSources
smokeSources
qualityTests
publicSurface
ownershipBoundary
DeferredRowsStillRequired
CanDeleteDeferredRecord=false
```

`implemented-with-deferred-history` 是一个很重要的状态：它说明已有安全替代面或 copied wrapper 可以使用，但旧 deferred row 仍作为边界记录保留。删除 deferred row 只能在真实 API、wrapper、smoke、release proof 都覆盖到对应风险后再讨论，不能用来让矩阵看起来更满。

## 安全正例

下列类型适合作为后续低风险批次的写法参考：

```text
TensorRtPluginRegistryInventory
TensorRtPluginCreatorSummary
TensorRtPluginFieldInfo
TensorRtEngineInspector
TensorRtOnnxParserDiagnosticSnapshot
TensorRtOnnxParserRefitterDiagnosticSnapshot
TensorRtBuilderConfigReadback
CudaDeviceInfo
CudaMemoryInfo
CudaDeviceGraphMemoryInfo
CudaDeviceGraphMemorySummary
OnnxEngineParserPreflightSnapshot
```

它们的共同点是：返回 copied scalar、caller-buffer string、immutable record、summary 或 pointer-free snapshot；C# public surface 不暴露 `IntPtr`、`nint`、native function pointer、device pointer、borrowed TensorRT object 或 `SafeHandle`。报告字段也应保留非 proof 口径，例如：

```text
RuntimeEvidenceKind=copied-readonly-summary
PointerFreeCopiedSummary=true
IsRuntimeExecutionProof=false
IsPackageConsumerRuntimeProof=false
CanPromoteRuntimeProof=false
CanPromoteReleaseProof=false
CanDeleteDeferredRecord=false
```

这类正例能提升用户可诊断性，也能减少直接 P/Invoke 的风险；但它们仍不是 enqueue、plugin lifecycle、allocator callback 或 package-consumer proof。

## 必交质量门

每批至少应按影响范围选择质量门，而不是只跑一个 broad build：

```text
ReadonlyDiagnosticsCandidateImplementationEvidenceTests
ReadonlySummaryEvidenceMatrixTests
PublicApiHandleExposureAuditTests
TensorRtNativeAbiSurfaceParityTests
PluginRegistryInventoryTests
RefitterEngineInspectorDiagnosticsTests
RuntimeDeserializationBoundaryPrecheckTests
CallbackAllocatorBoundaryTests
AllocatorInterfaceInfoDesignGateTests
AlgorithmSnapshotDesignGateTests
BuilderConfigScalarControlsTests
PublishingPublicArticleTests
```

如果改动触及 manifest/native/generated/wrapper，应跑 ABI 和 public handle 门；如果只改文章，也至少要跑对应 public article gate。quality test 通过只能证明本层行为被固定，不会自动升级为 runtime proof。

## 什么必须继续 Deferred

以下接口即使名字像 getter，也不能被当成低风险提升：

- algorithm selector borrowed objects：`IAlgorithm`、`IAlgorithmContext`、`IAlgorithmIOInfo`、`IAlgorithmVariant`。
- callback/allocator：`IGpuAllocator`、`IGpuAsyncAllocator`、`IOutputAllocator`、`IDebugListener`。
- plugin lifecycle：`registerCreator`、`deregisterCreator`、`loadLibrary`、plugin resource acquire/release、plugin instance create/clone/enqueue。
- execution boundary：`execute`、`executeV2`、`enqueueV2`、binding buffer、stream、profile、device memory 组合。
- expression/model builder：`IDimensionExpr`、`IExprBuilder` 这类 TensorRT 内部对象。
- runtime deserialization ownership：`deserializeCudaEngineV2`、`loadRuntime`、stream reader/writer callback、external lean runtime。
- graph/external resource ownership：CUDA external semaphore、surface/texture object、IPC handle、library/kernel handle 等只能先做 copied token/metadata，不开放 borrowed resource owner。
- callback proof attempt：debug listener、output allocator、profiler、progress monitor 的 runtime proof attempt preflight 不能代替真实 callback invocation。

这些接口共同的问题不是“有没有参数”，而是对象生命周期、回调异常、device pointer ownership、borrowed pointer 是否逃逸、native owner 是否可释放、plugin/library provenance 是否可信。没有这些设计门，public API 暴露得越早，后续 ABI 和用户代码越难收回。

## Runtime Deserialization 示例

`runtime-deserialization-deferred-boundary-audit` 是一个典型案例。当前普通 `IRuntime::deserializeCudaEngine` 已通过 scoped-buffer bridge 和 managed owner wrapper 覆盖，用户可以使用：

```text
TensorRtRuntime.Deserialize(byte[])
TensorRtRuntime.Deserialize(ArraySegment<byte>)
TensorRtRuntime.Deserialize(ReadOnlySpan<byte>)
TensorRtRuntime.Deserialize(Stream)
TensorRtRuntime.Deserialize(File)
TensorRtRuntime.Deserialize(HostMemory)
```

这些路径的安全点是 serialized buffer 在进入 native 前完成 copy、pin 或 scoped ownership，返回的 engine 由 managed wrapper 持有，不把 `ICudaEngine*` 暴露给用户。

但下面 5 条仍保持 `deferred-only`：

| Interface | TRT line | Manifest | Decision |
| --- | --- | --- | --- |
| `IRuntime::deserializeCudaEngineV2` | TRT10 | `trt10-runtime-deserialize-cuda-engine-v2-deferred` | keep-deferred |
| `IRuntime::deserializeCudaEngineV2` | TRT11 | `trt11-runtime-deserialize-cuda-engine-v2-deferred` | keep-deferred |
| `IRuntime::loadRuntime` | TRT8 | `trt8-runtime-load-runtime-deferred` | keep-deferred |
| `IRuntime::loadRuntime` | TRT10 | `trt10-runtime-load-runtime-deferred` | keep-deferred |
| `IRuntime::loadRuntime` | TRT11 | `trt11-runtime-load-runtime-deferred` | keep-deferred |

原因很具体：`deserializeCudaEngineV2` 引入 `IStreamReader` / `IStreamReaderV2` callback state、seek state、device-aware reads 和 returned engine ownership；`loadRuntime` 引入 external lean runtime、plugin host code、returned runtime owner、library path trust 和 provenance。它们不是 scalar getter，也不是 count/copy snapshot。

当前可用的安全替代面是：

```text
TensorRtRuntimeDeserializationBoundaryPrecheck.EvaluateKnownSurface
TensorRtRuntimeDeserializationDependencyDiagnostics.EvaluateKnownSurface
TensorRtStreamIoInterfaceInfoDesignGate.EvaluateKnownSurface
```

这些 surface 都是 pointer-free diagnostics/design gate，不暴露 `IRuntime*`、`ICudaEngine*`、`IStreamReader*`、`IStreamReaderV2*`、`IStreamWriter*`、`IntPtr`、`nint` 或 `SafeHandle`。因此它们能帮助用户判断边界状态，但不能把 direct `deserializeCudaEngineV2/loadRuntime` 宣布为已实现。

## proof 边界

以下状态都不能单独证明 API 可发布：

- manifest/source 匹配。
- no-arg deferred stub 存在。
- generated P/Invoke 存在。
- README、文章或 dashboard 标记 ready。
- build-only。
- dry-run。
- parse-only。
- copied-state。
- schema-only。
- preflight。
- dependency-probe-only。
- owner-action-required。
- local feed package consumer。
- ProjectReference consumer。
- direct `.nupkg` install。

Public API 必须避免暴露裸 `IntPtr` / `nint` / borrowed object。字符串、数组和 metadata 应通过 count/copy、caller buffer 或 immutable snapshot 方式返回。callback、allocator、plugin、external resource 和 execution path 在真实 owner/lifetime/no-throw/runtime proof 完成前，应继续保留 deferred history。

## 贡献者检查清单

准备提升一个 deferred API 前，至少逐项回答：

- 这个接口是否返回或接收 TensorRT-owned borrowed pointer。
- 返回对象是否需要 C# owning wrapper，还是只能复制成 snapshot。
- native C ABI 是否有真实参数，不是 no-arg placeholder。
- TRT8、TRT10、TRT11 是否各自有正确 version guard。
- public header、native source、manifest、generated binding 和 C# wrapper 是否一致。
- 是否会跨 ABI 抛异常。
- 是否涉及 callback、allocator、stream、device memory、plugin registry 或 external library。
- 是否已有 quality test、smoke 或 package-consumer proof 覆盖对应层级。
- 是否保留 deferred history，避免用删除记录制造完成度。
- 是否给出 `RuntimeEvidenceKind`、`CanPromoteRuntimeProof`、`CanPromoteReleaseProof` 和 `CanDeleteDeferredRecord` 这类 report 字段。
- 是否在文章、README、report、dashboard 中明确 forbidden substitutes。

只要其中任一项无法回答，就应该先写 design gate 或 pointer-free diagnostics，而不是直接开放 public API。

## 配图建议

- 一张梯子图：manifest/source match -> non-deferred bridge -> C# wrapper -> smoke -> package-consumer-runtime proof。
- 一张 A/B/C/D 风险分层图，把 copied value、safe alternative、design-gate-required、keep-deferred 分开。
- 一张 runtime deserialization 案例图：scoped-buffer `Deserialize` 安全路径与 `deserializeCudaEngineV2/loadRuntime` deferred 路径并排。
- 一张高风险边界图：callback trampoline、borrowed pointer、plugin instance create/enqueue、allocator owner、external lean runtime。

## 下一步

继续优先推进只读、可复制、无 ownership 争议的 API；对 allocator、debug listener、Plugin V2/V3 callback、borrowed pointer、external resource、runtime deserialization ownership 和 execution/enqueue 相关接口，继续走 owner-safe 设计门、native lifetime gate、no-throw callback boundary、pointer-free snapshot 和真实 callback/runtime proof。不要删除 deferred manifest 来制造完成度，也不要把 build-only、dependency-probe-only 或 local package consumer 当成 package-consumer-runtime proof。
