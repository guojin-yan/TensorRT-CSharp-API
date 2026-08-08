# Deferred 人工设计分组指南

TensorRtSharp4.0 当前已经没有默认低 ownership 风险的 deferred 候选。`Export-DeferredReadOnlyApiCandidatePlan.ps1` 在默认模式下不会再给出可以机械提升的接口；只有显式传入 `-IncludeMediumRisk` 时，才会列出需要人工设计的候选。

这篇文档说明这些候选为什么不是“直接实现清单”，以及下一阶段如何把它们拆成可验证的设计门。

配套质量门见 `deferred-boundary-risk-tier-gate.md`。该质量门使用 `deferred-boundary-risk-tier-gate`、`manifest-source-match-not-release-proof`、`no-public-raw-pointer` 与 `package-consumer-smoke-required` 标记，把剩余 deferred 分成 A/B/C/D 四层，明确 Algorithm selector borrowed object、Plugin V2/V3 callback、Allocator/OutputAllocator callback、Execution/NoCopy/Deserialize 等接口在安全 handle、pointer-free snapshot 或 runtime proof 完成前不能被误判为“已可用”。

## 当前分组

最新候选计划来自：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-DeferredReadOnlyApiCandidatePlan.ps1 -IncludeMediumRisk -MaxItems 60
```

输出文件：

- `artifacts/interface-coverage/deferred-readonly-api-candidate-plan.json`
- `artifacts/interface-coverage/deferred-readonly-api-candidate-plan.md`

当前汇总：

| 指标 | 数量 |
| --- | ---: |
| total deferred rows | 944 |
| low-risk deferred rows | 0 |
| medium-risk deferred rows | 148 |
| high-risk deferred rows | 796 |
| manual review design groups | 10 |

> `manualReviewDesignGroups` 现在同时汇总 medium 与 high ownership risk。默认候选仍只允许低风险行；高风险组只是边界规划输入，不能作为直接实现清单。

## 十个设计组

| 设计组 | 下一步 |
| --- | --- |
| `runtime-execution-boundary` | 不作为 deferred cleanup 直接提升。先设计 binding buffer ownership、stream/profile state、exception-to-status mapping 和真实 runtime smoke。 |
| `runtime-deserialization-boundary` | 已进入 `runtime-deserialization-boundary-precheck` 和 `runtime-deserialization-dependency-diagnostics`：当前 managed `Deserialize(byte[]/ArraySegment/ReadOnlySpan/Stream/File/HostMemory)` 边界已确认 buffer copy、scoped pin 和 engine wrapper ownership，并能区分 dependency-probe-only、`blocked-by-cuda-driver` 与 package-consumer runtime proof，但 direct `deserializeCudaEngineV2/loadRuntime` 继续 deferred，等待 plugin/library dependency diagnostics 完整化、`loadRuntime` ownership 和 package-consumer runtime proof。详见 `runtime-deserialization-boundary-precheck.md` 与 `runtime-deserialization-dependency-diagnostics.md`。 |
| `dimension-expression-snapshot-design` | 已进入 design gate：只能通过 pointer-free snapshot 推进，并且必须绑定到已知 owner object。不能暴露 borrowed `IDimensionExpr` 指针。详见 `dimension-expression-snapshot-design-gate.md`。 |
| `expression-builder-design-gate` | 继续保持 design gate。表达式节点的 ownership/lifetime 未建模前，不通过 public API create expression nodes，`ExprBuilderCreationEnabled=False`。 |
| `algorithm-selector-ownership-boundary` | 继续保持 ownership boundary。`IAlgorithm` / `IAlgorithmContext` / `IAlgorithmIOInfo` / `IAlgorithmVariant` 来自 TensorRT algorithm selector callback 的短生命周期 borrowed 结果；selector ownership、callback result lifetime 和 copied snapshot 模型没有完整设计前，不暴露 borrowed algorithm pointer，也不把 `getTimingMSec/getWorkspaceSize/getTactic/getImplementation` 机械提升为 public API。 |
| `error-recorder-diagnostics-design` | 已进入 design gate：优先 copied diagnostics、presence、snapshot。不能暴露 recorder pointer，也不能把 ref-count API 做成 public ownership 控制。详见 `error-recorder-diagnostics-design-gate.md`。 |
| `calibrator-callback-metadata-design` | 已进入 `calibrator-metadata-design-gate`：限定为 presence 和 copied metadata 规划。不能调用 calibration callback，不能接管 batch/cache buffer。详见 `calibrator-metadata-design-gate.md`。 |
| `plugin-ownership-boundary` | 只能使用 count/copy 或 copied metadata。不能暴露 borrowed plugin creator/resource/registry。已完成 registry copied metadata inventory；plugin instance/resource/library/callback 继续 deferred。详见 `plugin-ownership-boundary.md`。 |
| `callback-allocator-boundary` | 需要 owner ledger、no-throw callback boundary 和 runtime proof 后才能进入真实提升。 |

`manual-review-other` 中的 TensorRT 8 RNNv2 borrowed-state 子批次已单独收口：起始 12 条 triage 行中，2 条 `IRNNv2Layer::getDataLength` 已提升为 copied scalar，10 条 borrowed tensor/weights 行已通过 owner-bound/copy-out 安全封装提升；旧 deferred history 继续保留，真实 RNNv2 runtime proof 仍未完成。详见 `rnnv2-borrowed-state-design-gate.md`。

## 为什么不能直接提升

中风险/高风险候选里包含 `IExecutionContext::execute/executeV2`、`IRuntime::deserializeCudaEngine/loadRuntime`、`IAlgorithm::getTimingMSec/getWorkspaceSize`、`IAlgorithmVariant::getTactic/getImplementation`、`IInt8Calibrator::getBatch`、`IDimensionExpr`、`IExprBuilder`、`IErrorRecorder` 等接口。它们的共同问题不是参数个数，而是生命周期和语义边界：

- 执行类 API 涉及 binding buffer、stream、profile、device memory 和 runtime error mapping。
- 反序列化类 API 涉及 serialized buffer ownership、engine lifetime、plugin/library dependency 和 package layout。
- algorithm selector API 涉及 TensorRT callback 期间传入的 borrowed `IAlgorithm*` / context / IO info / variant 对象；这些对象不能脱离 callback result lifetime 暴露给 C# 用户。
- expression API 往往返回 TensorRT 内部对象，不能作为裸指针给 C# 用户。
- calibrator callback 会进入用户 buffer 和 TensorRT callback 生命周期。
- error recorder ref-count 不是普通只读属性，不能让 C# public API 直接控制 native ownership。

这些接口需要先有 pointer-free object model 或 design gate，再进入 native ABI 实现。

## 推荐推进顺序

1. 先做 `error-recorder-diagnostics-design`：
   - 输出 copied diagnostic snapshot。
   - 保持 ref-count API 不公开为 ownership 控制。
   - 用 quality tests 锁定 `IsRuntimeExecutionEvidence=False`、`DevicePointerExposed=False` 之类边界字段。
2. 再做 `dimension-expression-snapshot-design`：
   - 已有 `dimension-expression-snapshot-design-gate`，记录 `SnapshotTypeReady=True`、`OwnerLifetimeKnown=False`、`ExprBuilderCreationEnabled=False` 和 `RuntimeProofBlocked=True`。
   - 后续只在已知 owner object 下复制 `isConstant/getConstantValue/isSizeTensor`。
   - 不暴露 `IDimensionExpr*`。
   - 不允许 public API 持有 borrowed pointer。
3. 再推进 `runtime-deserialization-boundary`：
   - 已有 `runtime-deserialization-boundary-precheck`，记录 `ManagedByteArrayDeserializeReady=True`、`ManagedStreamDeserializeReady=True`、`HostMemoryDeserializeReady=True`、`SerializedBufferCopiedBeforeInterop=True`、`EngineHandleOwnedByWrapper=True`、`LoadRuntimeDeferred=True` 和 `RuntimeProofBlocked=True`。
   - 已有 `runtime-deserialization-dependency-diagnostics`，记录 `RuntimeEvidenceKind=dependency-diagnostics`、`FullPackageConsumerReportPresent`、`FullPackageConsumerSmokeRequested`、`DependencyProbeOnly`、`BlockedByCudaDriver`、`PluginLibraryDependencyDiagnosticsComplete=False`、`LoadRuntimeOwnershipModeled=False` 和 `CanPromoteRuntimeProof=False`。
   - 下一步补完整 plugin/library dependency diagnostics、真实 serialized engine package-consumer smoke 和 `loadRuntime` returned runtime ownership 模型。
   - 等 package-consumer runtime proof 可用后，再讨论 direct `deserializeCudaEngineV2/loadRuntime` 行的真实提升。

## 不允许

- 不允许删除 deferred 记录制造完成度。
- 不允许把 `-IncludeMediumRisk` 的输出当成低风险提升清单。
- 不允许 public API 暴露裸 `IntPtr`、borrowed TensorRT pointer、plugin creator pointer 或 expression pointer。
- 不允许跨 ABI 抛异常。
- 不允许只改 manifest，不改 native/source/wrapper/tests。

## B-tier 前 12 项 proof 批量收口

`artifacts/interface-coverage/deferred-btier-implementation-work-package.json` 的前 12 个工作项已作为第一批 proof 收口对象固定到 `DeferredBTierWorkItemProofClosureLedgerTests`。这批工作不是删除 deferred history，也不是 runtime proof、package-consumer proof 或 release proof，而是把已有 safe alternative / alias 证据链压实到 manifest、native source、C# wrapper、ProjectQuality test 和本文档。

本批 proof 的核心边界是：不能删除 deferred history，不能把 alias/safe alternative 证明解释为 100% runtime 可用，也不能把它作为发布许可。

| ID | 接口 | 版本 | 收口边界 |
| --- | --- | --- | --- |
| `btier-001` | `IExecutionContext::getName` | TRT10 | 使用 `trt10-execution-context-get-name` 的 count/copy 字符串 wrapper 证明可用；`trt10-algorithm-context-get-name-deferred` 继续保留为 algorithm-context deferred history。 |
| `btier-002` | `IProfiler::getInterfaceInfo` | TRT11 | 使用 TRT11 profiler interface-info copied metadata 证明 callback interface 查询路径；不提升 profiler callback trampoline。 |
| `btier-003` | `ICudaEngine::getProfileShape` | TRT8 | 使用 `trt8-engine-get-profile-shape` 与 `trt8-cuda-engine-get-profile-shape-values` 证明 copied shape/value wrapper；不能删除 `getProfileShapeValues` deferred history。 |
| `btier-004` | `IExecutionContext::getName` | TRT8 | 使用 `trt8-execution-context-get-name` 的 count/copy 字符串 wrapper 证明可用；algorithm-context name deferred history 继续保留。 |
| `btier-005` | `ILayer::getInput` | TRT8 | 使用 layer input count/get input 和 RNNv2 input-mode safe alternative；不暴露 borrowed tensor/plugin pointer。 |
| `btier-006` | `IBuilder::getMaxDLABatchSize` | TRT10 | 使用 builder scalar control native/source 证据和现有 ProjectQuality proof；deferred history 保留为审计记录。 |
| `btier-007` | `IBuilder::getMaxThreads` | TRT10 | 使用 `TensorRtBuilder.MaxThreads` 与 native scalar getter 证明 wrapper 路径；不把 TRT11-only boundary guard 弱化。 |
| `btier-008` | `IBuilder::isNetworkSupported` | TRT10 | 使用 managed `IsNetworkSupported` 和 native builder/config/network 三对象入参证明真实 ABI；不跨 ABI 抛异常。 |
| `btier-009` | `IBuilderConfig::canRunOnDLA` | TRT10 | 使用 config/layer safe query 证明只读结果；不暴露 layer borrowed pointer。 |
| `btier-010` | `IBuilderConfig::getAvgTimingIterations` | TRT10 | 使用 portable `GetAverageTimingIterations` wrapper 覆盖 official getter alias；旧 deferred 名称继续作为历史别名。 |
| `btier-011` | `IBuilderConfig::getDefaultDeviceType` | TRT10 | 使用 default device type getter/setter safe alternative 证明 wrapper；不删除 getter/setter deferred history。 |
| `btier-012` | `IBuilderConfig::getDeviceType` | TRT10 | 使用 layer device type get/is-set/reset/set 组合证明 safe alternative；仍按 wrapper/docs/quality proof 处理，不视为 runtime proof。 |

当前 alias-proof-ready 候选已完整展开为 `btier-001` 到 `btier-051`，并由 `deferred-btier-work-item-proof-closure-ledger.json` 标记为 `source-quality-proof-closed`。工作包采用 `stable-v1-existing-40-then-deterministic-append` 排序策略，既有 40 项编号不会因 dashboard 扩容而漂移，新候选只追加到末尾。后续批量推进不得再次选择这 51 项；必须先确认 `remainingWorkItemCount`，再从新 candidate audit 或独立 runtime/model gap 中选择任务。safe alternative manifest、deferred history、public pointer guard 和 wrapper/doc/test 证据仍须持续回归。

## B-tier 后 12 项 proof 批量收口

`btier-013` 到 `btier-024` 已并入同一个 `DeferredBTierWorkItemProofClosureLedgerTests` 门禁，覆盖 builder config scalar getter、TRT11 parser copied diagnostics、ParserRefitter copied diagnostics 以及 TRT8 builder compatibility getter。该批仍然只是 wrapper/docs/quality proof：不能删除 deferred history，不能替代 runtime proof，也不能作为 package-consumer 或 release 许可。

| ID | 接口 | 版本 | 收口边界 |
| --- | --- | --- | --- |
| `btier-013` | `IBuilderConfig::getDLACore` | TRT10 | 使用 `jyppx-trt10-builder-config-get-dla-core` 的 scalar getter wrapper 证明可追踪；`trt10-builder-config-get-dla-core-deferred` 继续保留。 |
| `btier-014` | `IBuilderConfig::getL2LimitForTiling` | TRT10 | 使用 `GetL2LimitForTiling` 与 native `getL2LimitForTiling()` 证明 safe getter；不扩展 tiling runtime proof。 |
| `btier-015` | `IBuilderConfig::getMaxNbTactics` | TRT10 | 使用 managed `GetMaxTactics` 对应 native `getMaxNbTactics` scalar getter/setter 证据；不删除 `getMaxNbTactics` deferred history。 |
| `btier-016` | `IBuilderConfig::getQuantizationFlag` | TRT10 | 使用 `GetQuantizationFlag` 与 `GetQuantizationFlags` 组合证明 quantization flag safe alternative；两个 deferred history 均保留。 |
| `btier-017` | `IBuilderConfig::getQuantizationFlags` | TRT10 | 使用 copied flags scalar wrapper 证明查询路径；不把 flag mutation 或 calibrator ownership 纳入本批。 |
| `btier-018` | `IBuilderConfig::getAvgTimingIterations` | TRT11 | 使用 TRT11 average timing iterations wrapper 与 native entrypoint；保持 official alias 与 deferred history 分离。 |
| `btier-019` | `IParser::getError` | TRT11 | 使用 ONNX parser copied error/count wrapper 和 official-token alias；ParserRefitter deferred history 继续作为审计记录。 |
| `btier-020` | `IParser::isSubgraphSupported` | TRT11 | 使用 parser subgraph support copied scalar query；不扩大到 parser model buffer lifecycle。 |
| `btier-021` | `IParserRefitter::getError` | TRT11 | 使用 ParserRefitter copied diagnostic snapshot/error wrapper；仍不提升 advanced model-buffer ownership 边界。 |
| `btier-022` | `IBuilder::getMaxBatchSize` | TRT8 | 使用 `MaxBatchSizeCompatibility` 与 TRT8 native getter 证明 legacy compatibility path；deferred history 保留。 |
| `btier-023` | `IBuilder::getMaxDLABatchSize` | TRT8 | 使用 `MaxDlaBatchSize` 与 TRT8 native getter 证明 compatibility path；不声称 DLA runtime proof。 |
| `btier-024` | `IBuilder::getMaxThreads` | TRT8 | 使用 `MaxThreads` 与 TRT8 native getter 证明 scalar control wrapper；不削弱 TRT8/TRT10/TRT11 version guard。 |

## B-tier 第三批 proof 批量收口

`btier-025` 到 `btier-040` 已进入同一个 `DeferredBTierWorkItemProofClosureLedgerTests` 门禁，重点覆盖 TRT8 builder/config/engine/context/parser 的 legacy safe alternative。该批不是新的 native ABI 扩展，而是把已经存在的 manifest、native source、generated interop、高层 wrapper 与 ProjectQuality proof 串成可回归证据链；deferred history 必须继续保留，不能删除来制造完成度。

本批边界仍然是 wrapper/docs/quality proof：不是 runtime proof，不是 package-consumer smoke，不是公开发布许可；也不允许把 layer、engine、parser 等 borrowed native object 以裸 `IntPtr`/`nint` 暴露给 public API。

| ID | 接口 | 版本 | 收口边界 |
| --- | --- | --- | --- |
| `btier-025` | `IBuilder::isNetworkSupported` | TRT8 | 使用 `IsNetworkSupported` 与 `jyppx-trt8-builder-is-network-supported` 证明三对象入参 safe query；`trt8-builder-is-network-supported-deferred` 继续保留。 |
| `btier-026` | `IBuilderConfig::canRunOnDLA` | TRT8 | 使用 `CanRunOnDla` 与 TRT8 config/layer copied bool 查询证明 wrapper；不暴露 borrowed layer pointer。 |
| `btier-027` | `IBuilderConfig::getAvgTimingIterations` | TRT8 | 使用 `GetAverageTimingIterations` 与 `trt8-builder-config-get-average-timing-iterations` 证明 compatibility getter；旧 avg timing deferred alias 保留为历史。 |
| `btier-028` | `IBuilderConfig::getDefaultDeviceType` | TRT8 | 使用 `GetDefaultDeviceType` 与 default-device getter/setter safe alternative；不删除 default-device deferred history。 |
| `btier-029` | `IBuilderConfig::getDeviceType` | TRT8 | 使用 layer device type get/is-set/reset/set 组合证明 safe wrapper；仍按 alias proof 处理，不视作 runtime DLA proof。 |
| `btier-030` | `IBuilderConfig::getDLACore` | TRT8 | 使用 `GetDlaCore` 与 TRT8 scalar getter 证明 copied result；不声称 DLA runtime execution proof。 |
| `btier-031` | `IBuilderConfig::getFlag` | TRT8 | 使用 `GetFlag`/`GetFlags` safe alternative 证明 BuilderFlag 查询路径；保留 `getFlags` deferred history。 |
| `btier-032` | `IBuilderConfig::getFlags` | TRT8 | 使用 `GetFlags` copied flags wrapper 证明 bulk flag 查询；不扩大到 mutation 或 calibrator ownership。 |
| `btier-033` | `IBuilderConfig::getMaxWorkspaceSize` | TRT8 | 使用 `MaxWorkspaceSizeCompatibilityInBytes` 与 native legacy getter 证明 compatibility path；TRT10/TRT11 仍建议 memory pool limit。 |
| `btier-034` | `IBuilderConfig::getMinTimingIterations` | TRT8 | 使用 `MinTimingIterationsCompatibility` 与 TRT8 native getter 证明 legacy path；跨版本 portable path 仍优先 average timing。 |
| `btier-035` | `IBuilderConfig::getQuantizationFlag` | TRT8 | 使用 `GetQuantizationFlag` 与 `GetQuantizationFlags` 组合证明 quantization flag 查询；两个 deferred history 均保留。 |
| `btier-036` | `IBuilderConfig::getQuantizationFlags` | TRT8 | 使用 copied quantization flags wrapper 证明只读 flags path；不触碰 calibrator callback 边界。 |
| `btier-037` | `ICudaEngine::getHardwareCompatibilityLevel` | TRT8 | 使用 `EngineHardwareCompatibilityLevel` 与 native scalar getter 证明 copied enum result；不把 engine runtime proof 混入本批。 |
| `btier-038` | `ICudaEngine::getProfileDimensions` | TRT8 | 复用 `GetProfileShape` 与 `GetProfileShapeValues` copied shape/value wrapper；`getProfileShapeValues` deferred history 继续保留。 |
| `btier-039` | `IExecutionContext::getNvtxVerbosity` | TRT8 | 使用 `GetNvtxVerbosity` 与 native scalar getter 证明 execution-context diagnostics wrapper；不触碰 execute/enqueue 边界。 |
| `btier-040` | `IParser::getError` | TRT8 | 使用 ONNX parser `ErrorCount`/`GetError` copied diagnostics wrapper；Caffe/UFF parser error-recorder deferred history 保留。 |

## B-tier 第四批 proof 批量收口

`btier-041` 到 `btier-046` 是 dashboard 扩到 60 个候选后按稳定排序追加的六个 TRT8 legacy parser alias-proof-ready 工作项。它们通过 caller-owned buffer 或 copied scalar snapshot 提供 safe alternative，并继续保留原始 deferred history；本批只固定编号、文档与质量门证据，不把 package restore/build 当作 runtime proof。

| ID | 接口 | 版本 | 收口边界 |
| --- | --- | --- | --- |
| `btier-041` | `IBinaryProtoBlob::getData` | TRT8 | 使用 `ReadCaffeBinaryProto` 将数据复制到 caller-owned managed buffer；不暴露 blob-owned pointer。 |
| `btier-042` | `IBinaryProtoBlob::getDataType` | TRT8 | 复用 binaryproto copied snapshot 返回数据类型；不公开 legacy blob handle。 |
| `btier-043` | `IBinaryProtoBlob::getDimensions` | TRT8 | 复用 binaryproto copied snapshot 返回维度；不延长 parser/blob native lifetime。 |
| `btier-044` | `IUffParser::getUffRequiredVersionMajor` | TRT8 | 使用 `GetUffRequiredVersion` copied version snapshot 返回 major。 |
| `btier-045` | `IUffParser::getUffRequiredVersionMinor` | TRT8 | 使用同一 copied version snapshot 返回 minor。 |
| `btier-046` | `IUffParser::getUffRequiredVersionPatch` | TRT8 | 使用同一 copied version snapshot 返回 patch；不公开 `IUffParser*`。 |

第四批仍保留全部 deferred history，`canDeleteDeferredRecord=false`、`canPromoteReleaseProof=false`、`isRuntimeExecutionProof=false`、`isPackageConsumerRuntimeProof=false`。

## 验证

每次修改分组脚本或设计门后至少运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-DeferredReadOnlyApiCandidatePlan.ps1 -IncludeMediumRisk -MaxItems 60
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-DeferredBTierProofClosureDashboard.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-DeferredBTierAliasProofClosureRecord.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-DeferredBTierImplementationWorkPackage.ps1
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build /p:UseSharedCompilation=false /nr:false --filter "FullyQualifiedName~DeferredBTier"
```

如果后续真的修改 native ABI，再追加：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
```
