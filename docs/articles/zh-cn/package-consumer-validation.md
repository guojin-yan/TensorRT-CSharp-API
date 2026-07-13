# 包消费端验证

`eng/Test-PackageConsumer.ps1` 用于验证托管主包与 runtime 包在真实消费端项目中的还原、构建、native asset 复制和可选 smoke。

发布前 PublicDocs proof boundary freeze 要求本文持续保留 package-consumer runtime proof 的边界：`local feed`、`ProjectReference`、direct `.nupkg`、build-only、dependency-probe、dashboard、runbook、candidate 和 draft 都是 non-proof 替代物。只有仓库外 clean consumer 项目从声明的 package source restore managed/runtime package，并提供 runtime smoke stdout/stderr、log SHA256、nupkg/native asset SHA256、host metadata，且 strict validator 通过后，才允许晋级 package-consumer runtime proof。

当前 Windows 重点 runtime key：

- `win-x64-trt8.6-cuda11.8-cudnn8.9`
- `win-x64-trt8.6-cuda12.1-cudnn8.9`
- `win-x64-trt10.11-cuda11.8-cudnn8.9`
- `win-x64-trt10.11-cuda12.9-cudnn9.22`
- `win-x64-trt11.0-cuda12.9-cudnn9.22`
- `win-x64-trt11.0-cuda13.2-cudnn9.22`

CUDA `12.9` 已安装。目标为 CUDA `12.9` 的包必须使用 CUDA `12.9` 以及匹配 TensorRT/cuDNN 资产完成验证；之前 CUDA `12.3` 的临时 fallback 已废弃。

示例命令：

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RuntimePackageKey win-x64-trt10.11-cuda11.8-cudnn8.9 -RunSmoke -SmokeRuntimePackageKey win-x64-trt10.11-cuda11.8-cudnn8.9
```

脚本默认会在每个 runtime key 验证结束后删除生成的消费端项目和临时 NuGet restore 缓存，避免自托管 runner 在矩阵打包时反复堆积 TensorRT/CUDA/cuDNN 大文件。本地排查 restore 或 native-copy 问题时，可以传入 `-KeepConsumerOutput` 保留 `build-out/package-consumer/<runtime-key>`。

报告中会写入 `RealCallbackRuntimeEvidence` 对象，用于区分真实 callback runtime 证据与普通 smoke 结果。`SmokeResult=passed` 不能自动证明 callback 已由 TensorRT 触发；只有 `RealCallbackRuntimeEvidence.Status=ready`、`EvidenceKind=real-callback-runtime`、`RealCallbackRuntime=True`、所有 required markers 齐全且 `IsRealCallbackRuntimeProof=True` 时，readiness 才能晋级。CUDA driver/runtime 不兼容、应用控制阻塞和普通 smoke 失败会分别记录为 `blocked-by-cuda-driver`、`blocked-by-application-control` 或 `blocked`，不会被解释为 API 缺失。

`output-allocator-runtime-proof-precheck` 和 `debug-listener-runtime-proof-precheck` 也会被归类为非 proof callback evidence。即使它们报告 `RuntimeEvidenceKind=runtime-gate`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，也只是真实 runtime proof 前置阻塞项，不表示 TensorRT 已调用 `IOutputAllocator::notifyShape`、`IOutputAllocator::reallocateOutput` 或 `IDebugListener::processDebugTensor`。

2026-06-12 本机最新证据：

- `win-x64-trt10.11-cuda11.8-cudnn8.9`：`16/16` native assets，smoke `passed`，探针输出 TensorRT `10.11.0`、CUDA `11.8`。
- `win-x64-trt10.11-cuda12.9-cudnn9.22`：`19/19` native asset patterns，smoke `passed`，探针输出 TensorRT `10.11.0`、CUDA `12.9`。
- `win-x64-trt11.0-cuda12.9-cudnn9.22`：`19/19` native asset patterns，smoke `passed`，探针输出 TensorRT `11.0.0`、CUDA `12.9`。
- `win-x64-trt11.0-cuda13.2-cudnn9.22`：`19/19` native asset patterns，restore/build/native-copy 通过；full package consumer smoke 已请求，packaged runtime 已启动到 TensorRT/bridge 探测，但当前机器在 `cudaRuntimeGetVersion` 处返回 CUDA error 35，报告为 `blocked-by-cuda-driver`，`IsRealCallbackRuntimeProof=False`。
- managed package：`JYPPX.TensorRT.CSharp.API 4.0.0-alpha.1`
- 报告：`artifacts/package-consumer/package-consumer-validation-summary.md`

如果 Windows Defender Application Control / 应用控制策略阻止新构建的消费端输出，并出现 `0x800711C7`，可以显式传入 `-SignConsumerOutput`。该开关会在 smoke 前使用本地开发代码签名证书签名生成的消费端程序、托管程序集和桥接 DLL：

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RuntimePackageKey win-x64-trt11.0-cuda12.9-cudnn9.22 -RunSmoke -SmokeRuntimePackageKey win-x64-trt11.0-cuda12.9-cudnn9.22 -SignConsumerOutput
```

消费端验证应至少检查：

- managed package restore
- runtime package restore
- native 资产复制数量
- 缺失 native 资产列表
- 可选 smoke 结果
- 高层 wrapper 编译面 marker，例如 plugin registry inventory、ONNX parser diagnostic/subgraph 查询、builder config 只读 getter 和无 borrowed plugin creator 指针暴露边界。

`eng/Test-PackageConsumer.ps1` 的报告现在还会显式区分 `ReadonlySummaryEvidenceKind`、`WrapperSurfaceEvidenceKind`、`RuntimeSmokeClassification`、`IsRuntimeExecutionEvidence` 和 `IsPackageConsumerRuntimeProof`。`ReadonlySummaryEvidenceKind=readonly-summary-diagnostics-not-runtime-proof` 说明 `EngineDeploymentSummary=`、`BuilderConfigDeploymentSummary=`、`ExecutionContextDeploymentSummary=`、`SerializationConfigSummary=`、`RuntimeConfigSummary=`、`GraphDiagnosticSummary=`、`GraphExecDiagnosticSummary=`、`MemoryRangeSummary=` 等 marker 只能作为 wrapper/diagnostic evidence，不能替代 runtime proof。

`IsPackageConsumerRuntimeProof=True` 只能来自 clean consumer restore/build/native-copy/runtime smoke、package hash、host metadata、stdout/stderr summary、runtime smoke log SHA256 和 strict validator 共同通过。bridge-only、dependency probe、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、OnnxToEngine report、YoloVision matrix、readonly summary、build-only、dry-run、template 和 `blocked-by-cuda-driver` 都必须保持 forbidden substitute。

更多预检字段见 `docs/articles/zh-cn/package-consumer-runtime-proof-preflight-matrix.md` 与 `artifacts/final-release/package-consumer-runtime-proof-preflight-matrix.json`。Package Consumer Runtime Proof 预检矩阵是 owner 执行合同，不是 proof 本身。

`eng/Test-PackageConsumer.ps1` 会把预检矩阵中的当前 runtime key 合并到报告的 `RuntimeProofPreflight` 字段和 Markdown 的 `Runtime Proof Preflight Boundary` 小节。该字段用于审计 restore source mode、native asset 预期数量、owner action required、blocked reason 和 validator command 是否与矩阵一致；它不允许把 `IsPackageConsumerRuntimeProof` 从 `false` 提升为 `true`。真正晋级只能由 strict external proof validator 在读取真实 clean consumer runtime smoke log、nupkg/hash、host metadata 和 no-ProjectReference 证据后完成。

`win-x64-trt11.0-cuda13.2-cudnn9.22` 已在 2026-06-25 完成完整 split/full 包打包，并通过 restore/build/native-copy 验证，native asset patterns 为 `19/19`；但不能仅凭这些证据视为真实 runtime callback 可用。当前 full package consumer smoke 结果是 `blocked-by-cuda-driver`，需要在 CUDA 13-capable driver/runtime 上重新运行 runtime smoke；即使普通 smoke 通过，也只有完整 `real-callback-runtime` markers 齐全时才能提升 callback proof。

2026-07-05 新增 local feed consumer proof：`eng/Test-LocalNuGetFeedConsumer.ps1` 现在按 `.nupkg` 最新写入时间优先选择本地包，避免旧 `4.0.0-local` 遮蔽刚打出的 `4.0.0`。本地验证使用 `JYPPX.TensorRT.CSharp.API 4.0.0` 与 `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda13.2.cudnn9.22 4.0.0`，`RestoreSourceMode=local-feed-only`、`UsesProjectReference=False`、native assets `19/19`、`RunStatus=dependency-probe-passed`，并在 consumer 输出中确认 `HighLevelWrapperSurface` 包含 `IsSubgraphSupported`、`FindCreator`、`TryFindCreator`、`GetDlaCore`、`GetMaxTactics`、`GetQuantizationFlags`、`MaxBatchSizeCompatibility`、`MaxDlaBatchSize`、`no-public-borrowed-plugin-creator-pointer` 和 `local-feed-is-not-post-publish-proof`。该 proof 仍只属于本地 feed restore/build/dependency-probe，不是公开 channel post-publish proof，也不是 TensorRT callback/runtime 推理 proof。

2026-07-09 新增 CUDA 初始化本地 smoke 分类：`smoke/CudaDeviceInitializationProofRunner/Program.cs` 会按 `SetValidDevices -> InitDevice -> ChooseDevice` 顺序验证初始化路径，并显式输出 `ProofKind=local-smoke-not-external-proof`、`IsPackageConsumerRuntimeProof=False` 和 `CanPromoteRuntimeProof=False`。`artifacts/final-release/cuda-device-initialization-local-smoke-classification.json` 只把它归类为本地 smoke scaffold；成功日志不能自动晋级，`Skipped=True` 也必须视为 forbidden substitute。只有 clean consumer restore/build/native-copy/runtime smoke、包 hash、host metadata、runtime log SHA256 与 strict external validator 同时通过后，才允许形成 package-consumer-runtime proof。

正式发布前仍需完成 NVIDIA CUDA / cuDNN / TensorRT 再分发许可复核。
