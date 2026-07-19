# Deferred B-tier 41-45 proof closure

## 适用读者

本文面向继续推进 deferred 边界提升的维护者，用于说明 `btier-041` 到 `btier-045` 的真实完成边界。

## 结论

`btier-041` 到 `btier-045` 不再作为缺 native 实现的接口处理。它们已经有真实参数 ABI、native source、managed wrapper 和 deferred history。当前阶段的目标是把 proof closure 固化为文档与 ProjectQuality 质量门，而不是删除 deferred history 或把 readonly diagnostics 晋级为 runtime proof。

| 工作项 | 接口 | Safe alternative | Deferred history | Proof closure |
|---|---|---|---|---|
| `btier-041` | `IBuilderConfig::getTilingOptimizationLevel` | `jyppx-trt10-builder-config-get-tiling-optimization-level` | `trt10-builder-config-get-tiling-optimization-level-deferred` | `TensorRtBuilderConfig.GetTilingOptimizationLevel()` 通过 copied scalar getter 返回 tiling level。 |
| `btier-042` | `ICudaEngine::hasImplicitBatchDimension` | `jyppx-trt10-cuda-engine-has-implicit-batch-dimension` | `trt10-cuda-engine-has-implicit-batch-dimension-deferred` | `TensorRtEngine.HasImplicitBatchDimensionCompatibility` 仅暴露 legacy bool 查询，不暴露 engine borrowed pointer。 |
| `btier-043` | `IExecutionContext::getNvtxVerbosity` | `jyppx-trt10-execution-context-get-nvtx-verbosity` | `trt10-execution-context-get-nvtx-verbosity-deferred` | `TensorRtExecutionContext.GetNvtxVerbosity()` 只读取 scalar diagnostics，不触碰 enqueue/callback。 |
| `btier-044` | `IParser::getError` | `trt10-onnx-parser-get-error` / `trt10-onnx-parser-get-error-count` | `trt10-parser-refitter-get-error-deferred` | `TensorRtOnnxParser.GetError()` 和 `GetDiagnostics()` 复制错误结构与字符串，不返回 parser-owned 指针。 |
| `btier-045` | `IParserRefitter::getError` | `trt10-parser-refitter-get-error` / `trt10-parser-refitter-get-error-count` | `trt10-parser-refitter-get-error-deferred` | `TensorRtOnnxParserRefitter.GetError()`、`GetDiagnostics()`、`GetDiagnosticSnapshot()` 复制诊断信息并保留 owner/lifetime 边界。 |

## 执行边界

- `canDeleteDeferredRecord=false`：deferred manifest 是历史审计边界，不能删除来制造完成度。
- `canPromoteReleaseProof=false`：这些接口是 readonly API proof closure，不是 package-consumer-runtime proof、post-publish proof 或 release close proof。
- public C# API 必须保持 pointer-free，不能暴露 `IntPtr`、`nint`、`IParserError*`、`IParserRefitter*` 或 TensorRT-owned borrowed pointer。
- 字符串和数组输出只能通过 copied struct、count/copy 或 caller buffer 模式进入 managed wrapper。
- TRT10/TRT11 version guard 必须继续由 native manifest、native source 和 managed interop 同时覆盖。

## 验证入口

推荐每轮直接运行以下质量门，而不是重新扫描全量 deferred 文件：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-DeferredReadOnlyApiCandidatePlan.ps1 -IncludeMediumRisk -MaxItems 60
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-DeferredBTierProofClosureDashboard.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-DeferredBTierAliasProofClosureRecord.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-DeferredBTierImplementationWorkPackage.ps1
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~DeferredBTier41To45ProofClosureTests|FullyQualifiedName~DeferredBTierWorkItemProofBatchTests"
```

## 下一步

`deferred-btier-work-item-proof-closure-ledger.json` 已将 `btier-001` 到 `btier-045` 全部标记为 `source-quality-proof-closed`，work package 的 `remainingWorkItemCount=0`。后续阶段应从新 candidate audit、真实 external model/runtime gap 或已通过 ownership design gate 的候选中选择批次；遇到 callback、allocator、plugin instance、resource acquire/release 或 borrowed pointer lifetime 不明确的接口，立即降级为 design gate。
