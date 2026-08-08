# Deferred B-tier 41-46 proof closure

## 适用读者

本文面向继续推进 deferred 边界提升的维护者，用于说明 `btier-041` 到 `btier-046` 的真实完成边界。文件名保留历史批次入口，正文和机器台账以当前稳定工作项编号为准。

## 结论

`btier-041` 到 `btier-046` 不再作为缺 native 实现的接口处理。它们已有 caller-owned native ABI、native source、managed copied snapshot 和 deferred history。当前阶段的目标是把 proof closure 固化为文档与 ProjectQuality 质量门，而不是删除 deferred history 或把 readonly diagnostics 晋级为 runtime proof。

| 工作项 | 接口 | Safe alternative | Deferred history | Proof closure |
|---|---|---|---|---|
| `btier-041` | `IBinaryProtoBlob::getData` | `trt8-legacy-caffe-binary-proto-copy` | `trt8-binary-proto-blob-get-data-deferred` | `TensorRtLegacyParserDiagnostics.ReadCaffeBinaryProto()` 将数据复制到 caller-owned buffer。 |
| `btier-042` | `IBinaryProtoBlob::getDataType` | `trt8-legacy-caffe-binary-proto-copy` | `trt8-binary-proto-blob-get-data-type-deferred` | `TensorRtCaffeBinaryProtoSnapshot` 保存 copied data type，不暴露 blob handle。 |
| `btier-043` | `IBinaryProtoBlob::getDimensions` | `trt8-legacy-caffe-binary-proto-copy` | `trt8-binary-proto-blob-get-dimensions-deferred` | `TensorRtCaffeBinaryProtoSnapshot` 保存 copied dimensions，不返回 blob-owned pointer。 |
| `btier-044` | `IUffParser::getUffRequiredVersionMajor` | `trt8-legacy-uff-get-required-version` | `trt8-uff-parser-get-uff-required-version-major-deferred` | `TensorRtLegacyUffRequiredVersionSnapshot` 保存 copied major。 |
| `btier-045` | `IUffParser::getUffRequiredVersionMinor` | `trt8-legacy-uff-get-required-version` | `trt8-uff-parser-get-uff-required-version-minor-deferred` | 同一 snapshot 保存 copied minor。 |
| `btier-046` | `IUffParser::getUffRequiredVersionPatch` | `trt8-legacy-uff-get-required-version` | `trt8-uff-parser-get-uff-required-version-patch-deferred` | 同一 snapshot 保存 copied patch，不公开 `IUffParser*`。 |

## 执行边界

- `canDeleteDeferredRecord=false`：deferred manifest 是历史审计边界，不能删除来制造完成度。
- `canPromoteReleaseProof=false`：这些接口是 readonly API proof closure，不是 package-consumer-runtime proof、post-publish proof 或 release close proof。
- public C# API 必须保持 pointer-free，不能暴露 `IntPtr`、`nint`、`IBinaryProtoBlob*`、`IUffParser*` 或 TensorRT-owned borrowed pointer。
- 字符串和数组输出只能通过 copied struct、count/copy 或 caller buffer 模式进入 managed wrapper。
- TRT8 legacy parser version guard 必须继续由 native manifest、native source 和 managed interop 同时覆盖。

## 验证入口

推荐每轮直接运行以下质量门，而不是重新扫描全量 deferred 文件：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-DeferredReadOnlyApiCandidatePlan.ps1 -IncludeMediumRisk -MaxItems 60
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-DeferredBTierProofClosureDashboard.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-DeferredBTierAliasProofClosureRecord.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-DeferredBTierImplementationWorkPackage.ps1
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~DeferredBTier41To45ProofClosureTests|FullyQualifiedName~DeferredBTierWorkItemProofClosureLedgerTests|FullyQualifiedName~LegacyParserReadonlyDiagnosticsUpliftTests"
```

## 下一步

`deferred-btier-work-item-proof-closure-ledger.json` 已将 `btier-001` 到 `btier-051` 全部标记为 `source-quality-proof-closed`，work package 的 `remainingWorkItemCount=0`。后续阶段应从新 candidate audit、真实 external model/runtime gap 或已通过 ownership design gate 的候选中选择批次；遇到 callback、allocator、plugin instance、resource acquire/release 或 borrowed pointer lifetime 不明确的接口，立即降级为 design gate。
