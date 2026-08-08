# Deferred B-tier 47-51 proof closure

本批收口 `btier-047` 到 `btier-051` 的 safe-alternative proof。文件名保留历史批次入口，正文和机器台账以当前稳定工作项编号为准：

- `btier-047` `IBuilderConfig::getTilingOptimizationLevel`：通过 TRT10 scalar getter 读取 tiling level。
- `btier-048` `ICudaEngine::hasImplicitBatchDimension`：通过 TRT10 legacy compatibility bool 查询。
- `btier-049` `IExecutionContext::getNvtxVerbosity`：通过 TRT10 scalar diagnostics 查询。
- `btier-050` `IParser::getError`：通过 ONNX parser copied error struct/string 查询。
- `btier-051` `IParserRefitter::getError`：通过 parser-refitter copied error struct/string 查询。

这些接口已有 native ABI、managed wrapper 和版本 guard；本批新增显式 coverage alias、
work-package proof gate、ProjectQuality assertions 和文章记录。所有 public API 保持
pointer-free，不暴露 `IntPtr`、`nint` 或 vendor-owned pointer。

`canDeleteDeferredRecord=false`，`canPromoteReleaseProof=false`。这些证据是 source-quality
proof closure，不是 runtime execution、package-consumer runtime、post-publish 或 release proof。
