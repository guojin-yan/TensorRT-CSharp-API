# TensorRT Compatibility Diagnostics Candidate Audit

生成日期：2026-07-26

本批收口的是已经存在真实 route 的 B-tier compatibility/diagnostics 接口，重点是
显式 deferred-history 归并与跨版本 proof，不是删除 deferred manifest 或宣称 release
runtime proof。

| Candidate | Real route | Deferred history | Ownership / boundary | Cross-version decision |
|---|---|---|---|---|
| `IBuilderConfig::getTilingOptimizationLevel` | TRT10 scalar getter；TRT11 有独立 scalar getter | `trt10-builder-config-get-tiling-optimization-level-deferred` | copied `int32_t`/enum；无 callback 或 borrowed object | TRT10 `implemented-with-deferred-history`，TRT11 `implemented` |
| `ICudaEngine::hasImplicitBatchDimension` | TRT8 legacy alias、TRT10 scalar bool query | `trt10-cuda-engine-has-implicit-batch-dimension-deferred` | copied bool；仅 legacy compatibility 查询 | TRT8 `implemented`，TRT10 `implemented-with-deferred-history`，TRT11 public route 明确 `NotSupported` |
| `IUffParser::getUffRequiredVersionPatch` | TRT8 native temporary parser 将 major/minor/patch copied 到 snapshot | `trt8-uff-parser-get-uff-required-version-patch-deferred` | `std::unique_ptr` 只在 native 内部；public 只返回 `Version` snapshot | TRT8 `implemented-with-deferred-history`；TRT10/11 不提供 legacy UFF route |
| `IParser::getError` | TRT8/10/11 copied error/count wrapper | TRT10/11 parser-refitter deferred history；TRT8 Caffe/UFF recorder rows 继续 deferred | copied error struct/string；不暴露 `IParserError*` | 所有已有 route 保持 `implemented-with-deferred-history` |
| `IParserRefitter::getError` | TRT10/11 copied error/count/snapshot wrapper | `trt10-parser-refitter-get-error-deferred`、`trt11-parser-refitter-get-error-deferred` | owner-bound refitter handle；错误文本 caller-buffer copy | TRT10/11 `implemented-with-deferred-history`；TRT8 不宣称支持 |

## 审计结论

- `NvInfer.h`、`NvInferRuntime.h`、`NvUffParser.h`/`NvOnnxParser.h` 的 vendor declarations
  与对应 native manifests/sources 已对齐。
- import library/DLL 证据沿用现有 native ABI declaration/export parity 与版本构建记录；
  本批没有引入新的 vendor binary 下载。
- C# public surface 只出现 enum、bool、int、`Version`、copied error/snapshot；没有
  `IntPtr`、`nint`、`UIntPtr`、`SafeHandle` 或 TensorRT-owned borrowed pointer。
- coverage exporter 现在为 tiling、implicit-batch、parser/refitter rows 建立显式 deferred
  aliases；UFF patch alias 已存在并由本批补齐 tuple proof。
- deferred history remains：所有历史 records 保留，`canDeleteDeferredRecords=false`。

## Smoke / Proof Boundary

现有 `NetworkBuilderSmokeRunner`、`LegacyParserDiagnosticsSmokeRunner` 和
`OnnxToEngineSmokeRunner` 覆盖这些 copied/scalar routes。它们证明 wrapper 可调用与
边界可诊断，不等于真实模型 runtime、clean package-consumer runtime、release proof 或
公开发布授权。
