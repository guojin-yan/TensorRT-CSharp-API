# Deferred Compatibility Diagnostics Proof

## 文章定位

本文记录下一批 B-tier compatibility/diagnostics proof：
`IBuilderConfig::getTilingOptimizationLevel`、
`ICudaEngine::hasImplicitBatchDimension`、
`IUffParser::getUffRequiredVersionPatch`、`IParser::getError` 和
`IParserRefitter::getError`。这些接口已经有真实 native entry、manifest、C# 路由或
copied snapshot；本批的工程目标是把历史 deferred 归并和跨版本边界写成可回归证据。

## 安全实现

- tiling level 只返回 enum/scalar；TRT10 有 deferred history，TRT11 是独立真实 route。
- implicit-batch 只返回 copied `bool`；TRT8 是 legacy alias，TRT10 保留历史 deferred，
  TRT11 的 public route 明确 `NotSupported`。
- UFF version 使用 TRT8 native 临时 `unique_ptr`，只把 major/minor/patch 复制为
  `TensorRtLegacyUffRequiredVersionSnapshot`；managed wrapper 不持有 parser。
- ONNX parser/refitter 错误通过 copied error struct、count 和 caller-buffer 字符串进入
  C#；不暴露 `IParserError*` 或 refitter pointer。

## 证据链

| 层 | 证据 |
|---|---|
| vendor | `NvInfer.h`、`NvInferRuntime.h`、`NvUffParser.h`、`NvOnnxParser.h` declarations |
| native | version guard、handle validation、output reset、copy/count contracts |
| managed | enum/bool/Version/copy snapshot wrapper，public surface 无裸指针 |
| smoke | `NetworkBuilderSmokeRunner`、`LegacyParserDiagnosticsSmokeRunner`、`OnnxToEngineSmokeRunner` |
| coverage | `trt-compatibility-diagnostics-candidate-audit.json` 与显式 deferred aliases |

## Proof 边界

这些 smoke 只证明 scalar/copy-only API 可调用、版本 guard 和 ownership 边界可诊断，
不等于真实模型 runtime、clean package-consumer runtime、release proof 或公开发布授权。
历史 deferred manifest 不能删除，`canDeleteDeferredRecords=false`、
`canPublishPublicly=false` 保持不变。

## 验证命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~DeferredCompatibilityDiagnosticsProofTests|FullyQualifiedName~BuilderConfigScalarControlsTests|FullyQualifiedName~EngineAndRnnReadonlyDiagnosticsTests|FullyQualifiedName~LegacyParserReadonlyDiagnosticsUpliftTests"
```
