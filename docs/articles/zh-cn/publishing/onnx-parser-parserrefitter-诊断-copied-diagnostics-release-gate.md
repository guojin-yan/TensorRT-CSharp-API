# ONNX Parser 与 ParserRefitter 诊断：copied diagnostics 到 release gate

ONNX 转换失败时，真正有价值的不是把 native parser 指针暴露给 C#，而是把 TensorRT 报告的错误数量、详细文本、节点信息、local function stack、VC plugin library 使用情况和 ParserRefitter 诊断复制到可审计的托管结构中。`TensorRtOnnxParserDiagnosticSnapshot`、`TensorRtOnnxParserDiagnosticSummary`、`TensorRtOnnxParserRefitterDiagnosticSnapshot` 和 `TensorRtOnnxParserRefitterDiagnosticSummary` 就是这条低风险诊断链路。

这些类型提供的是 copied diagnostics 和 wrapper surface evidence，不是 runtime proof、不是 package-consumer-runtime proof、不是 post-publish proof、不是 publish approval、不是 release close approval，也不是 package push。它们能帮助用户定位 ONNX export、unsupported operator、plugin、shape/profile 或 refit 问题，但不能证明 engine 已执行、输出正确或公开包可被 clean external consumer 使用。

## 适合

- 遇到 ONNX parse 失败，需要从 C# 读取 parser error 和 diagnostic text 的用户。
- 正在实现 OnnxToEngine/TensorRtExec report 的维护者。
- 需要审查 ParserRefitter、stripped plan refit 和 copied diagnostics 边界的发布负责人。
- 想确认 package consumer 至少能编译并访问 parser/refitter diagnostic wrapper 的质量门维护者。

## 关键路径

- Parser wrapper：`src/JYPPX.TensorRtSharp/TensorRtOnnxParser.cs`。
- Parser snapshot：`src/JYPPX.TensorRtSharp/TensorRtOnnxParserDiagnosticSnapshot.cs`。
- Parser diagnostic model：`src/JYPPX.TensorRtSharp/TensorRtOnnxParserDiagnostic.cs`。
- Parser model support：`src/JYPPX.TensorRtSharp/TensorRtOnnxParser.ModelSupport.cs`。
- ParserRefitter wrapper：`src/JYPPX.TensorRtSharp/TensorRtOnnxParserRefitter.cs`。
- ParserRefitter snapshot：`src/JYPPX.TensorRtSharp/TensorRtOnnxParserRefitterDiagnosticSnapshot.cs`。
- ParserRefitter native interop：`src/JYPPX.TensorRtSharp/Internal/Interop/NativeBridgeApi.ParserRefitterDiagnostics.cs`。
- TensorRtExec report projection：`src/JYPPX.TensorRtSharp.Tools/OnnxEngineBuildDiagnostics.cs`。
- Parser preflight model：`src/JYPPX.TensorRtSharp.Tools/OnnxEngineParserPreflightSnapshot.cs`。
- Build service capture：`src/JYPPX.TensorRtSharp.Tools/OnnxEngineBuildService.cs`。
- Consumer/package gates：`eng/Test-BridgePackageConsumer.ps1`、`eng/Test-RuntimePackageReadiness.ps1`、`eng/Test-ReleaseQualityGate.ps1`。
- Quality tests：`BridgePackageConsumerTests`、`RuntimePackageReadinessTests`、`RuntimeSerializationOnnxSupportTests`、`ParserRefitterBoundaryTests`、`TensorRtExecReportSchemaTests`。

## 为什么要单独进 release gate

ONNX parser diagnostics 是用户能否排障模型转换的基础能力。release gate 至少要确认这些 wrapper 没有从包里丢失：

```text
TensorRtOnnxParser.GetDiagnosticSnapshot()
TensorRtOnnxParser.GetDiagnosticSummary()
TensorRtOnnxParser.GetDiagnostics()
TensorRtOnnxParser.TryParse(...)
TensorRtOnnxParser.GetUsedVCPluginLibraries()
TensorRtOnnxParser.CheckModelSupport(...)
TensorRtOnnxParserRefitter.GetDiagnosticSnapshot()
TensorRtOnnxParserRefitter.GetDiagnosticSummary()
TensorRtOnnxParserRefitter.GetDiagnostics()
TensorRtOnnxParserRefitter.RefitFromBytes(...)
TensorRtOnnxParserRefitter.RefitLoadedModel()
```

这不是为了把诊断包装成 runtime proof，而是为了保证 package consumer 可以编译、调用并读取 pointer-free 结构。`BridgePackageConsumerTests` 与 `RuntimePackageReadinessTests` 会检查 `hasOnnxParserDiagnosticReadiness`、`hasOnnxParserRefitterDiagnosticReadiness`、snapshot、summary 和 runtime evidence kind。

## Parser snapshot 字段

`TensorRtOnnxParserDiagnosticSnapshot` 捕获这些 copied 字段：

```text
Line
ErrorCount
Diagnostics
DiagnosticSummary
UsedVCPluginLibraries
IdentityOperatorSupported
```

`ToSummary()` 会进一步给出：

```text
CopiedDiagnosticCount
DiagnosticSummaryLength
UsedVCPluginLibraryCount
RuntimeEvidenceKind = copied-readonly-summary
IsRuntimeExecutionEvidence = false
IsRuntimeExecutionProof = false
PointerFreeCopiedSummary = true
CanPromoteRuntimeProof = false
CanPromoteReleaseProof = false
CanDeleteDeferredRecord = false
```

`Diagnostics` 里的 `TensorRtOnnxParserDiagnostic` 会复制 description、file、function name、node name、node operator、local function stack 等信息。长字符串通过 variable-length string 读取，避免固定结构体截断；C# 只拿 copied string，不拿 parser-owned error pointer。

## ParserRefitter snapshot 字段

`TensorRtOnnxParserRefitterDiagnosticSnapshot` 捕获这些 copied 字段：

```text
Line
ErrorCount
Diagnostics
DiagnosticSummary
```

`TensorRtOnnxParserRefitterDiagnosticSummary` 也保持同样边界：

```text
CopiedDiagnosticCount
DiagnosticSummaryLength
RuntimeEvidenceKind = copied-readonly-summary
IsRuntimeExecutionEvidence = false
IsRuntimeExecutionProof = false
PointerFreeCopiedSummary = true
CanPromoteRuntimeProof = false
CanPromoteReleaseProof = false
CanDeleteDeferredRecord = false
```

ParserRefitter 的用途是 stripped plan / ONNX refit 生命周期中的诊断，不是 plugin lifecycle proof。`TensorRtOnnxParserRefitter` 会保持 `TensorRtRefitter` 与 `TensorRtLogger` keep-alive，`RefitFromBytes` 和 `LoadInitializer` 只在 native 调用期间 pin 或按 parser-refitter 生命周期持有初始化数据；诊断仍通过 copied APIs 读取。

## 与 TensorRtExec report 的关系

`OnnxEngineBuildService` 会在 build/preflight 过程中捕获 parser snapshot，并写入 `OnnxEngineParserPreflightSnapshot`。`OnnxEngineBuildDiagnostics` 再把它投影到 report：

```text
ParserPreflightSnapshot
DiagnosticsState
ErrorCount
CopiedDiagnosticCount
IdentityOperatorSupported
ModelSupportState
ModelSupported
CopiedSubgraphCount
CopiedUnsupportedSubgraphCount
CopiedNodeCount
CopiedSubgraphCountsMatchReportedCounts
ParserDiagnosticsEvidenceKind = copied-parser-diagnostics
ParserRefitterDiagnosticsEvidenceKind = copied-parser-refitter-diagnostics
CopiedDiagnosticsBoundary
ForbiddenSubstitutes
CanPromoteCopiedDiagnosticsToRuntimeProof = False
```

这使 TensorRtExec report 可以告诉用户“模型转换为什么失败”或者“parser preflight 看到了什么”，但报告仍不能从 copied diagnostics 自动晋级到 real-model-runtime proof。

## 使用边界

使用方可以把 parser/refitter diagnostics 当作模型转换排障入口：

1. 先运行 `OnnxToEngine` 或 `TensorRtExec` 生成 build/report evidence。
2. 如果 parser 失败，读取 copied diagnostic snapshot/summary。
3. 根据 error count、diagnostic text、used VC plugin libraries、Identity support、model support 和 unsupported subgraph 信息回到 ONNX 导出或 plugin 计划。
4. 如果走 stripped/refit 生命周期，再检查 ParserRefitter diagnostic snapshot、refit snapshot 和 persistence snapshot。
5. 只有真实模型输入、真实 engine、输出 JSON、stdout/stderr、SHA256、host metadata 和 owner review 齐备后，才可能进入 `real-model-runtime` 候选。
6. 只有 clean external consumer 安装公开包或候选包、restore/build/smoke 成功并通过 owner input validator，才可能进入 `package-consumer-runtime`。

不要把以下内容写成 proof：

- `TensorRtExec report`
- `OnnxToEngine report`
- `ParserPreflightSnapshot`
- `TensorRtOnnxParserDiagnosticSnapshot`
- `TensorRtOnnxParserRefitterDiagnosticSnapshot`
- readonly diagnostics
- local feed package consumer
- ProjectReference consumer
- direct `.nupkg` install
- dry-run
- build-only
- dependency-probe-only
- GitHub Actions dry-run
- GUI screenshot
- `failedBlockerCount=0`

## 版本和 deferred 边界

本阶段已经把 runtime readiness 拆成独立 wrapper groups：

- `onnx-parser-diagnostic-readiness`：覆盖 parser error count、copied diagnostics、summary、used VC plugin library summary、identity operator support、model support 和 `TensorRtOnnxParser.GetDiagnosticSnapshot`。
- `onnx-parser-refitter-diagnostic-readiness`：覆盖 ParserRefitter error count、copied diagnostics、summary 和 `TensorRtOnnxParserRefitter.GetDiagnosticSnapshot`。

这两个分组的 `evidence-kind` 都是 `compile-surface-proof`，`runtime-evidence` 分别是 `copied-parser-diagnostics` 和 `copied-parser-refitter-diagnostics`，并且 `proof=false`。

不要因为 wrapper surface 已经存在，就删除 deferred history。ParserRefitter create/refit/load model proto/initializer 的 ownership、pinning、stripped plan lifecycle 和 TRT10/TRT11 guard 必须继续由 manifest、native implementation、generated interop、C# wrapper 和质量门共同证明；不确定的 pointer/ownership 行仍要留在 deferred audit 中。

## 验证命令

```powershell
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~BridgePackageConsumerTests|FullyQualifiedName~RuntimePackageReadinessTests|FullyQualifiedName~RuntimeSerializationOnnxSupportTests|FullyQualifiedName~ParserRefitterBoundaryTests|FullyQualifiedName~TensorRtExecReportSchemaTests"
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseQualityGate.ps1 -Strict
```

这些命令是 release gate / wrapper surface / report schema 证据，不是公开发布动作。当前用户已要求不要使用 GitHub Actions 额度，因此不要 workflow dispatch，不要 push，不要发布 NuGet/GitHub Packages，不要上传 GitHub Release。

## 配图建议

- 一张 copied diagnostics 流程图：TensorRT parser error -> native bridge caller buffer -> C# snapshot -> report boundary。
- 一张 ParserPreflightSnapshot JSON 摘要图，突出 `ParserDiagnosticsEvidenceKind` 和 `CanPromoteCopiedDiagnosticsToRuntimeProof=False`。
- 一张 release evidence ladder 图，把 copied diagnostics 放在 build/report context evidence 层，而不是 runtime proof 层。

## 下一步

后续可以继续把 parser diagnostics 和 TensorRtExec report 联系起来：在不加载 plugin library、不注册 creator、不暴露 native borrowed pointer 的前提下，补更多模型转换失败分类、trtexec-like 参数来源和 owner action 提示。
