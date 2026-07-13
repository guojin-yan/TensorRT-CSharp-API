# ONNX Parser 与 ParserRefitter 诊断：copied diagnostics 到 release gate

本文说明 `TensorRtOnnxParserDiagnosticSnapshot`、`TensorRtOnnxParserDiagnosticSummary`、`TensorRtOnnxParserRefitterDiagnosticSnapshot` 和 `TensorRtOnnxParserRefitterDiagnosticSummary` 在发布候选阶段的证据边界。它们提供的是 copied diagnostics 和 wrapper surface evidence，不是 runtime proof、不是 post-publish proof、不是 publish approval、不是 release close approval，也不是 package push。

## 为什么要单独进 release gate

ONNX 转换失败时，最重要的不是暴露 native parser 指针，而是把错误数量、错误文本、summary、VC plugin library 使用情况和 ParserRefitter 诊断复制到 C# 可审计结构中。这样 package consumer 可以编译并读取诊断类型，release gate 也能确认 wrapper surface 没有从包里丢失。

本阶段已经把 runtime readiness 拆成独立 wrapper groups：

- `onnx-parser-diagnostic-readiness`：覆盖 parser error count、copied diagnostics、summary、used VC plugin library summary、identity operator support 和 `TensorRtOnnxParser.GetDiagnosticSnapshot`。
- `onnx-parser-refitter-diagnostic-readiness`：覆盖 ParserRefitter error count、copied diagnostics、summary 和 `TensorRtOnnxParserRefitter.GetDiagnosticSnapshot`。

这两个分组的 `evidence-kind` 都是 `compile-surface-proof`，`runtime-evidence` 分别是 `copied-parser-diagnostics` 和 `copied-parser-refitter-diagnostics`，并且 `proof=false`。

## 使用边界

使用方可以把 parser/refitter diagnostics 当作模型转换排障入口：

1. 先运行 `OnnxToEngine` 或 `TensorRtExec` 生成 build/report evidence。
2. 如果 parser 失败，读取 copied diagnostic snapshot/summary。
3. 根据 error count、diagnostic text、used VC plugin libraries 和 unsupported operator 信息回到 ONNX 导出或 plugin 计划。
4. 只有真实模型输入、真实 engine、输出 JSON、stdout/stderr、SHA256、host metadata 和 owner review 齐备后，才可能进入 `real-model-runtime` 候选。

不要把以下内容写成 proof：

- `TensorRtExec report`
- `OnnxToEngine report`
- readonly diagnostics
- local feed
- ProjectReference
- direct `.nupkg`
- dry-run
- build-only
- `failedBlockerCount=0`

## 验证命令

```powershell
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~BridgePackageConsumerTests|FullyQualifiedName~RuntimePackageReadinessTests|FullyQualifiedName~RuntimeSerializationOnnxSupportTests"
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseQualityGate.ps1 -Strict
```

## 下一步

后续可以继续把 parser diagnostics 和 TensorRtExec report 联系起来：在不加载 plugin library、不注册 creator、不暴露 native borrowed pointer 的前提下，补更多模型转换失败分类、trtexec-like 参数来源和 owner action 提示。
