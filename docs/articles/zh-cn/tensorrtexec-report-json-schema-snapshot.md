# TensorRtExec Report JSON Schema Snapshot

## 适用读者

本文面向维护 TensorRtExec report 的开发者和测试人员，用于定义 report JSON 的稳定字段，并防止 GUI/CLI 输出悄悄漂移。

## 解决问题

TensorRtExec report 是重要诊断材料，但如果字段不稳定，用户和 release gate 都无法可靠消费。本文建议建立 schema snapshot，锁定 `OnnxEngineBuildDiagnostics.ToJson` 的真实顶层字段和 `ReportBoundary`，同时明确 build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 都不是 runtime proof。

## 背景与场景

TensorRtExec 同时服务 CLI 和 WinForms。两种入口都应生成相同语义的 report：CLI 适合自动化，WinForms 适合人工排查。Schema snapshot 现在以 `OnnxEngineBuildDiagnostics.ToJson` 的真实输出为准，可以在不执行真实 TensorRT 构建的情况下检查字段契约，避免 UI 改动破坏下游工具。

## 实现路径

1. 定义真实顶层字段：`Success`、`Skipped`、`State`、`TensorRtLine`、`ModelSource`、`EnginePath`、`NormalizedCommandLine`、`NormalizedCommandSha256`、`DeploymentOptions`、`RuntimeOptions`、`PreflightMetadata`、`LoadedEngineDiagnostics`、`WorkspaceBytes`、`OptionImplementationStatus`、`BenchmarkSummary`、`ProofClassification`、`EvidenceClassifications`、`BuildEvidenceOnly`、`IsRuntimeExecutionProof`、`ModelEvidence`、`EvidenceSidecarDiagnostics`、`Diagnostics`、`LogLines` 和 `ReportBoundary`。
2. 为 `input` 锁定 ONNX path/hash、input names、shape profile 和 dynamic shape 字段。
3. 为 `environment` 锁定 OS、RID、CUDA、TensorRT、cuDNN 和 runtime package 字段。
4. 为 `boundary` 锁定 `isRuntimeProof=false`、`isBuildOnly` 和 forbidden substitute reason。
5. 在质量测试中读取示例 report 或 schema 文档，确认关键字段和非 proof 边界存在。

## 代码与文件入口

- `applications/TensorRtExec`：report 生成逻辑。
- `applications/TensorRtExec/tensor-rt-exec-feature-matrix.json`：工具功能矩阵。
- `samples/OnnxToEngine/trtexec-parity-matrix.json`：参数 parity 数据。
- `docs/articles/zh-cn/tensorrtexec-report-schema-guide.md`：report schema 说明。
- `tests/JYPPX.ProjectQuality.Tests`：schema snapshot 测试入口。

## 图示建议

建议用一张 JSON skeleton 展示当前扁平字段层级，并在 `ReportBoundary` 节点用红色标注 “diagnostic only, not runtime proof”。

## 边界说明

即使 schema snapshot 通过，TensorRtExec report 仍是构建和诊断材料，不是 runtime proof。build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、YoloVision matrix、OnnxToEngine report 和 readonly diagnostics 不能替代外部 clean consumer 运行记录。

## 下一步

后续如要新增字段，必须同时更新 `applications/TensorRtExec/tensor-rt-exec-report.schema.json`、真实 `ToJson` 输出、schema snapshot 测试和本文档，避免再次出现 schema 草案与真实 report 输出漂移。
