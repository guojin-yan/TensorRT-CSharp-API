# TensorRtExec Report Schema 指南

## 适用读者

本文面向需要消费 `applications/TensorRtExec` report JSON 的开发者、测试人员和发布负责人，重点说明哪些字段用于排查，哪些字段不能作为 runtime proof。

## 解决问题

构建工具 report 容易被误用为发布证据。本文把 TensorRtExec report schema 拆为输入、环境、构建配置、结果、错误、hash 和边界字段，明确 build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 与真实 runtime proof 的区别。

## 背景与场景

TensorRtExec 的价值是把 ONNX 解析、profile、precision、builder config 和 engine 输出记录成机器可读报告。报告能帮助定位 TensorRT/CUDA 版本、DLL 缺失、shape 不匹配和 parser 错误，但它仍然是构建面证据，不是 clean consumer 运行面证据。

## 实现路径

1. `input` 字段记录 ONNX 路径、SHA256、input names 和 shape profile。
2. `environment` 字段记录 OS、RID、CUDA、TensorRT、cuDNN 和 runtime package。
3. `builder` 字段记录 precision、workspace、timing cache、profile 和 tactic source。
4. `result` 字段记录 engine 路径、engine SHA256、elapsed time、warnings 和 native status code。
5. `boundary` 字段显式标注是否为 build-only、是否缺少 runtime execution、是否可用于 release proof。

## 代码与文件入口

- `applications/TensorRtExec`：report 生成入口。
- `applications/TensorRtExec/tensor-rt-exec-feature-matrix.json`：功能覆盖矩阵。
- `samples/OnnxToEngine/trtexec-parity-matrix.json`：trtexec parity 数据。
- `docs/articles/zh-cn/tensorrtexec-external-onnx-build-report.md`：外部 ONNX 构建报告说明。
- `docs/articles/zh-cn/tool-report-to-release-proof-record.md`：工具报告到发布证据的边界说明。

## 图示建议

当前 JSON schema 已对齐 `OnnxEngineBuildDiagnostics.ToJson` 的真实输出，而不是早期设想的 `input/environment/builder/result/boundary` 分层草案。顶层字段包括 `Success`、`State`、`TensorRtLine`、`NormalizedCommandLine`、`NormalizedCommandSha256`、`DeploymentOptions`、`RuntimeOptions`、`PreflightMetadata`、`LoadedEngineDiagnostics`、`WorkspaceBytes`、`OptionImplementationStatus`、`BenchmarkSummary`、`ProofClassification`、`EvidenceClassifications`、`BuildEvidenceOnly`、`IsRuntimeExecutionProof`、`IsRealModelRuntimeProof`、`IsPackageConsumerRuntimeProof`、`ModelEvidence`、`EvidenceSidecarDiagnostics`、`Diagnostics`、`LogLines` 和 `ReportBoundary`。

## 边界说明

TensorRtExec report 是重要诊断材料，但不能自动晋级为 runtime proof。OnnxToEngine report、YoloVision matrix、readonly diagnostics、build-only、dry-run、template、local feed、ProjectReference 和 direct `.nupkg` 都需要被 release gate 识别为非替代项。

## 下一步

质量测试会直接用 `OnnxEngineBuildDiagnostics.ToJson` 生成真实 report，并检查 schema 的 required 顶层字段、`ReportBoundary`、`OptionImplementationStatus`、`DeploymentOptions`、`RuntimeOptions`、`PreflightMetadata` 和 `LoadedEngineDiagnostics` 均能覆盖当前输出。
