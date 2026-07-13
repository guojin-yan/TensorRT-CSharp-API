# Builder Config Readback：发布前如何确认构建参数

Builder config readback 用来确认 workspace、memory pool、profiling verbosity、flags 和 runtime serialization/refit 相关配置是否被正确应用。它适合做构建诊断和报告补强，但不能替代真实推理运行。

## 适合

- 正在使用 OnnxToEngine 或 TensorRtExec 构建 engine 的使用者。
- 需要确认 builder config option layering 是否按预期工作的维护者。
- 想理解 `WorkspaceBytes`、memory pool readback 和 report 字段边界的发布审计人员。

## 关键路径

- 高层配置：`src/JYPPX.TensorRtSharp/TensorRtBuilderConfig.cs`。
- TRT11 diagnostics：`src/JYPPX.TensorRtSharp/TensorRtBuilderConfig.Trt11Diagnostics.cs`。
- 工具报告：`src/JYPPX.TensorRtSharp.Tools/OnnxEngineBuildDiagnostics.cs`。
- TensorRtExec gap list：`applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json`。

## proof 边界

Builder config readback 是配置确认，不是 runtime proof。`WorkspaceBytes`、memory pool 读回、profiling verbosity 和 flags 应作为 build/report evidence 使用；它们不能证明 engine 已经在外部 clean consumer 中完成真实 TensorRT inference。

不能晋级 proof 的典型替代项：

- build-only report。
- parse-only report。
- TensorRtExec GUI screenshot。
- load-engine readonly diagnostics。
- local feed package consumer。

## 配图建议

- 一张 option layering 图：CLI/GUI options -> OnnxEngineBuildOptions -> TensorRtBuilderConfig -> report readback。
- 一张边界图：build/report evidence 与 package-consumer-runtime proof 分开。

## 下一步

继续增加安全的 readback 字段和报告一致性测试，同时把真实 proof 留给 clean external consumer runtime smoke 与 owner input validator。
