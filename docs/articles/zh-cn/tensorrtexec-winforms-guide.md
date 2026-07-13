# TensorRtExec WinForms 使用指南

TensorRtExec 的 WinForms 入口面向 Windows 桌面用户。它把 CLI 参数投影成控件，但底层仍调用同一个 `TensorRtExecService`，因此 GUI 与 CLI 的 normalized command、report 和 proof boundary 保持一致。

## 启动

```powershell
dotnet run --project .\applications\TensorRtExec -- --ui
```

如果直接运行应用且没有 CLI 参数，也会进入桌面界面。

## 推荐流程

1. 选择 ONNX 文件。
2. 选择 Save Engine 输出路径。
3. 选择 TensorRT line。
4. 填写 dynamic shape profile。
5. 选择 precision，例如 FP16。
6. 填写 workspace。
7. 选择 report 输出路径。
8. 先勾选 DryRun/PreviewOnly 做预检。
9. 再切到 BuildOnly 生成 engine 和 build report。

## 控件与参数关系

| 控件 | 参数 | 说明 |
| --- | --- | --- |
| ONNX | `--onnx` | 外部模型 |
| Save Engine | `--saveEngine` | engine 输出 |
| Load Engine | `--loadEngine` | engine preflight |
| Precision | `--fp16 --int8 --bf16 --noTF32` | precision intent |
| Shapes | `--minShapes --optShapes --maxShapes` | dynamic profile |
| Builder | `--builderOptimizationLevel --maxAuxStreams` | builder config |
| Runtime | `--iterations --warmUp --duration --streams` | runtime/report |
| Output | `--exportOutput --exportTimes --exportProfile` | runtime artifact |
| Evidence | `--exportReport --evidenceSidecar` | report/sidecar |

## 输出解读

GUI 日志会显示：

- `TensorRtExec ReportPath=`；
- `TensorRtExec ProofClassification=`；
- `TensorRtExec NormalizedCommandSha256=`；
- `BuildEvidenceOnly=`；
- `DryRun=`。

这些字段用于审计 GUI 到 CLI 的一致性。它们不代表 public package proof 或 post-publish proof。

## 边界说明

WinForms 是用户入口，不是发布按钮。它不会执行 NuGet 真实发布命令，不会注册/注销 plugin library，不会绕过 release evidence classification audit。GUI 生成的 build-only、precheck、sidecar-only 报告不能作为 release close approval。Boundary keywords: not public package proof, not post-publish proof, not package push, not release close approval.
