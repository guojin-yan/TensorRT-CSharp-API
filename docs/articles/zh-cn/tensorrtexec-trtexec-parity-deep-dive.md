# TensorRtExec 与官方 trtexec Parity 深入

TensorRtExec 的目标是提供一个 .NET 生态里的 trtexec-like 应用：既能命令行使用，也能通过 WinForms 页面配置 ONNX-to-engine workflow。本文解释 parity 的分层：参数解析、报告输出、engine build、runtime 执行和 proof 晋级。

## 适用读者

适合熟悉 NVIDIA `trtexec`、希望在 C# 项目中获得类似转换工具的用户，也适合维护 `applications/TensorRtExec` 的开发者。

## 解决问题

直接复制官方 `trtexec` 的全部行为并不现实，尤其是 plugin、DLA、calibration、timing cache、profile dump 和 runtime measurement。本文解决“哪些已实现、哪些 parse-only、哪些需要真实 runtime proof”的沟通问题。

## 背景与场景

用户通常先用官方 `trtexec` 验证模型，再希望在 C# 项目中自动化同样流程。TensorRtExec 把 CLI、WinForms 和 tools service 聚合起来，但必须避免把 GUI 截图或 build report 写成推理结果正确。

## 操作路径

1. 用 `tensorrtexec-cli-parameter-map.md` 对齐参数状态。
2. 用 `TensorRtExecOptions` 和 `TensorRtExecService` 保持 CLI/GUI 共用实现。
3. 用 dry-run 生成 normalized command 和 SHA256。
4. 用 build-only 生成 engine/report。
5. 用真实样例 runner 或 package consumer proof record 采集 runtime proof。

## 代码与文件入口

- `applications/TensorRtExec/README.md`
- `applications/TensorRtExec/tensor-rt-exec-feature-matrix.json`
- `src/JYPPX.TensorRtSharp.Tools`
- `samples/OnnxToEngine/trtexec-parity-matrix.json`
- `tests/JYPPX.ProjectQuality.Tests/TensorRtExecApplicationTests.cs`

## 边界说明

build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 都不是 runtime proof。parity 文档只能说明工具能力和状态，真实模型 proof 还需要模型资产、输入、日志、hash 和 validator。

## 下一步

下一步继续补 TensorRtExec GUI user guide、参数状态矩阵和真实模型 build/run 案例，把 parse-only 与 implemented 状态逐步缩小。
