# TensorRtExec GUI 与 CLI 字段映射

## 适用读者

本文面向同时使用 TensorRtExec WinForms 和 CLI 的用户，帮助他们把界面选项还原为可复制命令，并理解 report 字段如何保持一致。

## 解决问题

GUI 操作如果不能还原为 CLI 命令，就难以复查、自动化和写入 evidence。本文梳理模型路径、engine 输出、profile、precision、workspace、timing cache 和 report 输出等字段映射，并强调 build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 都不是 runtime proof。

## 背景与场景

TensorRtExec 的设计目标之一是 GUI/CLI parity。WinForms 降低上手门槛，CLI 承接 CI 和脚本化。字段映射表可以让用户在 GUI 中尝试参数，再把同一配置保存为命令行，减少“界面能跑、脚本不能跑”的差异。

## 操作路径

1. 将 GUI 的 ONNX 选择框映射到 CLI 的 model/input 参数。
2. 将 engine 输出路径映射到 saveEngine 或 output engine 参数。
3. 将 min/opt/max shape 输入映射到 profile 参数，并保留 input name。
4. 将 precision、workspace、timing cache 和 tactic source 映射到 builder config 参数。
5. 将 report path、verbosity 和 boundary 字段同时写入 CLI 与 GUI report。

## 代码与文件入口

- `applications/TensorRtExec`：CLI 与 WinForms 实现。
- `applications/TensorRtExec/tensor-rt-exec-feature-matrix.json`：GUI/CLI 能力矩阵。
- `docs/articles/zh-cn/tensorrtexec-gui-user-guide.md`：包含真实配置页和成功结果页的 WinForms 完整教程。
- `docs/articles/zh-cn/tensorrtexec-report-json-schema-snapshot.md`：report snapshot 建议。
- `docs/articles/zh-cn/tensorrt-exec-trtexec-parity-matrix.md`：trtexec parity 说明。

## 图示建议

建议用三列表：GUI label、CLI option、report field。每行标注是否影响 engine、是否影响 runtime、是否仅用于 diagnostics。

## 边界说明

字段映射提升可复现性，但不改变 proof 边界。TensorRtExec report、OnnxToEngine report、YoloVision matrix、readonly diagnostics、build-only、dry-run、template、local feed、ProjectReference 和 direct `.nupkg` 仍不能替代 runtime proof。

## 下一步

下一轮应把字段映射表机器可读化，或在质量测试中抽查 GUI/CLI/report 三者是否共享同一字段名与边界描述。
