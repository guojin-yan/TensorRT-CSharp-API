# TensorRtExec WinForms 截图式操作 walkthrough

## 适用读者

本文面向希望用 `applications/TensorRtExec` WinForms 界面完成 ONNX 到 TensorRT engine 构建的用户，尤其是更习惯用界面选择模型、profile、precision 和输出目录的 Windows 开发者。

## 解决问题

GUI 工具容易被误解成“一键 proof”。本文按界面区域解释输入、配置、执行、报告和错误信息，让用户知道 WinForms screenshot 只能辅助排查；build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 都不是 runtime proof。

## 背景与场景

TensorRtExec 同时覆盖 CLI 和 WinForms。CLI 适合自动化，WinForms 适合初学者和人工排查。二者都应该输出一致的参数、report schema 和错误边界，避免 GUI 成为不可复现的黑盒。

## 操作路径

1. 在输入区域选择 ONNX 文件、输出 engine 路径和目标 TensorRT/CUDA runtime。
2. 在 profile 区域填写 min/opt/max shape，确认 dynamic shape 与模型输入名一致。
3. 在 precision 区域选择 FP32、FP16 或 INT8，并记录校准资产是否真实存在。
4. 点击构建后保存日志、report JSON、命令投影和错误码。
5. 用 CLI parity 字段复查 GUI 参数是否能还原为命令行。

## 代码与文件入口

- `applications/TensorRtExec`：CLI 与 WinForms 工具源码。
- `applications/TensorRtExec/tensor-rt-exec-feature-matrix.json`：功能矩阵。
- `docs/articles/zh-cn/tensorrtexec-winforms-guide.md`：WinForms 基础说明。
- `docs/articles/zh-cn/tensorrtexec-gui-user-guide.md`：GUI 用户指南。
- `docs/articles/zh-cn/tensorrt-exec-trtexec-parity-matrix.md`：CLI parity 矩阵说明。

## 图示建议

建议准备 4 张截图：模型选择区、profile 设置区、构建日志区和 report 输出区。每张截图只标注字段含义，不把截图标注为 proof。

## 边界说明

WinForms screenshot、TensorRtExec report 和 OnnxToEngine report 可以解释构建过程，但不能单独证明 package consumer runtime 可用。build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、YoloVision matrix 和 readonly diagnostics 都必须保持非 proof 边界。

## 下一步

下一轮应补充 WinForms 字段到 CLI 参数的映射表，并为 report JSON 增加截图编号或 UI session id，方便用户复盘。
