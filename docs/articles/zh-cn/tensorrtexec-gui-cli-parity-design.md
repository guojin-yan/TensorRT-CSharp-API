# TensorRtExec GUI/CLI Parity 设计：一个选项如何同时进入 CLI、WinForms 和 Preview

## 适用读者

这篇文章适合使用 `applications/TensorRtExec` 的用户、维护 CLI/WinForms 工具的开发者，以及想理解项目如何复刻官方 trtexec 模型转换体验的读者。

## 解决问题

官方 `trtexec` 的参数很多，直接照搬到 C# 工具容易出现 CLI 支持了、GUI 没有；GUI 有控件、命令预览不一致；report 里有字段、README 没解释。TensorRtExec 的设计目标是把 shared options、CLI parser、WinForms 控件、command preview 和 report 字段保持同步。

## Parity 的核心思路

项目用 `applications/TensorRtExec/Core/TensorRtExecOptions.cs` 表达共享选项，用 `Console/TensorRtExecCommand.cs` 做 CLI 入口，用 `WinForms/MainForm.cs` 做桌面配置。`artifacts/final-release/tensor-rt-exec-gui-cli-parity-checklist.json` 则把 `onnx`、`save-engine`、`load-engine`、shape profiles、precision、INT8 calibration、workspace、timing cache、plugins、profiling、layer info、runtime benchmark、binding output、safety/cache policy、device/DLA 等能力列成 machine-readable checklist。

## 用户如何使用

```powershell
dotnet run --project .\applications\TensorRtExec -- --help
dotnet run --project .\applications\TensorRtExec -- --onnx .\models\model.onnx --saveEngine .\models\model.plan --buildOnly --exportReport .\artifacts\model-build-report.json
```

GUI 用户可以通过 WinForms 配置同类选项，并从 command preview 中复制命令。CLI 和 GUI 的价值是降低 ONNX-to-engine build 的使用门槛，而不是绕过 proof 流程。

## 设计边界

TensorRtExec 可以生成 build/precheck report、profile/layer info 等诊断数据。它有助于模型转换和问题定位，但 report 不等于 inference output correctness，也不等于 public package consumer runtime proof。尤其是 build-only 和 parse-only 状态，只能说明 engine build 或参数解析路径，不代表真实 runtime enqueue 已验证。

## 边界说明

TensorRtExec parity checklist、GUI screenshot、build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、sidecar、readonly diagnostics 都不是 runtime proof。要形成 release proof，必须回到仓库外 clean consumer 与 strict validator。

## 下一步

工具使用者可以继续阅读 `onnx-to-engine-trtexec-conversion-guide.md`。维护者下一步可以把更多 trtexec 选项纳入 shared options，但任何 runtime proof 声明都必须等待真实模型或 clean consumer proof。
