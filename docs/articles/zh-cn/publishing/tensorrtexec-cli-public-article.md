# TensorRtExec CLI：用 C# 复刻 trtexec 模型转换体验

TensorRT 用户绕不开 `trtexec`。它是官方最常用的模型转换、engine 构建、profile 配置和快速诊断工具。问题是，当你的主项目是 C#/.NET 时，官方 `trtexec` 往往只是外部命令：参数、日志、报告、错误处理和应用内工作流都要再包一层。

TensorRtSharp4.0 中的 `applications/TensorRtExec` 目标就是做一个 C# 版 trtexec-like 应用：既能在控制台里以命令行方式使用，也能逐步扩展到 WinForms 图形界面；既服务 OnnxToEngine 的模型转换，也服务 YoloVision、文章案例和 release evidence ladder。

## 适合谁阅读

- 已经熟悉官方 `trtexec`，希望在 .NET 工具链中复用类似能力的用户。
- 需要把 ONNX -> TensorRT engine 构建流程做进内部平台、桌面软件或 CI 的团队。
- 想理解 TensorRtSharp4.0 如何区分 build report、sample runtime proof 和 package-consumer-runtime proof 的维护者。

## CLI 的定位

TensorRtExec 不是简单的 demo runner。它承担三件事：

1. 把常用 `trtexec` 参数映射到 C# CLI。
2. 输出机器可读 report 和 evidence sidecar，便于后续审计。
3. 为 WinForms 页面和更高层样例提供统一的转换后端。

仓库中已有 parity matrix：

- `applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json`
- `applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.md`
- `docs/articles/zh-cn/tensorrt-exec-trtexec-parity-matrix.md`

matrix 的意义是说明“哪些官方 trtexec 能力已经映射、哪些仍是后续工作”，它不是 runtime proof。

## 常见命令

最小 ONNX 构建：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model.plan `
  --buildOnly
```

带 FP16、workspace 和 report：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\yolov8-det.onnx `
  --saveEngine .\models\yolov8-det.plan `
  --fp16 `
  --workspace 1024 `
  --buildOnly `
  --exportReport .\artifacts\yolov8-det-build-report.json `
  --evidenceSidecar .\artifacts\yolov8-det-evidence.sidecar.json
```

动态 shape profile：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\dynamic.onnx `
  --saveEngine .\models\dynamic.plan `
  --minShapes images:1x3x320x320 `
  --optShapes images:1x3x640x640 `
  --maxShapes images:4x3x1280x1280 `
  --fp16 `
  --buildOnly
```

这些命令适合做模型转换教程、开发调试和内部验证。若要作为 package-consumer-runtime proof，还需要外部干净 consumer 使用公开包执行，不能用本仓库 ProjectReference 或 local feed 替代。

## 与 OnnxToEngine 的关系

`samples/OnnxToEngine` 更像“面向样例读者的模型转换路径”，重点是清晰、易懂、适合教程。`applications/TensorRtExec` 则更像正式工具，目标是覆盖官方 `trtexec` 的主要模型转换功能，并且同时支持 CLI 和 WinForms。

两者应该共享同一套原则：

- 参数语义尽量贴近官方 `trtexec`。
- 输出 report 可被 quality gate 和文章引用。
- build-only 不冒充 runtime proof。
- 动态 shape、FP16/INT8、workspace、profile、日志、导出文件都要可追踪。

## 当前 proof 边界

TensorRtExec 可以生成 build report，也可以辅助 YoloVision 真实资产案例，但它不能单独证明下面这些事：

- 不能证明模型精度正确。
- 不能证明 YoloVision 后处理结果正确。
- 不能证明公开 NuGet 包在外部 consumer 项目可用。
- 不能证明 package-consumer-runtime。

它能证明的是：在当前机器和当前依赖下，指定 ONNX build 命令按工具路径执行到了某个结果，并留下 report/evidence sidecar。更高等级的 proof 要继续交给 YoloVision validator、external runtime proof 和 package consumer proof。

## WinForms 方向

后续 WinForms 页面应围绕真实使用流设计，而不是只做参数堆叠：

- 左侧选择 ONNX、engine 输出路径、TensorRT line、精度和 profile。
- 中间提供参数分组：构建、shape、精度、日志、报告。
- 右侧显示命令预览、执行日志、report 摘要和错误诊断。
- 底部保留可复制命令，让 GUI 操作可以回到 CLI 自动化。

配图建议：CLI 输出截图、WinForms 参数页截图、build report JSON 摘要截图、trtexec parity matrix 截图。

## 下一步

TensorRtExec 的下一阶段应该继续补齐官方 `trtexec` conversion parity：更多 profile/precision/serialization 参数、错误诊断、report 字段和 GUI 参数同步。同时要让 OnnxToEngine 和 YoloVision 文章直接引用同一套命令，避免样例、工具和文档各自发散。
