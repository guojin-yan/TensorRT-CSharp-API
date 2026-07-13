# TensorRtExec Console + WinForms 应用路线图

`applications/TensorRtExec` 是面向最终用户的 trtexec-like 应用路线。目标是同时支持控制台命令行和 Windows WinForms 页面，让用户可以用命令行批处理，也可以在桌面界面中选择 ONNX、profile、precision、report 和输出路径。

## 产品定位

| 层级 | 目录 | 定位 |
| --- | --- | --- |
| 教学 sample | `samples/OnnxToEngine` | 讲清楚 ONNX 到 engine 的最小转换路径 |
| 视觉 sample | `samples/YoloVision` | 讲清楚真实视觉模型推理和后处理 |
| 用户应用 | `applications/TensorRtExec` | 提供 trtexec-like console/WinForms 工具体验 |

TensorRtExec 应用不应该只是一层 README。它需要逐步承载：

- CLI parser。
- normalized command preview。
- build-only report。
- engine readback diagnostics。
- timing cache / profile / layer info 输出。
- WinForms field map。
- GUI command preview 与 CLI parity。
- proof boundary 字段。

## Console 路线

Console 入口应覆盖：

- `--onnx`
- `--saveEngine`
- `--loadEngine`
- `--minShapes` / `--optShapes` / `--maxShapes`
- `--shapes` / `--inputShapes`
- `--fp16` / `--int8`
- `--workspace` / `--memPoolSize`
- `--builderOptimizationLevel`
- `--timingCache` / `--exportTimingCache`
- `--dumpLayerInfo` / `--exportLayerInfo`
- `--dumpProfile` / `--exportProfile`
- `--buildOnly`
- `--previewOnly`
- `--exportReport`

高级参数可以先进入 parse-only/report-only，但必须在 report 中明确 `TrtexecAlignmentStatus=parse-only` 或对应状态。

## WinForms 路线

WinForms 页面至少需要这些区域：

- 模型输入：ONNX path、engine output、load engine。
- Shape profile：input name、min/opt/max。
- Precision：FP32/FP16/INT8、calibration cache。
- Builder：workspace、optimization level、aux streams。
- Diagnostics：layer info、profile、timing cache、report。
- Command preview：实时生成 CLI 命令。
- Run panel：执行、stdout/stderr、artifact links。
- Boundary panel：显示 build-only/precheck/runtime proof 分类。

GUI 截图、command preview、dry-run 和 build report 都不是 runtime proof。只有真实模型、真实输入、日志、hash、host metadata 和 validator 才能形成 real case evidence candidate。

## 质量门

后续质量测试应至少覆盖：

- CLI 参数和 WinForms 字段一一对应。
- GUI command preview 与 normalized CLI 一致。
- parse-only 参数不会被写成 applied。
- report 明确 build-only/precheck/probe-only 边界。
- README 与 docs 不宣称 package-consumer-runtime。
- 旧 detection-only 命名已经被 YoloVision 取代；公开应用/样例入口继续使用 YoloVision。

## 对外文章角度

TensorRtExec 可以拆成多篇文章：

1. TensorRtExec CLI 快速上手。
2. TensorRtExec 参数矩阵与 trtexec 差异。
3. TensorRtExec WinForms 桌面转换流程。
4. Timing cache 与 profile report。
5. 为什么 build report 不是 runtime proof。
6. 与 YoloVision/Classification 真实样例 proof 的衔接。

## 发布边界

TensorRtExec 是工具应用路线，不是发布授权。它生成的 report、sidecar、截图、dry-run、build-only 输出都不能替代公开包 proof、package-consumer-runtime、Owner approval 或 post-publish verification。
