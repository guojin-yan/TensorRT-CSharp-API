# OnnxToEngine 与 trtexec-like 转换能力路线图

`applications/OnnxToEngine` 是用户友好的 ONNX 到 TensorRT engine walkthrough。它正在向官方 `trtexec` 的模型转换能力靠拢，但当前不能宣称全量等价。本文记录目标能力、已覆盖边界和后续实装路线。

## 当前目标

- 保持一个简单、可读、可运行的 sample。
- 支持常见 ONNX build-only 和 identity synthetic runtime。
- 将 trtexec-like 参数解析、normalized command、report、proof boundary 写入可审计输出。
- 与 `applications/TensorRtExec` 形成分层：sample 侧更适合教学，application 侧更适合完整工具化。

## 能力矩阵

| 能力 | OnnxToEngine 当前边界 | 后续目标 |
| --- | --- | --- |
| ONNX path / saveEngine | 已支持 | 保持 |
| min/opt/max shapes | 已支持 | 多 profile 更完整 |
| fp16 | 已解析并在可用时应用 | 增加更多 precision 诊断 |
| int8/calib | report/diagnostic 边界 | owner-provided calibration 真实路径 |
| workspace/memory | `--workspace` 与已知 `--memPoolSize` pool 已在真实 build 中应用并 read back | 继续补 compatible-host owner build record；不把 builder-config readback 晋级为 runtime proof |
| timing cache | 路径、导入/导出和 report 边界 | 在兼容主机补真实构建日志、owner review 和模型级输出校验 |
| plugin library | 当前谨慎处理 | 等待 load/register/deregister ownership 设计 |
| dumpLayerInfo/exportLayerInfo | implemented-inspector-readback | 真实 build/load-engine 通过 copied `TensorRtEngineInspector` 文本支持日志和 UTF-8 文件导出；dry-run/缺依赖保持 report-only，后续只需补兼容主机和 owner 的真实 engine 证据 |
| exportTimes/exportProfile/exportOutput | artifact proof-boundary 字段 | 真实 runtime 输出需 sample runner 证明 |
| loadEngine | readonly diagnostics | 不升级为 enqueue runtime proof |

## 参数策略

参数分为三类：

- `applied`：已真正影响 builder/runtime 行为，例如基础 profile、部分 builder config。
- `parse-only`：CLI/README/report 接住，但不宣称等价官方行为，例如高级 DLA/safety/cache/precision 组合。
- `probe-only`：只读能力或依赖诊断，例如 load-engine readback、dependency probing。

这三个状态必须写进 report，不允许用“参数已存在”推导“功能已完整实现”。

## 示例命令

```powershell
dotnet run --project .\applications\OnnxToEngine -- `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model.plan `
  --minShapes input:1x3x640x640 `
  --optShapes input:1x3x640x640 `
  --maxShapes input:4x3x640x640 `
  --fp16 `
  --workspace 1024 `
  --builderOptimizationLevel 4 `
  --exportReport .\models\model-build-report.json `
  --buildOnly
```

该命令最多形成 build-only evidence。It is not runtime proof。它不能证明模型输出正确，也不能替代 YoloVision、Classification 或 package consumer proof。

## 与官方 trtexec 的差异口径

对外文档可以写“trtexec-like option parsing”和“向 trtexec conversion parity 靠拢”，但不能写“复刻官方 trtexec 全部功能”，除非：

- 参数已从 parse-only 变成真实 applied。
- native TensorRT 行为已实现。
- report 能显示真实影响。
- smoke 或 owner real model runtime 已验证。
- quality gate 覆盖该能力。

## 下一步工作

1. 将参数矩阵从 README 提炼为机器可读 JSON。
2. 区分 builder、runtime、deployment、diagnostic、proof-boundary 参数。
3. 扩展 `OptionImplementationStatus`，避免高级参数被误写成 applied。
4. 衔接 `applications/TensorRtExec` 的 console/WinForms field map。
5. 为真实外部 ONNX 模型准备 owner evidence record，而不是在 sample 中伪造 proof。
