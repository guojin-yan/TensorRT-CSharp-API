# OnnxToEngine 与 trtexec parity：把模型转换做成可审计流程

`samples/OnnxToEngine` 的目标不是做一个最小 demo，而是把 ONNX 到 TensorRT engine 的转换流程做清楚。对于熟悉 NVIDIA 官方工具的用户来说，它应该尽量贴近 `trtexec` 的模型转换能力；对于 .NET 用户来说，它又要比直接调用外部命令更容易集成、记录和排障。

## 适合谁阅读

- 正在把 ONNX 模型转换为 TensorRT engine 的 .NET 开发者。
- 熟悉官方 `trtexec`，希望理解 TensorRtExec / OnnxToEngine 差距的用户。
- 需要审查 build-only、real-model-runtime、package-consumer-runtime proof 边界的维护者。

## 两条路径的关系

项目中有两条相关路径：

- `samples/OnnxToEngine`：偏教程和样例，适合新用户理解模型转换。
- `applications/TensorRtExec`：偏正式工具，目标是 CLI + WinForms 复刻官方 `trtexec` 的主要能力。

两者共享同一个原则：build-only 只是构建证据，不是 runtime proof。

## 常见转换命令

```powershell
dotnet run --project .\samples\OnnxToEngine -- `
  --onnx .\models\model.onnx `
  --engine .\models\model.plan `
  --fp16
```

TensorRtExec 的 trtexec-like 形态：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model.plan `
  --minShapes images:1x3x640x640 `
  --optShapes images:1x3x640x640 `
  --maxShapes images:1x3x640x640 `
  --fp16 `
  --workspace 1024 `
  --buildOnly `
  --exportReport .\artifacts\model-build-report.json
```

## Parity 关注点

当前重点不是宣称已经复刻所有官方行为，而是把差距透明化：

- ONNX 输入和 engine 输出是否完整。
- dynamic shape profile 是否可表达。
- FP16/INT8/workspace/timing cache 是否只是参数解析、report，还是有真实生命周期 proof。
- layer dump、profile、verbose log 是否只是 diagnostic metadata。
- WinForms 是否与 CLI 选项对齐。

发布候选 gap list 位于：

```text
applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json
```

## Proof 边界

OnnxToEngine 和 TensorRtExec 都不能单独证明模型输出正确。它们能生成 engine、report、sidecar、日志，但 real-model-runtime proof 需要样例 runner 读取真实输入并产生 `Passed=True`，package-consumer-runtime proof 还需要外部 consumer 使用公开包。

## 配图建议

- 官方 trtexec 参数到 TensorRtExec 参数的对照表。
- ONNX -> engine -> sample run -> proof validator 的流程图。
- build report JSON 示例截图。

## 下一步

下一阶段应优先实现 `--loadEngine` 只读诊断、builder config readback 和 WinForms 参数 parity，再把 OnnxToEngine 文档与 TensorRtExec gap list 互相链接。
