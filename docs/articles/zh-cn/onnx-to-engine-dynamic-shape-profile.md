# ONNX 到 Engine 动态 Shape Profile

TensorRT 动态 shape 需要 min/opt/max profile。`applications/OnnxToEngine` 用内置 identity ONNX 演示 profile 绑定，`applications/TensorRtExec` 则用 trtexec-like 参数处理外部 ONNX。

## 参数格式

```text
input:1x3x224x224
```

多个输入可用逗号分隔：

```text
image:1x3x640x640,scale:1x2
```

## TensorRtExec 示例

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\dynamic.onnx `
  --saveEngine .\models\dynamic.plan `
  --minShapes images:1x3x320x320 `
  --optShapes images:1x3x640x640 `
  --maxShapes images:4x3x960x960 `
  --fp16 `
  --buildOnly `
  --exportReport .\models\dynamic-build-report.json
```

报告应保留 shape profile、workspace、precision、normalized command 和 SHA256，方便后续和真实样例运行记录关联。
该命令仍然是 build-only 路径；它不能证明输出 tensor 语义、真实输入质量或 post-publish clean consumer 状态。

## 常见错误

- 输入名与 ONNX graph 不一致；
- min/opt/max rank 不一致；
- opt 不在 min/max 范围内；
- batch 或 spatial dimension 超过模型导出边界；
- dynamic axis 缺少 profile；
- 真实样例运行时使用的输入 shape 不在 profile 范围内。

## 与样例的关系

- `samples/Inference/02.DynamicShapes` 用于理解 profile 和 binding；
- `applications/OnnxToEngine` 用于最小 ONNX round-trip；
- `applications/TensorRtExec` 用于外部模型 build/report；
- `applications/YoloVision` 和 `samples/ComputerVision/01.Classification` 用于模型任务语义和真实输入。

## 边界说明

动态 shape profile 配置成功不等于真实模型 proof。profile 是构建条件；真实 runtime proof 还需要输入 tensor、输出语义、日志、hash 和 validator。build-only 不是 inference proof，post-publish proof 仍必须由公开包源 clean consumer 产生。
