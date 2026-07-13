# ONNX 到 Engine 快速开始

本文给出 TensorRtSharp4.0 中最短的 ONNX 到 TensorRT engine 路径。推荐从 `samples/OnnxToEngine` 的内置 identity ONNX 开始，再切到 `applications/TensorRtExec` 处理外部模型。

## 目标读者

- 第一次把 ONNX 模型转换为 TensorRT engine 的 .NET 用户。
- 需要比较 `samples/OnnxToEngine` 和 `applications/TensorRtExec` 适用场景的模型部署工程师。
- 想先生成 build/precheck report，再交给 Classification 或 YoloVision 做真实模型运行的维护者。

## 1. 内置 round-trip

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"

dotnet run --project .\samples\OnnxToEngine -- `
  --tensor-rt-line 11 `
  --batch 2
```

成功时应看到 parser、profile、serialized engine、runtime deserialize、binding 和 output match 的摘要。这证明最小样例路径可用，不证明任意外部模型都可运行。

## 2. 外部 ONNX dry-run

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model.plan `
  --previewOnly `
  --exportReport .\models\model-precheck.md
```

dry-run 只解析参数并输出 normalized command/report。它允许在没有 GPU、没有 ONNX 文件或没有 runtime 的环境里先做交接检查。

## 可复制命令

最短路径是先跑内置 identity round-trip，再转向外部 ONNX：

```powershell
dotnet run --project .\samples\OnnxToEngine -- --tensor-rt-line 11 --batch 2

dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model.plan `
  --minShapes input:1x3x224x224 `
  --optShapes input:4x3x224x224 `
  --maxShapes input:8x3x224x224 `
  --fp16 `
  --buildOnly `
  --exportReport .\models\model-build-report.json
```

对应代码和矩阵：

- `samples/OnnxToEngine`
- `applications/TensorRtExec`
- `samples/OnnxToEngine/trtexec-parity-matrix.json`
- `artifacts/user-acceptance/trtexec-option-coverage.md`

## 3. 外部 ONNX build-only

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model.plan `
  --minShapes input:1x3x224x224 `
  --optShapes input:4x3x224x224 `
  --maxShapes input:8x3x224x224 `
  --fp16 `
  --workspace 512MiB `
  --buildOnly `
  --exportReport .\models\model-build-report.json
```

## 4. 下一步选择

| 场景 | 推荐入口 |
| --- | --- |
| 分类模型 | `samples/Classification` |
| YOLO-family 模型 | `samples/YoloVision` |
| 动态 shape 练习 | `samples/DynamicShape` |
| 多流推理练习 | `samples/MultiStream` |
| GUI 构建 | `applications/TensorRtExec -- --ui` |

## 边界说明

ONNX 到 engine 快速开始只证明构建链路或最小样例链路。build-only 不是 inference proof，local feed 不是 public package proof，ProjectReference 不是 package-consumer-runtime proof，post-publish proof 必须来自公开包源上的 clean consumer 验证。Boundary keywords: not public package proof, not post-publish proof, not package push, not release close approval.

它同样是 not runtime proof：外部模型 build-only 报告不能证明任意输入绑定、真实输出、后处理质量或 real-model-runtime。需要继续进入 `samples/Classification` 或 `samples/YoloVision`，并补齐 sample-run evidence record。

## 截图与图示建议

- `samples/OnnxToEngine` 成功输出截图：parser、serialized engine、deserialize、output match。
- `applications/TensorRtExec` JSON/Markdown report 截图。
- ONNX build-only 到 YoloVision/Classification real-model-runtime 的证据流图。

## 下一步

- 需要 YOLO-family 模型时，继续阅读 `docs/articles/zh-cn/yolovision-sample-overview.md`。
- 需要 trtexec-like 参数时，继续阅读 `docs/articles/zh-cn/tensorrtexec-cli-parameter-map.md`。
- 需要确认参数覆盖时，检查 `samples/OnnxToEngine/trtexec-parity-matrix.json`。
