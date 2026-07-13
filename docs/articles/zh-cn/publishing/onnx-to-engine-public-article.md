# ONNX 到 Engine：把模型转换做成可审计流程

ONNX 到 TensorRT engine 的转换是 TensorRtSharp4.0 最容易被用户感知的能力。`samples/OnnxToEngine/Program.cs` 提供了面向样例的转换入口，`applications/TensorRtExec` 则承担更完整的 trtexec-like 参数、报告和 evidence sidecar。

## 适合

- 想把 ONNX 模型转成 TensorRT engine 的 C# 用户。
- 需要比较 OnnxToEngine 与官方 `trtexec` 参数覆盖的人。
- 准备为 YoloVision 或自有模型生成真实资产 proof 的维护者。

## 基本流程

推荐先用 TensorRtExec 或 OnnxToEngine 生成 engine 和报告：

```powershell
dotnet run --project .\samples\OnnxToEngine\OnnxToEngine.csproj -- --onnx <model.onnx> --engine <model.engine>
dotnet run --project .\applications\TensorRtExec\TensorRtExec.csproj -- --onnx <model.onnx> --saveEngine <model.engine> --fp16
```

关键产物包括：

```text
samples/OnnxToEngine/Program.cs
applications/TensorRtExec/Core/TensorRtExecReport.cs
applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json
applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json
```

这些报告适合进入文章、排障和候选矩阵，但它们本身仍然属于 build/report 层。

## 与 YoloVision 的关系

YoloVision 负责把真实模型资产、输入样例、输出 schema 和任务 metadata 组织起来。OnnxToEngine 负责转换，TensorRtExec 负责更接近 `trtexec` 的参数和报告，YoloVision 则把 detection、classification、segmentation、OBB、pose、semantic segmentation 等任务串成可复核案例。

早期过窄的检测样例命名已不再适合作为项目名称口径；新的公开材料应使用 `samples/YoloVision`，避免把项目误解为只支持 detection。

## proof 边界

ONNX 转换成功是必要条件，但不是 package-consumer-runtime proof。build-only、dry-run、template、local feed、ProjectReference 和 direct `.nupkg` 不能证明公开包可被用户消费。真实 proof 需要 owner 提供外部 clean consumer 或真实模型运行日志、SHA256、host metadata、exitCode=0、passed=true 和严格 validator 输出。

## 配图建议

- 一张 ONNX -> engine -> runtime smoke 的流程图。
- 一张 TensorRtExec report JSON 摘要截图。
- 一张 YoloVision matrix 截图，展示模型系列和任务类型。

## 下一步

继续补齐 TensorRtExec 参数 parity、YoloVision 真实资产候选和 owner result input。转换链路稳定后，把真实运行结果写入 `artifacts/final-release/owner-external-proof-execution-result.input.json`，再通过 import、candidate、strict validator 和 release close bridge 逐级推进。
