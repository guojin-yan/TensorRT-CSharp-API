# Refitted Plan 生命周期

[English](README.md) | 简体中文

本示例通过公开的 `JYPPX.TensorRT.CSharp.API` 包演示完整的持久化 refit 生命周期：构建可 refit engine，保存并从磁盘反序列化 baseline plan，通过 `TensorRtOnnxParserRefitter` 更新 ONNX initializer，提交 refit，再次序列化 engine，最后用新的 runtime 加载 refitted plan 并验证输出。

离线帮助不会加载 CUDA 或 TensorRT：

```powershell
dotnet run --project .\samples\Inference\04.RefittedPlan -- --help
```

在 TensorRT 10 上运行确定性 synthetic 流程：

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"
dotnet run --project .\samples\Inference\04.RefittedPlan -- --synthetic --tensor-rt-line 10 --plan .\artifacts\refitted-scale.engine --output-json .\artifacts\refitted-plan.json
```

生成的 baseline 模型使用 initializer `scale=[1,1,1,1]` 对 `1x4` 输入执行乘法；refit 模型只把该 initializer 改为 `[2,2,2,2]`。报告通过的条件是：baseline 输出等于输入，refit 后和磁盘重载后的输出都等于 `input * 2`，并且 refit 前后输出确实发生变化。

也可传入结构一致的外部 ONNX 模型：

```powershell
dotnet run --project .\samples\Inference\04.RefittedPlan -- --baseline-model .\models\baseline.onnx --refit-model .\models\updated.onnx --plan .\artifacts\updated.engine
```

当前紧凑推理验证器要求模型具有名为 `input` 和 `output` 的 FP32 tensor，元素数均为 4。JSON 使用 `proofClassification=synthetic-input-runtime`；它属于源码树运行 smoke，不是外部真实模型或 `package-consumer-runtime` 发布证明。释放顺序应为 parser-refitter、refitter、engine，Logger 等 owner 必须在所有 TensorRT borrower 解除借用后再释放。
