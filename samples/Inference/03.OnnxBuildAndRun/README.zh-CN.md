# ONNX 构建与运行

[English](README.md) | 简体中文

本示例通过公开的 `JYPPX.TensorRT.CSharp.API` 包解析 ONNX、构建序列化 TensorRT engine、反序列化、执行一次推理，并输出结构化 JSON 报告。

离线帮助不会加载 CUDA 或 TensorRT：

```powershell
dotnet run --project .\samples\Inference\03.OnnxBuildAndRun -- --help
```

运行确定性的 synthetic Identity 模型：

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"
dotnet run --project .\samples\Inference\03.OnnxBuildAndRun -- --synthetic --tensor-rt-line 10 --output-json .\artifacts\onnx-build-and-run.json
```

运行单 FP32 输入、单 FP32 输出的外部 ONNX 模型：

```powershell
dotnet run --project .\samples\Inference\03.OnnxBuildAndRun -- --model .\models\model.onnx --input-shape 1x4 --output-json .\artifacts\onnx-build-and-run.json
```

synthetic 路径会在系统临时目录生成固定的 `1x4` Identity ONNX。成功报告包含 `status=passed`、`proofClassification=synthetic-input-runtime`、`execution.enqueueCount=1` 和 `output.identityOutputMatch=true`。这属于确定性运行 smoke，不是外部真实模型或 package-consumer-runtime 证明。
