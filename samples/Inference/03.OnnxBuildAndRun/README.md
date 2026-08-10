# ONNX Build And Run

English | [简体中文](README.zh-CN.md)

This sample consumes the public `JYPPX.TensorRT.CSharp.API` package to parse an ONNX model, build a serialized TensorRT engine, deserialize it, execute one inference, and emit a structured JSON report.

Offline help does not load CUDA or TensorRT:

```powershell
dotnet run --project .\samples\Inference\03.OnnxBuildAndRun -- --help
```

Run the deterministic synthetic Identity model:

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"
dotnet run --project .\samples\Inference\03.OnnxBuildAndRun -- --synthetic --tensor-rt-line 10 --output-json .\artifacts\onnx-build-and-run.json
```

Run an external ONNX model with one FP32 input and output:

```powershell
dotnet run --project .\samples\Inference\03.OnnxBuildAndRun -- --model .\models\model.onnx --input-shape 1x4 --output-json .\artifacts\onnx-build-and-run.json
```

The synthetic path generates a stable `1x4` Identity ONNX file in the system temporary directory. A successful report has `status=passed`, `proofClassification=synthetic-input-runtime`, `execution.enqueueCount=1`, and `output.identityOutputMatch=true`. It is a deterministic runtime smoke, not real-model or package-consumer-runtime proof.
