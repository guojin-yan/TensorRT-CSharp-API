# Refitted Plan Lifecycle

English | [简体中文](README.zh-CN.md)

This sample consumes the public `JYPPX.TensorRT.CSharp.API` package and demonstrates the complete persisted-refit lifecycle: build a refittable engine, save and deserialize the baseline plan, update ONNX initializers through `TensorRtOnnxParserRefitter`, commit the refit, serialize the updated engine, then reload that plan in a new runtime and verify its output.

Offline help does not load CUDA or TensorRT:

```powershell
dotnet run --project .\samples\Inference\04.RefittedPlan -- --help
```

Run the deterministic synthetic workflow on TensorRT 10:

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"
dotnet run --project .\samples\Inference\04.RefittedPlan -- --synthetic --tensor-rt-line 10 --plan .\artifacts\refitted-scale.engine --output-json .\artifacts\refitted-plan.json
```

The generated baseline model multiplies a `1x4` input by initializer `scale=[1,1,1,1]`; the refit model changes only that initializer to `[2,2,2,2]`. A passing report requires the baseline output to equal the input, both post-refit outputs to equal `input * 2`, the in-memory output to change, and the reloaded plan output to remain unchanged from the refitted engine.

For external models, pass structurally matching baseline and refit ONNX files:

```powershell
dotnet run --project .\samples\Inference\04.RefittedPlan -- --baseline-model .\models\baseline.onnx --refit-model .\models\updated.onnx --plan .\artifacts\updated.engine
```

The compact inference verifier expects FP32 tensors named `input` and `output`, each with four elements. The JSON uses `proofClassification=synthetic-input-runtime`; it is a source-tree runtime smoke, not a real-model or `package-consumer-runtime` release proof. Dispose the parser-refitter before the refitter, the refitter before its engine, and callback/logger owners after every TensorRT borrower has detached.
