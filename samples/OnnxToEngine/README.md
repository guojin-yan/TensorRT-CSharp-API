# OnnxToEngine

Status: redirected to the runnable smoke runner.

The runnable ONNX-to-engine sample is:

```text
samples/OnnxToEngineSmokeRunner
```

That runner generates a minimal dynamic ONNX identity model in process, so it does not require external model files. It validates parser creation, in-memory ONNX parsing, optimization profile setup, serialized engine build, runtime deserialize, tensor binding, enqueue, and output readback.

Run from the repository root with development probing enabled:

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"
```

Run:

```powershell
dotnet .\samples\OnnxToEngineSmokeRunner\bin\Debug\net8.0\OnnxToEngineSmokeRunner.dll --tensor-rt-line 10
```

Expected evidence includes:

- `Parsed=True`
- `ProfileIndex=0`
- `EngineFileRoundTrip=True`
- `BindingReport Ready=True`
- `OutputMatch=True`

This directory remains as the user-facing topic name for future asset-based ONNX conversion walkthroughs. It intentionally has no duplicate project file today.

