# OnnxToEngine

Status: runnable common sample.

This sample generates a minimal dynamic ONNX identity model in process, so it does not require external model files. It demonstrates:

- ONNX parser creation
- in-memory ONNX parsing
- optimization profile setup
- serialized engine build
- runtime deserialize
- tensor binding
- enqueue and output readback

Run from the repository root with development probing enabled:

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"
```

Run:

```powershell
dotnet .\samples\OnnxToEngine\bin\Debug\net8.0\OnnxToEngine.dll --tensor-rt-line 10
```

Expected evidence includes:

- `Parsed=True`
- `ProfileIndex=0`
- `EngineFileRoundTrip=True`
- `BindingReport Ready=True`
- `OutputMatch=True`

