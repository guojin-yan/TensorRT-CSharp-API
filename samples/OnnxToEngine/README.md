# OnnxToEngine

Status: redirected to the runnable smoke runner.

The runnable ONNX-to-engine sample is:

```text
samples/OnnxToEngineSmokeRunner
```

That runner generates a minimal dynamic ONNX identity model in process, so it does not require external model files. It validates parser creation, in-memory ONNX parsing, optimization profile setup, serialized engine build, runtime deserialize, tensor binding, enqueue, and output readback.

Run:

```powershell
$env:JYPPX_NATIVE_BRIDGE_PATH = "E:\TensorRtSharp\TensorRtSharp4.0\build-out\win-x64-trt10-cuda11-release\bin\Release"
$env:JYPPX_TENSORRT_ROOT = "E:\TensorRtSharp\TensorRtSharp4.0\third_party\nvidia\TensorRT-10.11.0.33-cuda 11.8"
$env:JYPPX_CUDA_ROOT = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
dotnet .\samples\OnnxToEngineSmokeRunner\bin\Debug\net8.0\OnnxToEngineSmokeRunner.dll --tensor-rt-line 10
```

Expected evidence includes:

- `Parsed=True`
- `ProfileIndex=0`
- `EngineFileRoundTrip=True`
- `BindingReport Ready=True`
- `OutputMatch=True`

This directory remains as the user-facing topic name for future asset-based ONNX conversion walkthroughs. It intentionally has no duplicate project file today.
