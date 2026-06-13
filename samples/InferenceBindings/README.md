# InferenceBindings

Status: runnable common sample.

This sample builds a small explicit-batch identity network directly in C# and demonstrates:

- optimization profile setup
- `TensorRtInferenceBindings` input/output workflow
- host-to-device copy
- tensor binding
- enqueue and output readback

Run from the repository root with development probing enabled:

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"
```

Run:

```powershell
dotnet .\samples\InferenceBindings\bin\Debug\net8.0\InferenceBindings.dll --tensor-rt-line 10 --batch 2
```

Expected evidence includes:

- `BindingReport Ready=True`
- `Readiness Ready=True Bound=True`
- `Execution ... OutputMatch=True`
- `InferenceBindings Passed=True`
