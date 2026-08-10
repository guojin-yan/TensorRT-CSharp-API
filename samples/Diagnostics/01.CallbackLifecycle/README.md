# Callback Lifecycle

English | [简体中文](README.zh-CN.md)

This sample consumes the public `JYPPX.TensorRT.CSharp.API` package and runs Logger, ProgressMonitor, Profiler, and DebugListener in one small TensorRT workflow. Its focus is ownership: callback owners stay alive while TensorRT borrows them, every detachable callback is explicitly cleared, and only copied metadata reaches managed DebugListener code.

Offline help does not load CUDA or TensorRT:

```powershell
dotnet run --project .\samples\Diagnostics\01.CallbackLifecycle -- --help
```

Run the synthetic TensorRT 10 workflow:

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"
dotnet run --project .\samples\Diagnostics\01.CallbackLifecycle -- --tensor-rt-line 10 --output-json .\artifacts\callback-lifecycle.json
```

The sample builds a `1x1` convolution while ProgressMonitor is attached, clears the monitor after the build, then attaches Profiler and DebugListener to an execution context. It executes one `1x1x2x2` FP32 tensor, synchronizes the CUDA stream, copies the output, snapshots callback state, and clears both runtime callbacks before disposing their owners.

The JSON report includes callback counts, failure counts, attach/detach state, copied debug tensor metadata, `borrowedPointerExposed=false`, output equality, and the lifecycle order. Logger is intentionally kept alive through every builder/runtime borrower. The report uses `proofClassification=synthetic-input-runtime`; it is not external-model or `package-consumer-runtime` release proof.

TensorRT 8 is intentionally rejected because the combined workflow requires ProgressMonitor and DebugListener APIs available on TensorRT 10 and 11.
