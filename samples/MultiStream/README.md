# MultiStream

Status: runnable common sample.

This sample demonstrates CUDA stream and event primitives through the safe C# wrapper surface:

- two non-blocking CUDA streams
- async device memory fill and copy
- pinned host memory readback
- CUDA event record/synchronize
- cross-stream wait ordering

Run from the repository root with development probing enabled:

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"
```

Run:

```powershell
dotnet .\samples\MultiStream\bin\Debug\net8.0\MultiStream.dll
```

Expected evidence includes:

- `IndependentStreams=True`
- `CrossStreamWait=True`
- `StreamIds A=... B=...`
- `MultiStream Passed=True`

If the local CUDA driver/runtime cannot initialize, the sample prints `MultiStream=Skipped` with the diagnostic reason. That skip is an environment/runtime compatibility signal, not a managed API completion claim.
