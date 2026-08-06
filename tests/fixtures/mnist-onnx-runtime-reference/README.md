# MNIST ONNX Runtime Reference Producer

This source is copied into an isolated E-drive workspace by
`eng/Test-TensorRtExecMnistOnnxRuntimeReference.ps1`. The generated project restores only from
`.nupkg` files already present in the local NuGet global-package cache; all remote sources are
cleared. It does not add ONNX Runtime as a dependency of the TensorRtSharp solution.

The runner executes the repository MNIST model twice with the explicit ONNX Runtime CPU execution
provider, checks byte-exact determinism, parses the profiling trace to verify
`CPUExecutionProvider`, writes a structured reference, and compares it with the retained TensorRT
regression reference.

A clean clone retains the reference, sidecar, compact evidence, and validation summary. Raw output,
profiling traces, restore caches, and local run logs stay ignored and are required only when the
strict validator is invoked with `-RequireRuntimeArtifacts` on the capture host.

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Test-TensorRtExecMnistOnnxRuntimeReference.ps1 -Strict
```

An ONNX Runtime CPU reference is independent of TensorRT execution, but it is still an unreviewed
reference candidate. It is not Owner approval for model redistribution, public-package proof,
post-publish proof, or release authorization.
