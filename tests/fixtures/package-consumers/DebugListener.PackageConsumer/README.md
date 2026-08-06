# Debug Listener Local Package Consumer

`eng/Test-DebugListenerLocalPackageConsumer.ps1` copies this source into a repository-external
workspace and restores only the managed API and matching bridge-only packages. The generated
project has no `ProjectReference`, source-tree assembly load, development bridge path, or bundled
NVIDIA runtime.

The sample builds a `[1,4]` identity network in code, marks its output as a debug tensor, and proves
the native `processDebugTensor` callback through copied, pointer-free metadata. A controlled handler
rejection must increment the failure counters and detach cleanly.

Run the isolated restore, build, runtime, and evidence flow from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Test-DebugListenerLocalPackageConsumer.ps1 `
  -TensorRtRoot <TensorRT-root> `
  -CudaRoot <CUDA-root>
```

CUDA and TensorRT are host-installed dependencies. This local-feed test does not publish packages
and is not public-package, Release, or post-publish proof.
