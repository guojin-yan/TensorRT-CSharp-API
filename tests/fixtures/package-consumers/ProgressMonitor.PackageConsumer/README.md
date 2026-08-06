# Progress Monitor Local Package Consumer

`eng/Test-ProgressMonitorLocalPackageConsumer.ps1` copies this source into a repository-external
workspace and restores only the managed API and matching bridge-only packages. The generated
project has no `ProjectReference`, source-tree assembly load, development bridge path, or bundled
NVIDIA runtime.

The sample builds a small 1x1 convolution network in code and proves that TensorRT invokes
`IProgressMonitor` during a real engine build. It verifies copied phase metadata, callback counts,
clean detach behavior, and a controlled `stepComplete` cancellation that must stop the build.
Handlers use thread-safe state because TensorRT may report build phases concurrently.

Run the isolated restore, build, runtime, and evidence flow from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Test-ProgressMonitorLocalPackageConsumer.ps1 `
  -TensorRtRoot <TensorRT-root> `
  -CudaRoot <CUDA-root>
```

CUDA and TensorRT are host-installed dependencies. This local-feed test does not publish packages
and is not public-package, Release, or post-publish proof.
