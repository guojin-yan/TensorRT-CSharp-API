# Profiler Local Package Consumer

`eng/Test-ProfilerLocalPackageConsumer.ps1` copies this source into a repository-external workspace
and restores only the managed API and matching bridge-only packages. The generated project has no
`ProjectReference`, source-tree assembly load, development bridge path, or bundled NVIDIA runtime.

The sample builds a small 1x1 convolution network in code and performs real TensorRT inference. It
proves immediate layer timing with `EnqueueEmitsProfile=true`, deferred timing with
`EnqueueEmitsProfile=false` plus `ReportToProfiler()`, copied layer names and finite durations, clean
detach behavior, and a controlled managed handler exception. Handler state is thread-safe.

Run the isolated restore, build, runtime, and evidence flow from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Test-ProfilerLocalPackageConsumer.ps1 `
  -TensorRtRoot <TensorRT-root> `
  -CudaRoot <CUDA-root>
```

CUDA and TensorRT are host-installed dependencies. This local-feed test does not publish packages
and is not public-package, Release, or post-publish proof.
