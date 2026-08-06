# Logger Local Package Consumer

`eng/Test-LoggerLocalPackageConsumer.ps1` copies this source into a repository-external workspace
and restores only the managed API and matching bridge-only packages. The generated project has no
`ProjectReference`, source-tree assembly load, development bridge path, or bundled NVIDIA runtime.

The sample builds a programmatic 1x1 convolution network, deserializes it, and performs real
TensorRT inference. A managed `TensorRtLogger` receives native TensorRT messages without calling
the synthetic diagnostic helper. The sample verifies copied metadata, thread-safe handler state,
builder/runtime attachment and detach, deferred logger disposal, and managed handler-exception
isolation.

Run the isolated restore, build, runtime, and evidence flow from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Test-LoggerLocalPackageConsumer.ps1 `
  -TensorRtRoot <TensorRT-root> `
  -CudaRoot <CUDA-root>
```

CUDA and TensorRT are host-installed dependencies. This local-feed test does not publish packages
and is not public-package, Release, or post-publish proof.
