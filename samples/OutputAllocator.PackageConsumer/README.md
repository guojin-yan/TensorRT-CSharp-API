# Output Allocator Local Package Consumer

This source is copied into a repository-external workspace by
`eng/Test-OutputAllocatorLocalPackageConsumer.ps1`. The generated project restores only the
managed API and matching bridge-only package. It has no `ProjectReference`, source-tree assembly
load, development bridge path, or bundled NVIDIA runtime.

The sample creates an identity network in code and uses
`TensorRtOutputAllocatorCallbackOwner` to allocate the output during enqueue. The positive path
must allocate and release CUDA memory with no live allocation after detach. A controlled handler
rejection must fail enqueue without allocating memory.

Run the isolated restore, build, runtime, and evidence flow from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Test-OutputAllocatorLocalPackageConsumer.ps1 `
  -TensorRtRoot <TensorRT-root> `
  -CudaRoot <CUDA-root>
```

CUDA and TensorRT are host-installed dependencies. This local-feed test does not publish packages
and is not public-package, Release, or post-publish proof.
