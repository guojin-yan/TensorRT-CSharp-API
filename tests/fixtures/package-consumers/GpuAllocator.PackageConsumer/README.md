# GPU Allocator Local Package Consumer

This source is copied into a repository-external workspace by
`eng/Test-GpuAllocatorLocalPackageConsumer.ps1`. The generated project contains only two
`PackageReference` items: the managed API package and one bridge-only runtime package. It has no
`ProjectReference`, does not load source-tree assemblies, and does not enable development probing.

The sample creates a small identity network in code, attaches a managed GPU allocator to both a
TensorRT runtime and builder, and verifies detach and zero-live-allocation invariants. It also runs
controlled allocation rejection and callback-exception cases; both must fail the TensorRT build
without leaking allocations.

Run the isolated restore, build, runtime, and evidence flow from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Test-GpuAllocatorLocalPackageConsumer.ps1 `
  -TensorRtRoot <TensorRT-root> `
  -CudaRoot <CUDA-root>
```

CUDA and TensorRT must already be installed on the host. The script consumes local package files
only; it does not publish packages and is not public-feed or post-release proof.
