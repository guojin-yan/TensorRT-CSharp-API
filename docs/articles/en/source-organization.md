# Source Organization

The Windows API completion phase now treats source modularization as a quality gate, not only a readability preference.

## Native bridge

Large native bridge files are split by deployment area while preserving C ABI names and behavior.

- CUDA modules live under `native/src/cuda/modules`.
- TensorRT 8 modules live under `native/src/tensorrt/v8/modules`.
- TensorRT 10 modules live under `native/src/tensorrt/v10/modules`.

The first extracted native groups are:

- CUDA pitched memory allocation and 2D copy helpers.
- CUDA async memory allocation and default memory-pool helpers.
- TensorRT builder-config helpers.
- TensorRT network/tensor helpers.
- TensorRT parser/inspector helpers.
- TensorRT convolution / scale / padding layer helpers.
- TensorRT deconvolution layer helpers.
- TensorRT LRN and quantization layer helpers.
- TensorRT execution-context deployment helpers.

These modules are currently included from the original `api.cpp` files instead of being compiled as separate translation units. This keeps access to existing anonymous-namespace helpers and avoids accidental ABI or behavior changes. A module should move to standalone `.cpp` only after its validation, ownership, and version-guard helpers are promoted to reusable headers.

## Managed interop

Managed interop is split by feature area when a wrapper group becomes stable enough to isolate.

The first managed splits are CUDA pitched-memory, async-memory, and registered-host-memory interop/wrapper boundaries:

- `src/JYPPX.CudaSharp/Internal/Interop/Memory/NativeCudaApi.PitchedMemory.cs`
- `src/JYPPX.CudaSharp/Internal/Interop/Memory/NativeCudaApi.AsyncMemory.cs`
- `src/JYPPX.CudaSharp/CudaMemory.RegisteredHost.cs`
- `src/JYPPX.CudaSharp/CudaRegisteredHostMemory.cs`

TensorRT high-level wrappers are also being split by layer feature area:

- `src/JYPPX.TensorRtSharp/TensorRtNetworkDefinition.Deconvolution.cs`
- `src/JYPPX.TensorRtSharp/TensorRtNetworkDefinition.Lrn.cs`
- `src/JYPPX.TensorRtSharp/TensorRtNetworkDefinition.Quantization.cs`
- `src/JYPPX.TensorRtSharp/TensorRtLayer.Deconvolution.cs`
- `src/JYPPX.TensorRtSharp/TensorRtLayer.Lrn.cs`
- `src/JYPPX.TensorRtSharp/TensorRtLayer.Quantization.cs`
- `src/JYPPX.TensorRtSharp/TensorRtExecutionContext.Profile.cs`

Generated files remain under their `Generated` folders and should not be manually split. Generator output layout changes must happen in the generator itself and must pass the deterministic generator gate.

## Rules

- Do not rename exported C ABI entrypoints during modularization.
- Do not change manifest semantics just to move code.
- Do not expose raw `IntPtr` to ordinary C# users as part of source cleanup.
- Run native build, managed build, and binding-generator deterministic validation after every split.
