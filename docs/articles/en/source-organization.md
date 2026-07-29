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
- `src/JYPPX.CudaSharp/Memory/CudaMemory.RegisteredHost.cs`
- `src/JYPPX.CudaSharp/Memory/CudaRegisteredHostMemory.cs`

TensorRT high-level wrappers are also being split by layer feature area:

- `src/JYPPX.TensorRtSharp/Network/TensorRtNetworkDefinition.Deconvolution.cs`
- `src/JYPPX.TensorRtSharp/Network/TensorRtNetworkDefinition.Lrn.cs`
- `src/JYPPX.TensorRtSharp/Network/TensorRtNetworkDefinition.Quantization.cs`
- `src/JYPPX.TensorRtSharp/Layers/TensorRtLayer.Deconvolution.cs`
- `src/JYPPX.TensorRtSharp/Layers/TensorRtLayer.Lrn.cs`
- `src/JYPPX.TensorRtSharp/Layers/TensorRtLayer.Quantization.cs`
- `src/JYPPX.TensorRtSharp/Execution/TensorRtExecutionContext.Profile.cs`

Generated files remain under their `Generated` folders and should not be manually split. Generator output layout changes must happen in the generator itself and must pass the deterministic generator gate.

## Managed public API layout

Public API files are grouped by responsibility. Moving a file does not change its existing namespace, type name, or public API. The SDK-style projects compile these directories recursively, so the project files do not need per-file `Compile` entries.

| Project | Module directories |
| --- | --- |
| `JYPPX.CudaSharp` | `Core`, `Devices`, `Diagnostics`, `Drivers`, `Events`, `Graphs`, `IPC`, `Kernels`, `Memory`, `RuntimeCompilation`, `Streams` |
| `JYPPX.TensorRtSharp` | `Builder`, `ControlFlow`, `Core`, `Diagnostics`, `Engine`, `Execution`, `Inference`, `Interfaces`, `Layers`, `Network`, `Parsing`, `Plugins`, `Profiles`, `Refit`, `Runtime`, `Serialization`, `Weights` |
| `JYPPX.TensorRtSharp/Callbacks` | `Core`, `Debugging`, `MemoryAllocation`, `Monitoring` |
| `JYPPX.TensorRtSharp.Tools` | `Artifacts`, `Build`, `Core`, `Refit`, `Runtime`, `Trtexec` |

The current reorganization covers 294 `.cs` files that previously lived at the roots of these three projects. `ManagedSourceModuleLayoutTests` keeps public API files out of the project roots and verifies that every named module contains source files.

The CUDA Driver capability entry point, copied module owner, and typed launch owner are grouped in `JYPPX.CudaSharp/Drivers`. `Kernels` remains responsible for CUDA Runtime kernel-library ownership and typed argument/configuration values.

Copied TensorRT versioned-interface metadata and owner-scoped metadata queries live in `JYPPX.TensorRtSharp/Interfaces`. Managed weights payloads, copied weights metadata, and refit weight roles live in `JYPPX.TensorRtSharp/Weights`; `Core` remains limited to shared exceptions, dimensions, enums, and library information.

ONNX stripped-plan refit lifecycle and persisted-plan reload snapshots live in `JYPPX.TensorRtSharp.Tools/Refit`. The `Build` module consumes these copied evidence models while retaining responsibility for build options, services, diagnostics, and results.

## Rules

- Do not rename exported C ABI entrypoints during modularization.
- Do not change manifest semantics just to move code.
- Do not expose raw `IntPtr` to ordinary C# users as part of source cleanup.
- Place new public APIs in an existing responsibility module; add a directory only for a distinct ownership boundary.
- Run native build, managed build, and binding-generator deterministic validation after every split.
