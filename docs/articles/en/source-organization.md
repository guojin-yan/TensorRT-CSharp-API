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

Additional hand-written CUDA partial interop files follow the same feature directories as their public owners:

- `Internal/Interop/Devices` for device-resource and primary execution-context operations.
- `Internal/Interop/Diagnostics` for copied runtime logs.
- `Internal/Interop/Drivers` for optional Driver capability, module, and typed launch operations.
- `Internal/Interop/IPC` for owner-safe export/import token operations.
- `Internal/Interop/Kernels` for Runtime kernel-library ownership and launch operations.
- `Internal/Interop/RuntimeCompilation` for optional NVRTC program operations.

`NativeCudaApi.Deployment.cs` remains at the interop root because it currently spans error, PCI, stream/event, pinned-memory, atomic-capability, and device-selection operations. It must be split by behavior in a separate batch instead of being mislabeled as one feature.

TensorRT high-level wrappers are also being split by layer feature area:

- `src/JYPPX.TensorRtSharp/Network/TensorRtNetworkDefinition.Deconvolution.cs`
- `src/JYPPX.TensorRtSharp/Network/TensorRtNetworkDefinition.Lrn.cs`
- `src/JYPPX.TensorRtSharp/Network/TensorRtNetworkDefinition.Quantization.cs`
- `src/JYPPX.TensorRtSharp/Layers/TensorRtLayer.Deconvolution.cs`
- `src/JYPPX.TensorRtSharp/Layers/TensorRtLayer.Lrn.cs`
- `src/JYPPX.TensorRtSharp/Layers/TensorRtLayer.Quantization.cs`
- `src/JYPPX.TensorRtSharp/Execution/TensorRtExecutionContext.Profile.cs`

Hand-written TensorRT partial interop now starts following the same responsibility modules:

- `Internal/Interop/Builder` contains timing-cache, builder-config diagnostics, and runtime controls; `ControlFlow` contains
  loop/conditional operations.
- `Internal/Interop/Callbacks` contains allocator dry-run controls, callback interface/state copies, and logger/profiler/
  progress-monitor delegate signatures.
- `Internal/Interop/Diagnostics` contains copied error-code metadata and TRT11 build probes used only by the environment probe;
  `Internal/Interop/Interfaces` contains owner-scoped versioned-interface metadata copies.
- `Internal/Interop/Engine` contains copied engine/tensor/profile metadata, Dims64, engine-inspector diagnostics, and
  weight-streaming/stat runtime controls; `Execution` contains execution-context/runtime-config creation, allocation-strategy,
  deployment metadata, Dims64, diagnostics, and allocator/event presence controls.
- `Internal/Interop/Inference` contains synchronous execute/enqueue operations; `Weights` contains copied layer-weight metadata.
- `Internal/Interop/Layers` contains quantization, attention, fill-int64, compatibility/deployment layer attributes, Dims64,
  tensor metadata, transformer, and RNNv2 operations; `Network` contains deployment network-layer creation, tensor/network
  Dims64, debug/shape diagnostics, refittable-weight markers, and safe network-v2 operations.
- `Internal/Interop/Parsing` contains the global ONNX parser version, legacy parser diagnostics, ONNX config/model-buffer/support,
  builder-config attachment, layer-output metadata, and parser-refitter diagnostics.
- `Internal/Interop/Plugins` contains global/builder/runtime registry inventories and copied V2/V3 layer metadata/query snapshots.
- `Internal/Interop/Profiles` contains optimization-profile Dims64 shape queries.
- `Internal/Interop/Runtime` contains global runtime version/logger probes, runtime deployment controls, and copied diagnostics;
  `Serialization` contains engine serialization and serialization-config flags; `Refit` contains async refit,
  weights/dynamic-range, entry metadata, and refitter diagnostics.

`NativeBridgeApi.GlobalProbeShared.cs` remains at the interop root only for the common unsupported-line exception helper used by
the split global Runtime, Parsing, and Plugins partials; it contains no public method.

`NativeBridgeApi.SafeDeferredUplift.cs` also remains at the root because it combines plugin initialization with ONNX weight-descriptor parsing. Callback file placement is not callback trampoline, lifetime, or runtime proof.

Version-prefixed files remain at the root only when their method set still crosses owners or features and requires a separate
behavioral split; they are not classified by filename alone.
The former `Trt11DeploymentAdditions` is split by method owner into `Network/NativeBridgeApi.DeploymentNetworkLayers.cs` and
`Layers/NativeBridgeApi.DeploymentLayerAttributes.cs`; recombining both parts must reproduce the pre-split Git blob.
The former `Trt11RuntimeSerializationRefit` is also split across `Runtime`, `Serialization`, `Execution`, and `Refit`; recombining
the four files in original segment order must reproduce the pre-split Git blob.
The former `DeploymentMetadata` is split across `Engine`, `Execution`, `Layers`, and `Refit`; only delegates and cross-owner private
helpers remain in the root `NativeBridgeApi.DeploymentMetadataShared.cs`, and recombination must reproduce the pre-split Git blob.
The former `Trt11Dims64` is split across `Network`, `Engine`, `Execution`, `Profiles`, and `Layers`; layer getter delegates/helpers
move with `Layers`, and recombination in original segment order must reproduce the pre-split Git blob.
The former `Trt11RuntimeControls` is split by contiguous owner sections across `Engine`, `Execution`, and `Builder`; recombination
in original segment order must reproduce the pre-split Git blob.
The former `Trt11Diagnostics` is split across `Builder`, `Network`, `Engine`, and `Execution`; recombination in original segment
order must reproduce the pre-split Git blob.
The former `GlobalRuntimePluginProbe` is split across `Runtime`, `Parsing`, `Plugins`, and a helper-only root Shared partial;
recombination in original segment order must reproduce the pre-split Git blob.

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
