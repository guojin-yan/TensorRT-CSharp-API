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

The two remaining monolithic high-level wrappers now follow the same feature boundary. `Layers/TensorRtLayer.cs` retains
construction, general layer metadata, owner leases, output-index validation, disposal, and the shared Dims validator; 118
Shuffle/MatrixMultiply/Reduce/SoftMax/Unary/TopK/Gather/ElementWise/Activation/Pooling/Convolution/Scale/Padding/Resize/
Concatenation/Slice/Fill methods live in 17 feature partials. `Network/TensorRtNetworkDefinition.cs` retains input/output/layer
ownership, output marking, disposal, and shared tensor validation; 21 corresponding `Add*` methods live in 21 feature partials.
`ManagedWrapperFeatureLayoutTests` fixes each exact method set and recomposes the normalized pre-split sources whose Git blobs
are `faa5fa87fb24247a86f9d166073cf3858bad9ac2` and `933e7b253ecfefa00034da2544912b6e20c353ae`.
This source-only split does not change public signatures, SafeHandle/owner-lease behavior, line routing, validation order, native
entrypoints, generated bindings, manifests, or ABI evidence.

The common Engine and ONNX Parser wrappers now use the same pattern. `Engine/TensorRtEngine.cs` is a 124-line handle,
scalar-property, and disposal core; 36 tensor/profile metadata, binding-report, execution-context, refit, and inspection methods
live in five feature partials. The two private binding-report helpers move with `TensorRtEngine.BindingReports.cs`.
`Parsing/TensorRtOnnxParser.cs` is a 198-line constructor, logger/config/initializer lifetime, scalar-property, disposal, and
shared-validation core; 32 model parsing/loading, TryParse, diagnostic, operator-support, and flag methods live in six feature
partials. Parser flag validation and model segment/stream copy helpers remain in core because multiple partials consume them.
`ManagedEngineParserFeatureLayoutTests` fixes method/helper ownership and recomposes the pre-split Git blobs
`fd6907a9e03eab3b6f9a1b5820eea9e6e1e82ea9` and `d8e3f135b71f9b2fd893146776da7d538db5d020`.

BuilderConfig and ExecutionContext common wrappers are split without overlapping their existing TRT11 partials.
`Builder/TensorRtBuilderConfig.cs` is a 102-line handle, progress-monitor lifetime, shared validation, and disposal core;
34 profile/flag/compatibility/layer-device/memory-pool/scalar/tactic/timing-cache methods plus seven feature properties live
in eight partials. `ValidateLayer`, disposed-state, and progress-monitor helpers remain in core because existing diagnostics
partials consume them. `Execution/TensorRtExecutionContext.cs` is a 176-line metadata, profiler/aux-stream lifetime, cleanup,
and disposal core; 20 shape/address/device-memory/event/enqueue methods live in five feature partials. The layout gate recomposes
the pre-split Git blobs `592ce09c4da5fb4f7a376800cd5a6b309a22481b` and
`1614e46a4b0175ff9889302f977a0520be404b29`.

ParserRefitter and high-level InferenceBindings are split by call stage as well. `Parsing/TensorRtOnnxParserRefitter.cs` is now
a 114-line native-owner, refitter/logger borrower lifetime, initializer-pin lifetime, shared model segment/stream copy, and
disposal core; 21 refit/model-proto/initializer/diagnostic methods live in four feature partials.
`Inference/TensorRtInferenceBindings.cs` is now a 124-line engine/context reference, buffer owner, shared tensor lookup/report
refresh, disposal, and disposed-state core; 14 geometry/buffer/host-transfer/address-binding/execution/diagnostic methods live
in six feature partials. Feature-specific size-estimation, buffer-replacement, and readiness helpers stay with their owners.
The layout gate recomposes the pre-split Git blobs `07975ca6274fb64fcce06ca6e967515f07aa734c` and
`6257b4c8ad3c0f8c4ec349208e8583b670359d0a`.

Environment probing and plugin inventory no longer combine unrelated responsibilities in one file.
`Diagnostics/TensorRtEnvironmentProbe.cs` is reduced from 1,497 lines to a 62-line core for cross-feature exception
classification, diagnostic formatting, and stage helpers; 43 public static entry points live in seven feature partials.
`Plugins/TensorRtPluginRegistryInventory.cs` retains only the aggregate inventory type, while two enums, copied field/creator
metadata, creator/field summaries, and inventory diagnostics live in six dedicated files. The layout gate recomposes the
pre-split Git blobs `bcd1301777f81d9de32625b4c7be952de3126d92` and
`c39d9ec853987f50f452b0dfebc2658120aa95d2`.

The high-level CUDA graph and device-memory wrappers follow the same feature ownership rule.
`Graphs/CudaGraph.cs` is reduced from 1,630 lines to a 206-line graph-handle, capture/conditional/allocation owner-lifetime,
shared-validation, and disposal core. Conditional handles, graph composition, node creation, topology diagnostics, node
inspection, node mutation, node relations, and instantiation live in eight feature partials.
`Memory/CudaMemory.cs` is reduced from 951 lines to a 114-line allocation-handle, IPC-import metadata, shared range validation,
and disposal core. IPC, range diagnostics, async allocation, fill, prefetch/advice, host transfers, device transfers, async free,
and array conversion live in nine feature partials alongside the existing registered-host partial. The layout gate recomposes
the pre-split Git blobs `5b7e06e4e59d6961c9c848f88d1f9ace6a9c0450` and
`1d2085d8ffc527cfd280c2ba68836d14bb2c6334`.

Device-wide CUDA helpers and cross-module enums are no longer collected in two broad files either.
`Devices/CudaDevice.cs` is reduced from 793 lines to a 144-line runtime/driver/device identity and property-snapshot core;
graph resources, runtime configuration, initialization/selection, peer capabilities, memory pools, cache/RDMA, and
synchronization/error diagnostics live in seven feature partials. The former 912-line `Core/CudaFlags.cs` is removed: its
24 public enums now live in ten Streams, Events, Memory, Devices, and Graphs module files. Enum names, underlying types,
numeric values, and XML documentation remain unchanged. The layout gate recomposes the pre-split Git blobs
`df16e51427f82c9b99fa867819015539dd65ac0c` and `dca1aa594002506ce47bd247f47141201af6591d`.

Pitched and CUDA-array memory owners are split by transfer dimensionality while retaining validation in their owner cores.
`Memory/CudaPitchedMemory.cs` is reduced from 678 lines to a 181-line allocation/metadata, shared pitch/extent/pinned-buffer
validation, and disposal core; fill, 2D transfers, 3D transfers, and array conversion live in four feature partials.
`Memory/CudaArray.cs` is reduced from 609 lines to a 208-line allocation/metadata, memory-requirement-independent shared
validation, and disposal core; copied requirements/sparse diagnostics, 1D/2D/3D transfers, and array conversion live in five
feature partials. The layout gate recomposes the pre-split Git blobs `e30004cce7cc55c7b62e19478de913d68be3c591` and
`6de4bc82180a8539e3b6d6fa610c485c042d043e`.

The CUDA stream and memory-pool owners are split by call stage and responsibility as well. `Streams/CudaStream.cs` is reduced
from 439 lines to a 126-line handle/property, capture owner-count, and disposal core; general diagnostics, capture diagnostics,
capture dependencies, synchronization/event operations, and capture lifecycle live in five feature partials.
`Memory/CudaMemoryPool.cs` is reduced from 372 lines to a 31-line handle/value core; factories, allocation, access, and attributes
live in four feature partials, while the independent `CudaOwnedMemoryPool` owner and two pool enums move to dedicated files.
The layout gate recomposes the pre-split Git blobs `379fd60f08beca36711507a6c83a2192d20b6562` and
`7028220b628b3d2af9c3f007dda75ec4ff2f5fd3`.

The Tools ONNX engine build service is organized by build stage as well. `Build/OnnxEngineBuildService.cs` is reduced from
3,214 lines to a 540-line selected-device thread, dry-run, and build-orchestration core. Builder configuration, timing cache,
result creation, refit, diagnostics, runtime execution, benchmarking, runtime inputs, reference validation, and deployment
configuration live in ten feature partials; private lease, worker, and runtime-state types follow their owning features.
The former 970-line `Build/OnnxEngineBuildResult.cs` is reduced to a 304-line primary result. Timing-cache artifact, capability
probe, loaded-engine diagnostics, preflight metadata, model evidence, and benchmark summary are six independent public type
files. The layout gate recomposes the pre-split Git blobs `31f2c170c9c74b7278b4bc266eca76405dc33067` and
`97f6e992fe582513fcf77b10ce05de4de32af1a5`.

ONNX runtime artifacts and build-report projections are separated by output responsibility. The former 1,052-line
`Artifacts/OnnxEngineRuntimeArtifactWriter.cs` is reduced to an 85-line dispatch and hashing core; times, structured output,
benchmark profile, engine readback, raw bindings, proof boundaries, and file I/O live in seven feature partials, while runtime
artifact data and output records are independent public types. The former 776-line `Build/OnnxEngineBuildDiagnostics.cs` is
reduced to a 37-line report-format dispatch core; JSON, Markdown, option implementation status, and report proof boundaries
live in four feature partials, with option-status and report-boundary records in dedicated files. The layout gate recomposes
the pre-split Git blobs `5269b4685e87d8eb0e65b337499963b12de71aaa` and
`c7c832f862d8c3d4bff2b93666e9f842b24cdd91`.

The Tools model-specific MNIST runtime and trtexec-like parser are modularized without changing their command or proof
semantics. The former 932-line `Runtime/MnistOnnxRuntime.cs` is removed: options, PGM image/reader, classification,
environment, result, and diagnostics now have dedicated files, while execution, artifact hashing, tensor/shape handling,
and option validation live in four `MnistOnnxRuntimeService` partials. The 731-line `Trtexec/TrtexecLikeParser.cs` is reduced
to a 262-line Parse orchestration core; argument collection, scalar/reference parsing, build-option values, and memory units
live in four feature partials. The layout gate recomposes the pre-split Git blobs
`116ff0707028cd234819b023f6b2b5e0ff3cbb43` and `3c94e7b5fe1ed741b722f71eb10f2888f76567b7`.

Trtexec deployment projections and build policies now follow the same ownership rule. The 644-line
`Trtexec/TrtexecLikeDeploymentOptions.cs` is reduced to a 225-line constructor/default/property core; argument projection,
diagnostics, tactic-source resolution, memory-pool projection, and shared formatting live in five partials, and
`TrtexecLikeMemoryPoolSize` has its own file. The 568-line `TrtexecLikeBuildPolicy.cs` is reduced to a 93-line normalization
and Apply orchestration core; parsing, IO formats, precision constraints, layer policies, rule validation, and data-type/format
parsing live in six feature partials, while IO-format and layer-rule models are independent internal types. The layout gate
recomposes the pre-split Git blobs `d55d835f5dd934975af9a302391415e49030a14d` and
`ef5b4bf488124bb1d37090cbf5cf1b75cf345dfb`.

TensorRT public enums are no longer collected in the 2,456-line `Core/TensorRtEnums.cs`. Its 64 enums are distributed across
15 module files for shared Core tensor values, Network, Parsing, Execution, Serialization, Engine, Runtime, Builder, Profiles,
ControlFlow, and the Layers RNN, operation, resize, metadata, and attention domains. Single-value/flags pairs remain together;
names, underlying types, values, `Flags`, and XML documentation are unchanged. The layout gate recomposes the pre-split Git
blob `861460b834d7d415fc0497f6c5b05fecaba03707` in its original declaration order.

Hand-written TensorRT partial interop now starts following the same responsibility modules:

- `Internal/Interop/Builder` contains builder creation/capabilities, serialized build outputs, builder boundary controls,
  timing-cache lifecycle/TRT11 controls, builder-config core/scalar controls, diagnostics, plugin serialization, and runtime controls;
  `ControlFlow` contains loop/conditional operations.
- `Internal/Interop/Callbacks` contains allocator dry-run controls, callback interface/state copies, logger/profiler/progress-monitor
  delegate signatures, and managed callback diagnostics.
- `Internal/Interop/Diagnostics` contains copied error-code metadata plus cross-version/TRT11 build-chain probes used only by
  the environment probe;
  `Internal/Interop/Interfaces` contains owner-scoped versioned-interface metadata copies.
- `Internal/Interop/Engine` contains core/deployment engine metadata, inspector lifecycle/information/boundary/diagnostics,
  copied tensor/profile values, Dims64, error-recorder controls, and weight-streaming/stat runtime controls; `Execution` contains
  execution-context core creation, binding/address/enqueue/aux-stream controls, boundary controls, copied engine metadata, runtime-config creation,
  allocation-strategy, deployment metadata, Dims64, diagnostics, and allocator/event presence controls.
- `Internal/Interop/Inference` contains synchronous execute/enqueue operations; `Weights` contains copied layer-weight metadata.
- `Internal/Interop/Layers` contains identity/constant/convolution/deconvolution/scale/padding/element-wise/matrix-multiply/shuffle/reduce,
  softmax, unary, TopK, and gather features,
  activation, pooling, and LRN features, shared optional-weight helpers, quantization, attention, fill-int64,
  resize, concatenation, and slice features, compatibility/deployment layer attributes, Dims64,
  shape, select, and fill features, core layer metadata, tensor metadata, transformer, and RNNv2
  operations; `Network` contains core definition input/output/name/flags, boundary controls,
  deployment network-layer creation, core tensor name/type/shape/range metadata, tensor/network Dims64, debug/shape diagnostics,
  refittable-weight markers, and safe network-v2 operations.
- `Internal/Interop/Parsing` contains the global ONNX parser version, parser lifecycle/input/diagnostics/flags, legacy parser
  diagnostics, ONNX config/model-buffer/support, builder-config attachment, layer-output metadata, weight-descriptor parsing,
  parser-refitter diagnostics, and shared copied-string helpers.
- `Internal/Interop/Plugins` contains plugin initialization, global/builder/runtime registry inventories, and copied V2/V3 layer
  metadata/query snapshots.
- `Internal/Interop/Profiles` contains optimization-profile core creation/shape controls, Dims64, and shape-value queries.
- `Internal/Interop/Runtime` contains adapter information, runtime creation diagnostics, cross-version line-binding helpers,
  global runtime version/logger probes, runtime deployment controls, and copied diagnostics;
- `Serialization` contains engine deserialization/serialization, serialization-config flags, and host-memory buffer/metadata; `Refit` contains async refit,
  weights/dynamic-range, entry metadata, and refitter diagnostics.

`NativeBridgeApi.GlobalProbeShared.cs` remains at the interop root only for the common unsupported-line exception helper used by
the split global Runtime, Parsing, and Plugins partials; it contains no public method.
`NativeBridgeApi.OwnerErrorRecorderSnapshotShared.cs` remains at the root only for the copied error-recorder snapshot mapper used
by Builder, EngineInspector, Execution, and Network owners; it contains no public method.

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
The former `SafeDeferredUplift` is split across `Plugins` and `Parsing`; recombination in original segment order must reproduce
the pre-split Git blob. File placement does not change its deferred history or prove plugin/parser runtime and lifetime behavior.
The former `Trt11BoundaryControls` is split across `Builder`, `Engine`, `Execution`, `Network`, and a helper-only root Shared
partial; recombination in original segment order must reproduce the pre-split Git blob.
The former `Trt11FourteenthBatch` is split across two `Builder` feature files plus `Serialization`, `Profiles`, and `Execution`;
the serialized-network result struct moves with the Builder build-output methods, and recombination must reproduce the pre-split Git blob.
The former `Trt11FifteenthBatch` is split across two `Engine` feature files and `Execution`; profile-value validation helpers move
with the Engine profile methods, and recombination must reproduce the pre-split Git blob.
The root `NativeBridgeApi.cs` is now reduced in small owner/feature batches: its cross-version timing-cache create/set/serialize
segment moves to `Builder/NativeBridgeApi.TimingCacheLifecycle.cs`, and reinserting that segment at its original position must
reproduce the pre-split root Git blob.
The root ONNX parser core is split into Parsing lifecycle/input, diagnostics, flags/operator-support, and helper-only string-read
partials. The parser-specific diagnostic helper moves with diagnostics, while the delegate and copied-string allocator remain
shared by parser, parser-refitter, and support features; recombination in original segment order must reproduce the prior root blob.
The root tail owner block is split into Engine inspector core, ExecutionContext binding/enqueue, Serialization host-memory buffer,
and Engine core metadata partials. Engine-information and IO-tensor-name getters move with their owners. The BuilderConfig bit-flag
helper initially remains in the root; Tensor/Layer name helpers move later with their core owner partials. Original-order recombination remains required.
Cross-version line-binding delegates, the private bindings class, and version routing move from the root into a helper-only Runtime
partial; six environment-probe operations and their minimal build-chain helpers move into Diagnostics. Generated bindings/helpers
continue consuming the same private partial type, and original-order recombination must reproduce the prior root blob.
The root Network definition core moves to `Network/NativeBridgeApi.NetworkCore.cs`, including input/output ownership, layer lookup,
name/flags metadata, and its private name getter. Layer creation initially remains in the root for later feature splits; optional-weight
helpers likewise remain with their weighted-layer consumers.
Identity/constant, convolution, deconvolution, and scale creation now move into separate Layers feature partials. Scale keeps its
single-feature data-type selector; pin/validation helpers shared by convolution, deconvolution, and scale move into a helper-only
Layers Shared partial, with recombination preserving original helper order.
Padding, element-wise, and matrix-multiply creation/attributes move into three additional Layers feature partials. These blocks
have no private helper. Shuffle creation, reshape/transpose attributes, and zero-placeholder controls move into a fourth partial;
that block also has no private helper, and direct original-order recombination remains required. Reduce creation and readonly
operation/axes/keep-dimensions attributes move into another helper-free partial.
SoftMax, unary, TopK, and gather creation/attributes move into four independent helper-free partials. Activation, pooling, and
LRN creation/attributes move into three more independent helper-free partials. Resize, concatenation, and slice creation/attributes
move into three additional helper-free partials; resize keeps its array pinning inside the feature file. Shape, select, and fill
creation/attributes move into three final helper-free feature partials before the root enters general Layer metadata at
`GetLayerOutput`.
General Layer metadata moves into `Layers/NativeBridgeApi.LayerCoreMetadata.cs` with `MapLayerType` and its private name getter;
general Tensor metadata moves into `Network/NativeBridgeApi.TensorCoreMetadata.cs` with its private name getter.
The remaining adapter/callback/runtime/builder/network/serialization/execution/profile/config implementation moves into ten owner
partials, including `Builder/NativeBridgeApi.BuilderConfigCore.cs` with the bit-flag helper. Root `NativeBridgeApi.cs` is now a
13-line declaration shell with no public or private static implementation. Recombining the ten bodies between that shell's header
and close brace reproduces the prior root blob.

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
