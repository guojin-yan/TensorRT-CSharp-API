# CUDA Stream-Capture Variant Candidate Audit

Decision: `promote-by-explicit-alias-history`

This batch promotes three CUDA runtime variant families whose vendor symbols
are available in the local CUDA headers and whose managed route can copy all
inputs/outputs during the synchronous native call. The previous deferred
records remain in `cuda-thirty-fifth-batch-stream-device-boundaries.manifest.json`.

## Candidate Surface

| Official runtime function | Supported toolkit guard | Safe bridge shape | Ownership risk |
| --- | --- | --- | --- |
| `cudaStreamGetCaptureInfo_ptsz` | CUDA 12.x | typed stream owner in; copied `status` and `captureId` out | Low: scalar snapshot only |
| `cudaStreamUpdateCaptureDependencies_ptsz` | CUDA 11.6-12.1 | typed stream owner plus managed graph-node tokens and primitive mode | Medium-low: token array is borrowed only for the call |
| `cudaStreamUpdateCaptureDependencies_v2` | CUDA 12.3-12.9 | typed stream owner, managed graph-node tokens, copied `CudaGraphEdgeData`, primitive mode | Medium-low: edge data and token array are pinned only for the call |

## Vendor Evidence

| Function | Header evidence | Native declaration/link evidence | Native implementation |
| --- | --- | --- | --- |
| `cudaStreamGetCaptureInfo_ptsz` | CUDA 12.1, 12.3 and 12.9 runtime headers | cross-version native builds link the symbol through the CUDA runtime import library | `native/src/cuda/modules/graph/stream_capture_variants.inc` |
| `cudaStreamUpdateCaptureDependencies_ptsz` | CUDA 11.6, 11.8 and 12.1 runtime headers | CUDA 11.8 and CUDA 12.x native builds link the symbol; explicit vendor declaration is guarded for headers that omit it | `native/src/cuda/modules/graph/stream_capture_variants.inc` |
| `cudaStreamUpdateCaptureDependencies_v2` | CUDA 12.3 and 12.9 runtime headers | CUDA 12.x native builds link the symbol through the runtime import library | `native/src/cuda/modules/graph/stream_capture_variants.inc` |

The CUDA 13.2 build intentionally does not expose the 12.x-only variants.
Each entry has an independent `CUDART_VERSION` guard, so an unavailable
vendor symbol is not manufactured by the bridge on a later toolkit.

## Manifest and History Mapping

Real records are in:

- `native/manifests/cuda/cuda-fifty-sixth-batch-stream-capture-variants.manifest.json`

Historical deferred records remain in:

- `native/manifests/cuda/cuda-thirty-fifth-batch-stream-device-boundaries.manifest.json`

`eng/Export-InterfaceCoverageMatrix.ps1` maps the real IDs explicitly and
keeps the old deferred IDs in the deferred-history alias map. The coverage
rows for all three functions therefore report
`implemented-with-deferred-history` rather than erasing the historical
deferred inventory.

## Managed Route and Safety Contract

- `CudaStream.GetCaptureInfoPtzs` returns a readonly scalar snapshot and has a
  `Try` route for status-preserving callers.
- `CudaStream.UpdateCaptureDependenciesPtzs` and
  `CudaStream.UpdateCaptureDependenciesV2` accept managed `CudaGraphNode`
  tokens and copy `CudaGraphEdgeData` into a native value before the vendor
  call returns.
- Native helpers bound dependency counts and convert allocation, invalid
  argument, C++ exception and Windows SEH failures into bridge status codes.
- No public API exposes `IntPtr`, `nint`, `UIntPtr`, `SafeHandle`, device
  pointer, borrowed vendor pointer or external-resource ownership.

## Verification

- Binding generator: `3945 API records / 188 manifests`; output validation and
  idempotence passed.
- Managed solution and ProjectQuality build: 0 errors; existing nullable
  warnings unchanged.
- New focused tests: `4/4` passed; related CUDA graph/stream tests: `30/30`
  passed.
- Native ABI surface: all checked configurations reported
  `MissingDeclarations=0 MissingExports=0`.
- Four native configurations built successfully:
  TRT8/CUDA11.8, TRT8/CUDA12.1, TRT10/CUDA12.9 and TRT11/CUDA13.2.
- Coverage export shows all three functions as
  `implemented-with-deferred-history` on the applicable toolkit rows.

## Deliberately Deferred

CUDA graph user objects, allocator/resource APIs, callback trampolines,
borrowed/device pointers, plugin lifecycle and any variant whose vendor
symbol or ownership contract cannot be proven remain deferred.
