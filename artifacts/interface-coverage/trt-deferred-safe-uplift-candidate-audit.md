# TensorRT Deferred Safe-Uplift Candidate Audit

- Source baseline: `f4c38e74680c02cdb4c1be06ecc8d7347a591159`
- TensorRT deferred rows: `598`
- Reviewed unique candidates: `50`
- Risk split: low `0`, medium `112`, high `486`

## Decision

This batch does not promote a deferred TensorRT API. The remaining reviewed candidates still depend on calibrator/callback trampolines, borrowed `IDimensionExpr` lifetime, error-recorder refcounting, plugin ownership, runtime deserialization ownership, or the TensorRT 8 consistency-checker symbol that is absent from the inspected import libraries and DLLs.

Deleting deferred manifests or adding a pointer-shaped managed surface would change coverage numbers without producing an owner-safe runtime API, so the existing history remains intact.

## Selected Alternative

The implementation batch moves to the existing owner-scoped network surface:

- `--inputIOFormats`
- `--outputIOFormats`
- `--precisionConstraints`
- `--layerPrecisions`
- `--layerOutputTypes`

`TensorRtTensor`, `TensorRtLayer`, `TensorRtNetworkDefinition`, and `TensorRtBuilderConfig` already carry managed owner/lifetime rules and typed set/get APIs. This makes the batch useful to `TensorRtExec` without exposing `IntPtr`, `nint`, `UIntPtr`, `SafeHandle`, borrowed tensors, plugin objects, or callback state.

## Boundary

No callback, plugin lifecycle, refcount, allocator/resource, device pointer, borrowed pointer, runtime deserialization owner, or consistency checker API is promoted by this audit. Old deferred manifests remain present.
