# TensorRT Deferred Safe-Uplift Candidate Audit

- Source baseline: `fedf08d6659af1e7e614e3157e7bbe7d2de30cc9`
- TensorRT deferred rows: `598`
- Reviewed unique candidates: `50`
- Risk split: low `0`, medium `112`, high `486`
- CUDA immediate-safe candidates: `0`

## Decision

The automatic selector still has no low-risk candidate. Callback trampolines, borrowed expression pointers,
error-recorder refcounts, plugin mutation, allocator/resource ownership, device pointers, deserialization ownership,
and the missing TensorRT 8 consistency-checker symbol remain deferred.

This batch selects `IParser::getLayerOutputTensor` only as an owner-scoped copied-metadata safe alternative. It does
not promote the vendor's borrowed `ITensor*`, and it does not delete the original deferred record.

## Vendor ABI Audit

| Line | Header | LIB/DLL named symbol | Result |
| --- | --- | --- | --- |
| TRT8 | method absent | not applicable | controlled `NotSupported` |
| TRT10 | pure virtual method present | no named symbol expected or found | versioned adapter link plus real runtime smoke required |
| TRT11 | pure virtual method present | no named symbol expected or found | versioned adapter link plus compatible-host smoke required |

`getLayerOutputTensor` is dispatched through the C++ parser vtable. A `dumpbin` search therefore should not find a
same-named import-library member or DLL export. Header presence, independent native builds, bridge export parity and
runtime execution are the relevant checks.

## Copied Contract

The native bridge reads the parser-owned tensor only while the parser handle is valid and copies tensor name,
64-bit shape, data type, location, allowed formats, and dynamic/shape/execution/network-input/network-output flags.
The public surface returns `TensorRtOnnxLayerOutputTensorMetadata`; it creates no tensor handle and exposes no
`IntPtr`, `nint`, `UIntPtr`, `SafeHandle`, device pointer or borrowed tensor object.

## Boundary

This readonly diagnostic snapshot is not inference proof, external-model proof, package-consumer runtime proof,
post-publish proof or permission to delete deferred history. TRT11 remains dependency/runtime-probe-only until its
vendor builder/runtime owner can be created on a compatible host.
