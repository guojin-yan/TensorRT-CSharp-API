# YOLOX Official Source-Tree Runtime Proof Closure

- classification: `source-tree-real-model-runtime`
- upstream: YOLOX `0.1.1rc0` / `e1052df71842031413f6030723c3607b839c80ce`
- license provenance: Apache-2.0 file hash pinned
- model: `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`
- TensorRT: `10.11.0.33`, FP32
- tensors: `images [1,3,640,640] -> output [1,8400,85]`
- runtime: 5 detections, `bicycle=0.954841`, `dog=0.913382`
- passed marker: `YoloVision Passed=True`
- sample-run validator: `real-model-runtime`, promotion `True`, owner-action `0`

## Implementation

`YoloXOutputDecoder` applies `(xy + grid) * stride` and `exp(wh) * stride` for strides
8, 16, and 32 before objectness scoring and class-aware NMS. The built-in family is detection-only.
Its default preprocessing is NCHW, BGR, unnormalized float32 `0..255`, fill 114, and top-left
letterbox. Other YoloVision families preserve their existing centered letterbox defaults.

## Verification

- bindings: `191 manifests / 3961 records`, idempotent
- YoloVision ProjectQuality: `88/88`
- sample manifest audit: 9 manifests, 0 findings, YOLOX sidecar/run cross-check `checked`
- native builds: TRT8/CUDA12, TRT10/CUDA12, TRT11/CUDA12, TRT11/CUDA13
- ABI/PE parity: TRT8 `991/991`, TRT10 `1086/1086`, TRT11 `1233/1233`, missing 0
- classification audit: 0 findings
- strict release quality gate: 0 required failures

## Boundary

This record proves one real source-tree TensorRT execution with hash-pinned official assets. Large
assets and raw local-path logs remain outside Git on the E drive. It is not package-consumer runtime
proof, not public redistribution approval, not post-publish proof, and not release-close approval.
No package publish, GitHub Release upload, or issue close was performed.
