# TensorRtExec Multi-Input Reference Runtime Evidence

## Result

- Host path: local Windows `win-x64`, TensorRT 10.11.0, CUDA 12.9.
- Generated ONNX graph: two float inputs (`left`, `right`) and two float outputs (`sum`, `difference`).
- Runtime shapes: every tensor is `2x4`; both inputs contain 8 float32 values / 32 bytes.
- Validated run: real build, serialize, deserialize, two measured enqueues, output readback, and two structured reference comparisons completed.
- `OutputValidated=true`, `IdentityOutputMatch=false`, both output mismatch counts are zero.
- Combined output bytes: 64 bytes, SHA256 `05080c5591955c003b781dba0170ad3aca244e3033b89e81569a39abe305e2c5`.

## Negative Gate

The second run loaded the serialized engine independently and changed only `difference[7]` in the reference by `0.25`.
Inference and readback still completed, but the result was `load-engine-reference-validation-failed`, `Success=false`,
`OutputValidated=false`, with one mismatch at index 7. This confirms that capture, reference-file hash, and runtime execution
do not bypass numerical validation.

## Boundary

The model, inputs, and references are generated locally and classified as synthetic. This is real TensorRT runtime evidence for
the multi-input/multi-output implementation, but it is not real-model-runtime, package-consumer-runtime, public-package,
post-publish, Linux, Owner-accepted, or release proof.
