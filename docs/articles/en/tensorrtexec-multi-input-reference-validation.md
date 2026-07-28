# TensorRtExec Multi-Input Reference Output Validation

## Runtime Contract

The generic bounded runtime supports one or more float inputs and outputs. Input preparation, binding, capture, and artifacts
preserve `TensorRtEngineBindingReport` engine order. When `--loadInputs` is present, its `tensor:path` mappings must cover every
input. Missing, duplicate, and unknown names fail closed. Without the option, deterministic values are generated independently
for every input.

```powershell
TensorRtExec `
  --loadEngine .\model.plan `
  --shapes "left:2x4,right:2x4" `
  --loadInputs "left:.\left.bin,right:.\right.bin" `
  --referenceOutputs "sum:.\sum.reference.json,difference:.\difference.reference.json" `
  --referenceAbsTolerance 1e-5 `
  --referenceRelTolerance 1e-4 `
  --referenceNaNPolicy reject `
  --referenceInfinityPolicy exact `
  --exportOutput .\output.json
```

Each `InputTensors` entry records the name, shape, element and byte counts, an eight-value bounded preview, SHA256, source
classification, and source path. These are managed copies and never expose device pointers or borrowed handles.

## Reference JSON

Each output maps to one traceable structured document:

```json
{
  "schemaVersion": 1,
  "tensorName": "sum",
  "shape": [2, 4],
  "values": [2.25, 5.25, 8.25, 11.25, 14.25, 17.25, 20.25, 23.25],
  "sourceClassification": "synthetic-generated"
}
```

Validation checks mapping, file readability, tensor name, shape, element count, and then every value. Finite values pass when
either the absolute tolerance or the scale-aware relative tolerance succeeds. NaN defaults to `reject`; `equal` accepts only a
NaN/NaN pair. Infinity `exact` requires identical signs, while `reject` rejects any infinity. Per-tensor artifacts include the
reference path/hash/source, actual and reference shapes/counts, mismatch count, first mismatch, maximum errors, and diagnostic.

## Evidence Boundary

`OutputValidated` becomes true only when references cover every engine output and every comparison passes. The legacy identity
shortcut remains separately visible as `IdentityOutputMatch` and cannot set reference validation by itself. Capture, raw bytes,
a reference SHA256, or a single passing output are not all-output numerical correctness.

The repository TRT10/CUDA12.9 smoke builds and runs a generated two-input/two-output Add/Sub ONNX model, then changes only
`difference[7]` by `0.25` and verifies the `load-engine-reference-validation-failed` result. This is synthetic runtime evidence,
not real-model, package-consumer, public-package, post-publish, or release proof.
