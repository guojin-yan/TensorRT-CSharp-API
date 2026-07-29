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

## MNIST Reference Candidate

The local repository also applies the same contract to the existing TensorRT 10.11/CUDA 12.9 MNIST digit-7 assets. The model
is supplied under TensorRT `data/mnist`, whose README identifies ONNX Model Zoo as its source. The input is the `7.pgm`
asset preprocessed as the `Input3` float32 tensor with `1-pixel/255`. The reference contains the ten `[1,10]`
`Plus214_Output_0` logits and declares `repository-mnist-runtime-output-derived-unreviewed` as its source classification.

The same-runtime reference was copied from a prior TensorRT output, so it remains a regression-consistency input. Both the
source-tree build and independent `--loadEngine` paths recorded `OutputValidated=true`, 10/10 comparisons, and zero mismatches,
with maximum absolute/relative errors of `9.536743e-07` / `1.3443339e-06` under `1e-4` tolerances. The isolated local
`PackageReference` consumer compares the same reference, and the strict evidence validator records 53 passing checks. This layer
is retained in `artifacts/interface-coverage/tensorrtexec-mnist-reference-validation-evidence.json`.

## Independent ONNX Runtime CPU Candidate

An isolated producer also runs the same ONNX model and input with ONNX Runtime `1.23.2` and an explicit
`CPUExecutionProvider`. The runner copies four already-cached `.nupkg` files into a temporary E-drive feed, clears every remote
source in `NuGet.Config`, and keeps its restore cache and `DOTNET_CLI_HOME` in the isolated E-drive workspace. ONNX Runtime is not
added to the main solution. Two ORT executions produced byte-identical float32 output with raw SHA256
`a20932857fb2d51f5f0b79daa211140fce631b3f67787b69e0a23f33c8817d75` and predicted digit 7. The profiling trace contains only
`CPUExecutionProvider`, proving an execution path independent from TensorRT.

The ORT reference is classified as `onnxruntime-cpu-1.23.2-derived-unreviewed` and has SHA256
`1babfa81c0d277a3d483fdc6288285b26008ba5b382868272f1e2187338bd571`. Its comparison against the retained TensorRT logits covers
10/10 values with zero mismatches; maximum absolute/relative errors are `5.722046e-06` / `5.7323444e-07`, both within `1e-4`
tolerances. The compact evidence and validation summary use the `tensorrtexec-mnist-onnxruntime-reference-*` files under
`artifacts/interface-coverage`; the capture-host validator passes 51/51 when raw/profile/log artifacts are required. A clean clone
needs only the checked-in reference, sidecar, compact evidence, and validation summary.

## Controlled Negative Runtime

Five malformed variants are derived from the independent ORT candidate: tensor-name mismatch, shape mismatch, value-count
mismatch, NaN under the reject policy, and infinity under the reject policy. Each variant runs through both the source-tree CLI
and an isolated local-feed `PackageReference` consumer, for ten real TensorRT enqueue/readback executions. Both paths first capture
the same raw output SHA256 and then exit nonzero with `OutputValidated=false`; the consumer also records
`OwnerScopeExited=true`. The three metadata cases end with `Completed=false`, while the two special-value cases end with
`Completed=true`, one mismatch, and first mismatch index 0. The strict validator passes 72/72 checks.

These evidence layers remain distinct: the same-runtime reference is a regression baseline, the ORT CPU output is an independent
framework candidate, and the malformed variants prove fail-closed validation only. TensorRT sample terms remain license-review
input, and no Owner has approved repository redistribution of the model/input/reference or accepted the ORT candidate as golden.
Local feeds, CPU profiling, real GPU enqueue, and hashes are not public-package, post-publish, Owner-accepted real-model, or release
proof.
