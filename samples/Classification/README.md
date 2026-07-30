# Classification

This sample runs a user-provided float ONNX classifier through TensorRT and prints Top-K scores. The ordinary image path remains
single-input, while custom classifiers may bind multiple float inputs explicitly by name.

The repository does not bundle model, label, or image assets because those files have separate licensing and size constraints. The sample uses a synthetic input tensor by default, so it validates the deployment pipeline only.

```powershell
dotnet run --project .\samples\Classification -- `
  --model .\models\classifier.onnx `
  --labels .\models\labels.txt `
  --input-shape 1x3x224x224 `
  --tensor-rt-line 10 `
  --top-k 5
```

For a real BMP or PPM image, use the built-in resize/crop and normalization path:

```powershell
dotnet run --project .\samples\Classification -- `
  --model .\models\classifier.onnx `
  --labels .\models\labels.txt `
  --image .\models\input.ppm `
  --preprocessed-output .\artifacts\classification\input-f32.bin `
  --input-shape 1x3x224x224 `
  --image-resize shorter-side-center-crop `
  --resize-shorter-side 256 `
  --tensor-layout NCHW `
  --color-order RGB `
  --scale 0.0039215689 `
  --mean 0.485,0.456,0.406 `
  --std 0.229,0.224,0.225 `
  --score-transform softmax `
  --top-k 5 `
  --output-json .\artifacts\classification\output.json
```

`--image` accepts uncompressed 24/32-bit BMP and P3/P6 PPM/PNM files. `--input <path>` is different: it reads exactly one raw byte per tensor element and normalizes each byte to `[0,1]`. `--input-data <path>` reads an already-preprocessed float32 binary or text tensor. Externally preprocessed JPG/PNG inputs must therefore be decoded by the caller and passed with `--input-data`.

For dynamic classifiers, provide profile bounds:

```powershell
dotnet run --project .\samples\Classification -- `
  --model .\models\classifier.onnx `
  --input-shape 1x3x224x224 `
  --min-shape 1x3x224x224 `
  --opt-shape 4x3x224x224 `
  --max-shape 8x3x224x224
```

For a custom multi-input classifier, replace all singular input/profile options with complete named maps. Every input must have
exactly one source; missing, duplicate, and unknown names fail before engine execution.

```powershell
dotnet run --project .\samples\Classification -- `
  --model .\models\image-and-metadata.onnx `
  --input-shapes "images:1x3x224x224,metadata:1x8" `
  --min-shapes "images:1x3x224x224,metadata:1x8" `
  --opt-shapes "images:1x3x224x224,metadata:1x8" `
  --max-shapes "images:4x3x224x224,metadata:4x8" `
  --load-inputs "images:.\models\image.fp32.bin,metadata:.\models\metadata.txt" `
  --output-name logits
```

## Required Assets For A Full Demo

- `model.onnx`: an image-classification ONNX model with documented input tensor name, layout, data type, and normalization.
- `labels.txt`: one label per output class.
- `input.*`: a small image with redistribution rights.
- Preprocessing metadata: resize policy, crop policy, RGB/BGR order, scale, mean, and standard deviation.

## Asset Metadata Checklist

- Model source URL and license.
- Model SHA256 and opset.
- Input tensor name, shape, layout, and dtype.
- Output tensor name, shape, and class count.
- Label source, license, and line count.
- Test image source, license, and preprocessing notes.

When the sample runs with the default synthetic tensor, it is pipeline evidence only. It does not prove image classification accuracy for a real model.

See `docs/articles/zh-cn/classification-model-assets.md`, `docs/articles/zh-cn/classification-asset-candidates.md`, and `samples/assets/classification-assets.template.json` for the release-facing asset checklist.

## Real Model Evidence Backfill

Use an evidence sidecar when you move from a candidate manifest to a real model record:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OnnxEngineBuildEvidenceSidecarTemplate.ps1
```

Start from `artifacts/user-acceptance/onnx-engine-build-evidence-sidecar.classification.template.json`, copy the relevant fields into `models/classifier-evidence.sidecar.json`, then run the build-only command from `samples/assets/classification-assets.template.json` with `--evidenceSidecar`.

After that, run this sample with the real model, labels, image, and preprocessing metadata. Record `Classification Passed=True`, stdout/stderr summaries, the run log SHA256, model SHA256, labels SHA256, input image SHA256, and license notes in the asset manifest. Validate the owner backfill with:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-SampleRunEvidenceRecordTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OnnxEngineBuildEvidenceSidecar.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleAssetManifest.ps1
```

Use `artifacts/user-acceptance/sample-run-evidence-record.classification.template.json` as the owner-facing place for the real runner log path, `sampleRunLogSha256`, `stdoutSummary`, `stderrSummary`, and `canPromoteRealModelRuntime=false/true` decision. The sidecar enriches TensorRtExec/OnnxToEngine reports. The sample run evidence record connects the real `Classification` runner log to the asset manifest. Neither one can promote a build-only report to `package-consumer-runtime`.

## Reference Provenance Contract

The cross-task contract is `samples/assets/cross-task-reference-provenance-contract.json`. A Classification reference must record
the exact model, labels, input image, preprocessed tensor, input/output tensor contract, independent framework/provider, comparison
policy, and Owner decision. Its task semantics must also identify resize, crop, RGB/BGR order, scale, mean/std, whether outputs are
raw logits or probabilities, the score transform, label mapping SHA256, Top-K, and argmax rule.

Run the current-readiness audit with:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CrossTaskReferenceProvenanceMatrix.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-CrossTaskReferenceProvenanceMatrix.ps1 -Strict
```

The generic Classification sample and YoloVision `cls` task are separate profiles. A MNIST reference cannot be reused for either
profile unless model/input/preprocess/output/labels/task-semantics fingerprints all match and an Owner separately accepts golden
provenance and redistribution. Synthetic input, raw output hashes, same-runtime references, and local package feeds do not satisfy
that gate.

## Output And Reference JSON

Use `--output-json <path>` to write a `classification-output.v1` report containing ordered `inputTensors`, input/preprocessing fingerprints, raw and transformed output values, stable Top-K predictions, runtime summary, optional task-specific and raw-runtime reference comparisons, and an explicit non-proof boundary. Its schema is `samples/Classification/classification-output.schema.json`.

Use `--reference-output <path>` with `--reference-abs`, `--reference-rel`, `--reference-nan-policy reject|equal`, and `--reference-infinity-policy exact|reject` to compare the selected raw-logit or softmax output. The reference format is described by `samples/Classification/classification-reference.schema.json` and requires all six lowercase SHA256 fingerprints:

- `modelSha256`
- `inputTensorSha256`
- `preprocessContractSha256`
- `outputTensorContractSha256`
- `labelsSha256`
- `taskSemanticsSha256`

For `--image`, the preprocessing contract hash is generated from resize/crop, layout, color order, scale, mean/std, and interpolation settings. For `--input` or `--input-data`, a reference comparison also requires the caller to supply the exact `--preprocess-contract-sha256`; the runner does not infer preprocessing semantics from tensor bytes.

Metadata mismatches stop comparison with `Completed=false`. Value mismatches complete comparison with `Completed=true, Passed=false`, and the process returns exit code 1. A passing comparison remains a task-specific candidate: `boundary.ownerReviewedGolden`, `isPackageConsumerRuntimeProof`, `isPublicPackageProof`, `isPostPublishProof`, `canPublishPublicly`, and `canCloseReleaseIssue` all remain `false` until their separate Owner and public-package evidence exists.

`--reference-outputs "tensor:path,..."` adds raw runtime validation for every captured output. It uses
`--reference-abs-tolerance`, `--reference-rel-tolerance`, and the same NaN/Infinity policies. This generic tensor check is separate
from `--reference-output`, which validates Classification preprocessing, labels, score transform, and task semantics. When both
are requested, both must pass. Its JSON schema is `samples/JYPPX.SampleSupport/onnx-sample-reference.schema.json`. A raw tensor
reference by itself cannot replace the richer Classification provenance contract.

## Evidence Lines

- `Classification TensorRtLine=...`
- `Input=... Output=...`
- `ImagePreprocess Source=... TensorSha256=...`
- `ScoreTransform=... ValueKind=... OutputSha256=...`
- `TopK Index=... Label=... Score=...`
- `ClassificationReference Requested=True Completed=... Passed=...`
- `OutputJson=...`
- `Classification Passed=True`
