# Classification

This sample runs a user-provided single-input float ONNX classifier through TensorRT and prints Top-K scores.

The repository does not bundle model, label, or image assets because those files have separate licensing and size constraints. The sample uses a synthetic input tensor by default, so it validates the deployment pipeline. Replace the input generation with your own preprocessing when you wire it into an application.

```powershell
dotnet run --project .\samples\Classification -- `
  --model .\models\classifier.onnx `
  --labels .\models\labels.txt `
  --input-shape 1x3x224x224 `
  --tensor-rt-line 10 `
  --top-k 5
```

For dynamic classifiers, provide profile bounds:

```powershell
dotnet run --project .\samples\Classification -- `
  --model .\models\classifier.onnx `
  --input-shape 1x3x224x224 `
  --min-shape 1x3x224x224 `
  --opt-shape 4x3x224x224 `
  --max-shape 8x3x224x224
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

## Evidence Lines

- `Classification TensorRtLine=...`
- `Input=... Output=...`
- `TopK Index=... Label=... Score=...`
- `Classification Passed=True`
