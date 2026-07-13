# Sample Assets

This folder contains audit templates for asset-dependent samples. It does not contain model weights, labels, images, generated TensorRT engines, or private run logs.

## YoloVision reference acquisition

`yolovision-reference-assets.json` records the exact TensorRT 10.11 local-installation files used by the current YOLOv8s detection runtime candidate. Run:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Acquire-YoloVisionReferenceAssets.ps1 `
  -SourceRoot $env:TENSORRT_PATH
```

The command verifies file length and SHA256 and can copy the files into a local evidence cache. It intentionally keeps `canPromoteRealModelRuntime=false` while any model, labels, or image license remains `owner-review-required`. Hash verification is not redistribution approval.

## Local Asset Layout

Keep large or license-sensitive files in a local `models\` folder at the repository root, or in another owner-controlled path. Do not commit model weights, downloaded images, `.plan` engines, private build reports, or run logs unless their license and size have been explicitly approved.

Recommended local names:

| Sample | Local files |
| --- | --- |
| Classification | `classifier.onnx`, `classifier.labels.txt`, `classifier.input.png`, `classifier-evidence.sidecar.json`, `classifier-sample-run-evidence.json`, `classifier-run.log` |
| YoloVision | `yolo.onnx`, `coco.names`, `yolo.input.jpg`, `yolovision-evidence.sidecar.json`, `yolovision-sample-run-evidence.json`, `yolovision-run.log` |
| YOLOX-S walkthrough | `yolox_s.onnx`, `coco.names`, `yolox-test.jpg`, `yolox_s-evidence.sidecar.json`, `yolox_s-sample-run-evidence.json`, `yolox_s-run.log` |

`yolovision-family-task-real-asset-roadmap.json` is the cross-family planning artifact for YOLOv5/v6/v7/v8/v9/v10/v11/v26/custom and det/cls/seg/obb/pose/sem. It is an owner-action roadmap, not runtime proof. Use it to pick article/demo candidates and then create owner-filled asset manifests, sidecars, run logs, and sample-run-evidence records.

`yolovision-article-case-pack.json` is the publishable article case pack for YOLOv8n det/seg/pose/obb/cls/sem. It records model hints, export commands, TensorRtExec build-only commands, YoloVision run commands, required hashes, expected evidence lines, and the same owner-action proof boundary. It is useful for public article drafting and owner backfill, but it cannot replace `YoloVision Passed=True` from a real run log.

`yolovision-real-asset-owner-backfill-pack.json` is the owner execution contract that follows the article case pack. It keeps the six YOLOv8n task cases (`det/seg/pose/obb/cls/sem`), but expands each row into model/license/SHA256, labels, input image, preprocessed tensor, TensorRtExec report hash, engine hash, YoloVision run log hash, output JSON hash, stdout/stderr summaries, and owner review fields. Validate it with `eng/Test-YoloVisionRealAssetOwnerBackfillPack.ps1`; template rows stay `owner-action-required` and cannot promote real-model-runtime or package-consumer-runtime proof.

`eng/Export-YoloVisionRealAssetOwnerBackfillPack.ps1` projects the article case pack and owner backfill pack into drift-checked artifacts: `samples/assets/yolovision-real-asset-owner-backfill-pack.generated.json`, `artifacts/yolovision/yolovision-real-asset-owner-backfill-pack-projection-report.json`, and `artifacts/user-acceptance/yolovision-real-asset-owner-backfill-sample-run-evidence.template.json`. The exporter never overwrites owner proof and the generated sample-run-evidence template remains `template-only` until real logs, hashes, stdout/stderr summaries, and owner review are supplied.

`eng/Export-YoloVisionRealAssetOwnerProofInputTemplate.ps1`, `eng/Test-YoloVisionRealAssetOwnerProofInput.ps1`, and `eng/Import-YoloVisionRealAssetOwnerProofInput.ps1` are the next owner handoff layer. They create a six-task owner input contract aligned with `yolovision-task-output-contract.json`, validate real owner hashes/logs/host metadata/review fields, and project a sample-run-evidence candidate. The candidate may become `real-model-runtime` only after the real logs and hashes pass validation; it is never `package-consumer-runtime` proof and never closes release proof by itself.

Commit these files only after owner review:

- small JSON templates or sanitized real evidence records
- documentation describing model source, license, commands, and verification status
- tiny redistributable labels or images only when their license permits it

Keep these local by default:

- ONNX models and TensorRT `.plan` engines
- downloaded images
- raw stdout/stderr logs
- files containing local paths, machine names, tokens, or private package feeds

## Hashes

Record SHA256 values before promoting any sample to real model runtime evidence:

```powershell
Get-FileHash .\models\classifier.onnx -Algorithm SHA256
Get-FileHash .\models\classifier.labels.txt -Algorithm SHA256
Get-FileHash .\models\classifier.input.png -Algorithm SHA256
Get-FileHash .\models\classifier-run.log -Algorithm SHA256
```

Use the same command for `yolo.onnx`, `coco.names`, `yolo.input.jpg`, and `yolovision-run.log`.

## Evidence Chain

The asset manifest, sidecar, and sample run evidence record have different jobs:

- `samples\assets\*.json` records expected model, labels, input, tensor metadata, preprocessing, postprocessing, commands, and proof classification.
- `*-evidence.sidecar.json` connects a TensorRtExec or OnnxToEngine build report to model evidence such as model SHA256, input SHA256, license, and stdout/stderr summary.
- `*-sample-run-evidence.json` connects a real `Classification` or `YoloVision` runner command to the run log, log SHA256, stdout/stderr summary, expected evidence lines, and `canPromoteRealModelRuntime`.

Missing sidecars and missing sample run evidence records in templates are `owner-action-required`, not errors. They also do not make the sample ready.

## Required Validation

After owner backfill, run:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OnnxEngineBuildEvidenceSidecar.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-YoloVisionRealAssetOwnerBackfillPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-YoloVisionRealAssetOwnerProofInputTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionRealAssetOwnerProofInput.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-YoloVisionRealAssetOwnerProofInput.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -InputPath .\models\classifier-sample-run-evidence.json
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleAssetManifest.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-UserAcceptanceSampleCatalog.ps1
```

Never set `isSmokePassed=true`, write `Classification Passed=True`, or write `YoloVision Passed=True` until the real model, labels, image, hashes, license notes, sample run log, and validator output all agree. Sample assets and sample run evidence can promote only to `real-model-runtime`; `package-consumer-runtime` belongs to release proof records.
