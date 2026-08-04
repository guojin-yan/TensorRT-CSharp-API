# Sample Assets

This folder contains audit templates for asset-dependent samples. It does not contain model weights, labels, images, generated TensorRT engines, or private run logs.

## Demo model inventory

`demo-model-inventory.json` is the authoritative 10-model acquisition, conversion, outer `models` path, length, SHA256, article, and
source-tree runtime-evidence catalog. Every entry has a `runtimeEvidence` path to a tracked small JSON record; none of those links
permit model upload or public redistribution. `onnxtoengine-mnist-real-model-runtime-evidence.json` records the MNIST digit-7
TensorRT/ONNX Runtime match and the wrong-expected-digit controlled negative.
`tensorrtexec-refitted-plan-package-consumer-article-runtime-evidence.json` records the separate repository-external,
two-package local-feed run, its real stdout screenshot, the exact output hash, and the 53/53 strict validation result.

Run `eng/Sync-DemoOnnxModels.ps1 -VerifyOnly` to require all 10 ONNX files under
`<workspace-root>/models` and verify their pinned lengths and hashes without copying or publishing anything.

## YoloVision reference acquisition

`yolovision-reference-assets.json` records the exact TensorRT 10.11 local-installation files used by the current YOLOv8s detection runtime candidate. Run:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Acquire-YoloVisionReferenceAssets.ps1 `
  -SourceRoot $env:TENSORRT_PATH
```

The command verifies file length and SHA256 and can copy the files into a local evidence cache. It intentionally keeps `canPromoteRealModelRuntime=false` while any model, labels, or image license remains `owner-review-required`. Hash verification is not redistribution approval.

## Official YOLOX acquisition

`yolovision-yolox-official-assets.json` pins the official YOLOX-S 0.1.1rc0 ONNX model, Apache-2.0 license, dog image, COCO class source, and official preprocess/postprocess references. The acquisition script uses a repository-external `downloads/yolox-apache` workspace, creates deterministic PPM/labels derivatives, and writes a machine-readable report:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Acquire-YoloXOfficialAssets.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Acquire-YoloXOfficialAssets.ps1 -Offline
```

The corresponding strict source-tree runtime evidence is under `artifacts/yolovision/yolox-official-runtime`. It proves a real TensorRT enqueue and YOLOX decode, but it does not approve repository redistribution and is not package-consumer-runtime proof.

## Official YOLOv10 acquisition

`yolovision-yolov10-official-assets.json` pins the official THU-MIG YOLOv10n v1.1 ONNX model and AGPL-3.0 license. The acquisition script uses a repository-external `downloads/yolov10-agpl` workspace, verifies length and SHA256, and records whether the existing YOLOX-derived COCO labels and PPM input image are available for a source-tree runtime attempt:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Acquire-YoloV10OfficialAssets.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Acquire-YoloV10OfficialAssets.ps1 -Offline
```

This acquisition is not runtime proof or redistribution approval. The separate
`yolovision-yolov10n-local-package-consumer-runtime-evidence.json` record proves an isolated three-package restore/build/TensorRT run
with the fixed `[1,300,6]` contract, a CC0 result image, and a path-sanitized terminal screenshot. It does not claim an independent
raw-tensor comparison, public-feed availability, release authorization, or `AGPL-3.0-only` model redistribution approval.

## Official YOLOv8n Detection acquisition

`yolovision-yolov8n-det-official-assets.json` pins the Ultralytics `v8.3.0` detection weight, COCO YAML, source license, official `bus.jpg`, deterministic PPM, and exact 80-line labels. The acquisition script rejects C-drive output and performs no export, runtime, upload, or publish operation:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Acquire-YoloV8DetectionOfficialAssets.ps1 `
  -PythonPath <python-path>
```

`eng/Invoke-YoloVisionDetectionReference.py` validates `images:[1,3,640,640] -> output0:[1,84,8400]`, creates the raw ONNX Runtime reference for the exact C# letterbox tensor, retains a canonical image-pipeline comparison, and independently runs the Ultralytics/PyTorch NMS path. `yolovision-yolov8n-det-real-model-runtime-evidence.json` records the source-tree TensorRT 10.11 full-tensor, five-box, and controlled-negative results. Heavy assets remain in the repository-external workspace; package-consumer, public redistribution, and release claims remain false.

## Official YOLOv8n Pose acquisition

`yolovision-yolov8n-pose-official-assets.json` pins the Ultralytics `v8.3.0` release weight, source-commit license, human-containing `bus.jpg`, and deterministic P6 RGB PPM derivative. The acquisition script rejects C-drive output, verifies all lengths/SHA256 values, and never exports, runs, or publishes assets:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Acquire-YoloV8PoseOfficialAssets.ps1 `
  -PythonPath <python-path>
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Acquire-YoloV8PoseOfficialAssets.ps1 `
  -PythonPath <python-path> -Offline
```

The source-tree runtime record is `yolovision-yolov8n-pose-real-model-runtime-evidence.json`. Models, ONNX files, images, references, tensors, SVGs, and logs stay in the repository-external download workspace. The record is not package-consumer proof or public redistribution approval.

## Official YOLOv8n OBB acquisition

`yolovision-yolov8n-obb-official-assets.json` pins the official `yolov8n-obb.pt` Release asset ID/hash, source license, commit-pinned `boats.jpg`, deterministic P6 RGB PPM, and DOTA labels. The acquisition script keeps every heavy asset in the repository-external workspace and performs no export, runtime, upload, or publish operation:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Acquire-YoloV8ObbOfficialAssets.ps1 `
  -PythonPath <python-path>
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Acquire-YoloV8ObbOfficialAssets.ps1 `
  -PythonPath <python-path> -Offline
```

`yolovision-yolov8n-obb-real-model-runtime-evidence.json` records the source-tree TensorRT 10.11 case for `output0:[1,20,21504]`, the 430,080-value ONNX Runtime comparison, the independent Ultralytics/PyTorch rotated-box comparison, and the controlled negative reference mutation. It does not approve asset redistribution, package publication, or release.

## Official YOLOv8n Classification acquisition

`yolovision-yolov8n-cls-official-assets.json` pins the Ultralytics `v8.3.0` classification weight, source-commit ImageNet map/license/image, deterministic PPM and exact 1000-line labels. The acquisition script rejects C-drive output and performs no export, runtime, upload, or publish operation:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Acquire-YoloV8ClassificationOfficialAssets.ps1 `
  -PythonPath <python-path>
```

`eng/Invoke-YoloVisionClassificationReference.py` verifies the static `images:[1,3,224,224] -> output0:[1,1000]` Softmax graph, derives the authoritative Ultralytics center-crop tensor, compares PyTorch with ONNX Runtime, and writes positive plus controlled-negative structured references. `yolovision-yolov8n-cls-real-model-runtime-evidence.json` records the source-tree TensorRT 10.11 full-vector and Top-5 result. Models, ONNX, image, labels, tensors, references, SVGs, and logs stay in the repository-external workspace. The record is not package-consumer proof, public asset redistribution approval, or release authorization.

## Local Asset Layout

The current workspace convention is `<workspace-root>/models`, one level above the `TensorRtSharp4.0` Git repository. Every demo article must name the upstream acquisition method and ONNX conversion method. Converted ONNX files are staged under this outer directory until a separate Model Zoo exists; they are never committed to this repository.

`demo-model-inventory.json` is the complete first-release inventory for actual deep-learning demo models. It maps Classification,
OnnxToEngine/MNIST, YOLOv8 det/cls/seg/pose/OBB, YOLOv10n, YOLOX-S, and LRASPP semantic segmentation to acquisition sources,
conversion commands, outer `models` paths, lengths, SHA256 values, and articles. Placeholder paths for user-defined models and
code-generated identity networks are deliberately excluded. See
`docs/articles/zh-cn/demo-model-acquisition-and-onnx-conversion.md`.

Run `eng/Sync-DemoOnnxModels.ps1` after acquisition/export to materialize and hash-check every inventory entry under the outer
`models` directory. `-VerifyOnly` performs the same checks without copying. The script never uploads or publishes assets.

`classification-resnet18-official-assets.json` pins TorchVision `v0.25.0` ResNet18 `IMAGENET1K_V1`. Run
`eng/Acquire-TorchVisionResNet18OfficialAssets.ps1 -AllowDownload -ExportOnnx` to write its ONNX, ImageNet labels, and export report
under the outer `models\Classification` directory. None of those files is a repository or package asset.

`eng/Invoke-ClassificationResNet18Reference.py` consumes the exact C# image-preprocessing tensor and writes independent ONNX
Runtime raw logits, task probabilities, and a controlled-negative task reference. `classification-resnet18-real-model-runtime-evidence.json`
records the TensorRT 10.11 result: 1000/1000 raw logits and 1000/1000 probabilities matched, Top-5 order matched PyTorch/ORT, and
the one-value negative failed closed with exit code 1. The record does not approve weights/image redistribution or promote
package-consumer, public-package, post-publish, Owner release, or publication proof.

## Official torchvision LRASPP semantic acquisition

`yolovision-torchvision-lraspp-official-assets.json` pins torchvision `v0.25.0`, LRASPP MobileNetV3 Large weights, VOC labels metadata, the BSD-3-Clause license, and the PyTorch Hub dog image. Run:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Acquire-TorchVisionLrasppOfficialAssets.ps1 -AllowDownload
```

`eng/Invoke-YoloVisionSemanticReference.py --export-onnx` exports the fixed `images:[1,3,320,320] -> semantic:[1,21,320,320]` graph directly into the outer `models` directory. The runtime evidence compares 2,150,400 logits and 102,400 argmax pixels, and checks a controlled negative. Models, source images, references, and logs are not uploaded.

`eng/Test-YoloVisionSemanticLocalPackageConsumer.ps1` repeats the LRASPP runtime through a repository-external project that references only the managed API, YoloVision, and bridge-only packages. `yolovision-lraspp-semantic-local-package-consumer-runtime-evidence.json` fixes the three package hashes, zero-mismatch raw/class-index comparisons, and two fail-closed negatives. CUDA, cuDNN, and TensorRT remain user-installed dependencies; this local-feed record is not public-package or release proof.

`eng/Test-YoloVisionClassificationLocalPackageConsumer.ps1` runs the official YOLOv8n-cls graph through the same isolated three-package path. `yolovision-yolov8n-cls-local-package-consumer-runtime-evidence.json` records all 1,000 probability comparisons, the independent Top-5 order, a fail-closed single-value negative, and restored-package hash equality. The ONNX remains in the outer `models` directory and is not uploaded.

`eng/Test-YoloVisionPoseLocalPackageConsumer.ps1` runs the official YOLOv8n-pose graph through the isolated three-package path. `yolovision-yolov8n-pose-local-package-consumer-runtime-evidence.json` records byte-identical C# preprocessing, all 470,400 raw comparisons, four independent 17-keypoint pose comparisons, a fail-closed single-value negative, and restored-package hash equality. The ONNX remains in the outer `models` directory and is not uploaded.

`eng/Test-YoloVisionObbLocalPackageConsumer.ps1` runs the official YOLOv8n-obb graph through the isolated three-package path. `yolovision-yolov8n-obb-local-package-consumer-runtime-evidence.json` records byte-identical C# preprocessing, all 430,080 raw comparisons, 40 independent ship OBB comparisons with rotated IoU and angle errors, a fail-closed single-value negative, and restored-package hash equality. The ONNX remains in the outer `models` directory and is not uploaded.

`eng/Test-YoloVisionDetectionLocalPackageConsumer.ps1` runs the official YOLOv8n detection graph through the isolated three-package path. `yolovision-yolov8n-det-local-package-consumer-runtime-evidence.json` records byte-identical C# preprocessing, all 705,600 raw comparisons, four person and one bus comparison with source-space box IoU, a fail-closed single-value negative, and restored-package hash equality. The ONNX remains in the outer `models` directory for the future Model Zoo and is not uploaded.

`eng/Test-GpuAllocatorLocalPackageConsumer.ps1` copies `samples/GpuAllocator.PackageConsumer` into a repository-external workspace and restores only the managed API and matching bridge-only packages from local feeds. `gpu-allocator-local-package-consumer-tensorrt10.11-evidence.json` records package and bridge hashes, eight real builder callbacks, zero final live allocations, and fail-closed rejection and exception cases. The sample constructs an identity network in code, so model acquisition and ONNX conversion are explicitly not applicable. CUDA and TensorRT are host-installed; the record is not public-package, Release, or post-publish proof.

`eng/Test-OutputAllocatorLocalPackageConsumer.ps1` uses the same repository-external two-package harness for `samples/OutputAllocator.PackageConsumer`. `output-allocator-local-package-consumer-tensorrt10.11-evidence.json` records the package and restored bridge hashes, real `reallocateOutput` and `notifyShape` callbacks, paired CUDA allocation/release, zero live allocations after detach, and a rejection case that fails enqueue without allocating. Its identity network is created in code, so model acquisition, ONNX conversion, and image visualization are not applicable. This remains local-package evidence only.

`eng/Test-DebugListenerLocalPackageConsumer.ps1` applies the same repository-external two-package isolation to `samples/DebugListener.PackageConsumer`. `debug-listener-local-package-consumer-tensorrt10.11-evidence.json` records package and restored bridge hashes, a real `processDebugTensor` callback, copied `[1,4]` metadata, pointer isolation, clean detach, and a controlled handler rejection. TensorRT 10.11 records the rejected callback but completes this identity enqueue, so the proof uses failure state and lifecycle invariants instead of requiring an enqueue exception. No external model or image is involved, and this is not public-package, Release, or post-publish proof.

`yolovision-yolox-s-local-package-consumer-runtime-evidence.json` records the current three-package YOLOX-S TensorRT 10.11 run against a CC0 bus-station image. It fixes the official ONNX, local packages, preprocessed tensor, 1 bus plus 7 person detections, sanitized terminal transcript, terminal screenshot, and annotated result image. The heavy ONNX, engine, tensor, raw logs, and full path-bearing report remain outside Git. The run uses local file feeds and does not claim independent raw-tensor parity, public-package proof, redistribution approval, or release readiness.

Recommended local names:

| Sample | Local files |
| --- | --- |
| Classification | `classifier.onnx`, `classifier.labels.txt`, `classifier.input.png`, `classifier-evidence.sidecar.json`, `classifier-sample-run-evidence.json`, `classifier-run.log` |
| YoloVision | `yolo.onnx`, `coco.names`, `yolo.input.jpg`, `yolovision-evidence.sidecar.json`, `yolovision-sample-run-evidence.json`, `yolovision-run.log` |
| YOLOX-S walkthrough | `yolox_s.onnx`, `coco.names`, `yolox-test.jpg`, `yolox_s-evidence.sidecar.json`, `yolox_s-sample-run-evidence.json`, `yolox_s-run.log` |

`yolovision-family-task-real-asset-roadmap.json` is the cross-family planning artifact for YOLOv5/v6/v7/v8/v9/v10/v11/v26/custom and det/cls/seg/obb/pose/sem. It is an owner-action roadmap, not runtime proof. Use it to pick article/demo candidates and then create owner-filled asset manifests, sidecars, run logs, and sample-run-evidence records.

`yolovision-article-case-pack.json` is the publishable article case pack for YOLOv8n det/seg/pose/obb/cls/sem. It records model hints, export commands, YoloVision offline preflight commands/reports, TensorRtExec build-only commands, YoloVision run commands, required hashes, expected evidence lines, and the same owner-action proof boundary. A preflight report is `yolovision-preflight.v1`/`precheck` configuration evidence only; it cannot replace `YoloVision Passed=True` from a real run log.

`yolovision-real-asset-owner-backfill-pack.json` is the owner execution contract that follows the article case pack. It keeps the six YOLOv8n task cases (`det/seg/pose/obb/cls/sem`), but expands each row into family/task/article-entrypoint consistency, model/license/SHA256, labels, input image, preprocessed tensor, YoloVision preflight report/schema/hash/boundary, TensorRtExec report hash, engine hash, YoloVision run log hash, output JSON hash, stdout/stderr summaries, and owner review fields. Validate it with `eng/Test-YoloVisionRealAssetOwnerBackfillPack.ps1`; template rows stay `owner-action-required` and cannot promote real-model-runtime or package-consumer-runtime proof.

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

`cross-task-reference-provenance-contract.json` adds the independent-reference layer. It requires common model/input/tensor/provider/
reference/comparison/Owner fields plus a separate semantic profile for generic Classification and each YoloVision task. A reference
may cross a task boundary only when all six reuse fingerprints match and the Owner accepts both golden provenance and redistribution.
The current readiness projection is generated by `eng/Export-CrossTaskReferenceProvenanceMatrix.ps1`; a zero-ready-row result is an
honest Owner backlog, not a validator failure.

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
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CrossTaskReferenceProvenanceMatrix.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-CrossTaskReferenceProvenanceMatrix.ps1 -Strict
```

Never set `isSmokePassed=true`, write `Classification Passed=True`, or write `YoloVision Passed=True` until the real model, labels, image, hashes, license notes, sample run log, and validator output all agree. Sample assets and sample run evidence can promote only to `real-model-runtime`; `package-consumer-runtime` belongs to release proof records.
