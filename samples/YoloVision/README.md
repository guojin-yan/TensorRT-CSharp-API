# YoloVision

This sample runs a user-provided single-input float YOLO-family ONNX model through TensorRT, then decodes common output layouts and task profiles. The managed postprocess base now has a unified `YoloVisionResult` path for detection, classification, segmentation, OBB, pose, and semantic segmentation. Detection-style tasks share box decode, score filtering, and class-aware/class-agnostic NMS; classification and semantic segmentation have single-output decoders; segmentation, OBB, and pose also have pure managed multi-output helpers for model-specific auxiliary tensors.

- `[1, 84, 8400]` style channel-first output
- `[1, 8400, 84]` style box-first output
- YOLOv5/v6/v7/v8/v9/v10/v11/v26 family labels
- task labels: `det`, `cls`, `seg`, `obb`, `pose`, `sem`
- application-side confidence filtering, class-aware/class-agnostic NMS helpers
- dedicated YOLOv10-style `[1,N,6]` end-to-end `x1,y1,x2,y2,score,classId` decode without a second NMS pass, plus mask coefficient/prototype compose, pose keypoint, OBB angle, semantic map, and multi-output metadata helpers

`YoloVision` is the unified YOLO-family sample for detection, classification, segmentation, OBB, pose, and semantic segmentation. It is intentionally broader than detection: the same sample documents family/task selection, multi-output metadata, managed postprocess helpers, and real-asset evidence requirements across the supported YOLO-family tasks.

Start with `docs/articles/zh-cn/yolovision-all-task-overview.md` for the three-layer capability/evidence model and six-task workflow. For detection raw heads, YOLOv10 end-to-end output, YOLOX grid/stride decode, numeric fail-closed behavior, and the current source-image coordinate boundary, use `docs/articles/zh-cn/yolovision-detection-tutorial.md`.

## Offline Preflight

Run the deterministic YOLOv10 six-column managed decoder smoke without CUDA, TensorRT, ONNX, model files, or labels:

```powershell
dotnet run --project .\samples\YoloVision -- --self-test-end2end
```

The command must report `ManagedSmoke=YOLOv10EndToEnd Passed=True`, two retained detections, `ApplyNms=False`, and
`ManagedSmokeBoundary=managed-array-decode-only`. This proves the managed array contract only; it is not TensorRT
execution, real-model-runtime, or package-consumer-runtime proof.

Use `--preflight` when preparing an owner handoff or article case and the TensorRT runtime is not available yet. It parses the family/task/profile and output metadata, records model/labels/input existence and SHA256 values when files are present, and writes a `yolovision-preflight.v1` report. It does not open TensorRT, parse ONNX, build an engine, load plugins, or enqueue inference.

```powershell
dotnet run --project .\samples\YoloVision -- `
  --preflight `
  --family v8 `
  --task seg `
  --model .\models\yolov8n-seg.onnx `
  --labels .\models\coco.names `
  --input-data .\models\yolov8n-seg-fp32.bin `
  --input-shape 1x3x640x640 `
  --output-role-map boxes:det,proto:mask-prototypes `
  --mask-coefficient-count 32 `
  --preflight-report .\artifacts\yolovision\yolov8n-seg-preflight.json
```

`state=ready-for-runtime-precheck` means the supplied preflight inputs and metadata are present; `owner-action-required` means the command is syntactically usable but an owner still needs to supply model assets, labels, input tensors, or task metadata; `invalid` is reserved for `--strict-preflight` blockers. `--dryRun` and `--previewOnly` are aliases. The report's execution flags remain false and its boundary is always `proofClassification=precheck`, `isRuntimeProof=false`, and `canPromoteRealModelRuntime=false`. The schema is `samples/YoloVision/yolovision-preflight.schema.json`.

The repository does not bundle detector models, label files, or images because those assets have separate licensing and size constraints. The sample uses a synthetic input tensor by default, so detections are useful as pipeline evidence rather than as image-quality evidence.

For real image evidence, either preprocess the image outside the runner into the model's exact tensor layout and pass the tensor with `--input-data`, or use `--image` for the built-in `.bmp` / `.ppm` preprocessing path. The built-in path decodes the image, applies stretch or letterbox resize, RGB/BGR channel order, optional normalization, NCHW/NHWC layout, writes a float32 tensor to `--preprocessed-output`, and feeds that tensor through the same `--input-data` runtime path. The runner accepts float32 `.bin`/`.raw` files or comma/space/newline separated text with exactly `N*C*H*W` values. `--input` is intentionally narrower: it accepts a raw byte tensor with the same element count and normalizes bytes to `[0,1]`.

Use `--preprocess-only` when preparing evidence or a `trtexec --loadInputs` run before the TensorRtSharp bridge is available:

```powershell
dotnet run --project .\samples\YoloVision -- `
  --preprocess-only `
  --image .\models\dog.ppm `
  --preprocessed-output .\models\yolo-preprocessed-fp32.bin `
  --input-shape 1x3x640x640 `
  --tensor-layout NCHW `
  --color-order RGB `
  --resize letterbox
```

The command prints `ImagePreprocess`, `ImagePreprocessConfig`, source/tensor SHA256 values, source/target dimensions, resize scale, padding, normalization, layout, color order, and output element count. This is preprocessing evidence only; it is not a model runtime pass until the tensor is consumed by a successful TensorRT enqueue.

```powershell
dotnet run --project .\samples\YoloVision -- `
  --model .\models\yolo.onnx `
  --labels .\models\coco.names `
  --image .\models\dog.ppm `
  --preprocessed-output .\models\yolo-preprocessed-fp32.bin `
  --input-shape 1x3x640x640 `
  --tensor-rt-line 10 `
  --family v8 `
  --task det `
  --layout auto `
  --has-objectness auto `
  --nms-mode class-aware `
  --confidence 0.25 `
  --iou-threshold 0.45 `
  --output .\artifacts\yolovision\yolo-output.json
```

## YOLO Series And Task Capability Matrix

Print the offline capability matrix without TensorRT runtime, CUDA, ONNX model assets, or labels:

```powershell
dotnet run --project .\samples\YoloVision -- --list-capabilities
dotnet run --project .\samples\YoloVision -- --list-capabilities --json
dotnet run --project .\samples\YoloVision -- --self-test-capabilities
```

The matrix currently covers `custom`, YOLOv5/v6/v7/v8/v9/v10/v11/v26, detection-only YOLOX, and task aliases `det`, `cls`, `seg`, `obb`, `pose`, and `sem`. It records supported and unsupported family/task boundaries, the managed decode path, required auxiliary metadata, and evidence level for each pair:

| Task | Alias | Decode path | Required metadata | Evidence level |
| --- | --- | --- | --- | --- |
| Detection | `det` | Single-output boxes with score filtering and class-aware/class-agnostic NMS | none | runtime-smoke-ready |
| Classification | `cls` | Single-output logits/top-k classification decoder | none | managed-smoke-ready |
| Segmentation | `seg` | Detection rows plus mask prototype composition | mask coefficient count, prototype tensor role, optional auxiliary channel start/layout | managed-metadata-ready |
| Oriented bounding box | `obb` | Detection rows plus angle tensor conversion | angle tensor role, degrees/radians flag, optional auxiliary layout | managed-metadata-ready |
| Pose | `pose` | Detection rows plus keypoint tensor mapping | keypoint count, keypoint stride, optional auxiliary layout | managed-metadata-ready |
| Semantic segmentation | `sem` | Single-output semantic map decoder | class count and semantic tensor role | managed-smoke-ready |

This is a support matrix and smoke surface, not proof that a specific external model has passed real image validation. Real model promotion still requires a model/license manifest, TensorRtExec build sidecar, `YoloVision Passed=True` run log, stdout/stderr summaries, SHA256 values, and owner-reviewed evidence.

`--self-test-capabilities` validates the matrix JSON/table contract offline: 60 family/task rows, 55 supported rows, 5 explicit YOLOX unsupported task rows, and a proof boundary with `IsRuntimeProof=False`. It is a capability contract self-test only; it does not open TensorRT, build an engine, enqueue inference, or promote real-model/package-consumer runtime proof.

For YOLOv10 NMS-free/end-to-end exports, pass `--layout end2end`. The managed decoder requires a batch-1 `[1,N,6]` tensor whose columns are `x1,y1,x2,y2,score,classId`; it validates the six-column contract, converts `xyxy` coordinates to the shared center/width/height representation, filters by confidence, checks class bounds, and deliberately disables application-side NMS. It does not guess that an arbitrary YOLOv10 ONNX uses this contract. Inspect the real ONNX outputs first, and use the generic metadata-driven path when the exporter returns raw heads or a different column order. See `docs/articles/zh-cn/yolovision-yolov10-end-to-end-output-guide.md`.

The official THU-MIG YOLOv10n v1.1 ONNX path is now backed by source-tree `real-model-runtime` evidence for `[1,300,6]` output. Run `eng/Acquire-YoloV10OfficialAssets.ps1` to acquire hash-pinned AGPL-3.0 assets on the E drive. The closure record is `artifacts/interface-coverage/yolov10-official-runtime-proof-closure.json`; it proves a local source-tree TensorRT enqueue and managed end-to-end decode only, not package-consumer-runtime, public redistribution approval, or publish readiness.

The official YOLOX-S path is now backed by source-tree `real-model-runtime` evidence. `--family yolox` is detection-only and defaults to NCHW, BGR, raw `0..255` float values, fill 114, and top-left letterbox. Its `[1,8400,85]` raw output is transformed with `(xy + grid) * stride` and `exp(wh) * stride` for strides 8/16/32 before objectness scoring and NMS. Run `eng/Acquire-YoloXOfficialAssets.ps1` to acquire hash-pinned assets on the E drive, then follow `docs/articles/zh-cn/yolovision-yolox-official-runtime-tutorial.md`. This proof is not package-consumer-runtime and does not approve public asset redistribution.

## Local PackageReference Consumer

`YoloVision.csproj` also packs as `JYPPX.TensorRT.CSharp.API.YoloVision`. The package exposes the pointer-free `YoloVisionCommand.Run(string[] args)` entry so a repository-external application can reuse the same CLI, preprocessing, decode, NMS, report, and visualization path without a `ProjectReference`. The committed consumer template is `samples/YoloVision.PackageConsumer`; run the full clean E-drive restore/build/runtime validation with:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionLocalPackageConsumer.ps1 -PackageVersion 4.0.0
```

The script uses only local file feeds, puts its isolated NuGet cache and temporary project on the E drive, requires `ProjectReferenceCount=0` and `YoloVision Passed=True`, then removes the workspace. Its result is `local-package-consumer-runtime`, not public `package-consumer-runtime`, public redistribution approval, or post-publish proof. See `docs/articles/zh-cn/yolovision-yolox-local-package-consumer-tutorial.md`.

When a runtime run reaches `YoloVisionOutputReport`, the JSON now includes optional `bindingMetadata` copied from the existing `TensorRtEngineBindingReport`. It records engine/profile identity, enqueue readiness, tensor index/name, input/output mode, semantic output role, data type, engine/profile shapes, location, format, vectorization, byte-size fallback state, and non-fatal diagnostics. The console emits the same pointer-free summary as `BindingReport` and `BindingMetadata` lines. This is deployment metadata and does not promote the run to real-model-runtime or package-consumer-runtime proof.

The machine-readable task/output contract is `samples/YoloVision/yolovision-task-output-contract.json`. It keeps task names, output roles, required metadata, TensorRtExec profile hints, article entrypoints, and promotion boundaries in one place so docs, owner asset packs, and validators do not drift. The contract is still planning evidence only: it is not `real-model-runtime` proof, not `package-consumer-runtime` proof, and not a replacement for owner-filled run logs and hashes.

For the shared classification and semantic-segmentation workflow, including E-drive asset isolation, output-layout decisions, Top-K versus pixel argmax, build/preflight/runtime commands, report validation, and proof boundaries, see `docs/articles/zh-cn/yolovision-classification-semantic-tutorial.md`.

For instance segmentation, see `docs/articles/zh-cn/yolovision-segmentation-tutorial.md`. The managed multi-output path preserves detection source indices, composes embedded coefficients with `[P,H,W]` / `[1,P,H,W]` prototypes, applies a stable sigmoid, accepts `--mask-threshold`, reports active versus total prototype-grid pixels, and emits a bounded probability-mask SVG preview. The opt-in `--mask-spatial-transform` path additionally requires `--image` and `--mask-coordinate-space model-input|normalized`; it uses the exact preprocessing metadata for bilinear source-image resize-back and optional half-open detection-box crop. It never infers coordinates from an external tensor, and owner validation of exporter-specific mask alignment remains required.

For cross-family case planning, use `samples/assets/yolovision-family-task-real-asset-roadmap.json` and `docs/articles/zh-cn/yolovision-family-task-real-asset-roadmap.md`. That roadmap turns the broad matrix into owner-action candidate rows for YOLOv5/v6/v7/v8/v9/v10/v11/v26/custom, but it remains planning material until real assets and logs are backfilled.

For publishable YOLOv8n article cases, use `samples/assets/yolovision-article-case-pack.json`. The pack covers det, seg, pose, OBB, cls, and sem with export commands, YoloVision offline preflight commands/reports, TensorRtExec build-only commands, YoloVision run commands, required SHA256 fields, and expected evidence lines. A preflight report is `yolovision-preflight.v1`/`precheck` configuration evidence only; the pack remains article/template material until owner-provided assets and a real `YoloVision Passed=True` run log are validated.

For owner execution, use `samples/assets/yolovision-real-asset-owner-backfill-pack.json` and `eng/Test-YoloVisionRealAssetOwnerBackfillPack.ps1`. That pack expands each YOLOv8n case into family/task/article-entrypoint consistency plus concrete model, labels, input image, preprocessed tensor, YoloVision preflight report/schema/hash/boundary, TensorRtExec report, engine, YoloVision run log, output JSON, stdout/stderr summary, and owner review fields. Template rows are not runtime proof; they remain `owner-action-required` until real hashes, logs, and review data are backfilled.

To keep the article case pack, owner backfill pack, and sample-run evidence rows aligned, run `eng/Export-YoloVisionRealAssetOwnerBackfillPack.ps1`. It writes `artifacts/user-acceptance/yolovision-real-asset-owner-backfill-sample-run-evidence.template.json` plus a projection report under `artifacts/yolovision`. The six generated evidence rows preserve the det/seg/pose/OBB/cls/sem commands, YoloVision preflight report/schema/hash/boundary, TensorRtExec report and engine hash slots, `YoloVision Passed=True` expected line, output JSON hash, stdout/stderr summary, and owner review fields. The template is still not proof; each row stays `template-only` and must pass `eng/Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog` after owner backfill before it can become real-model-runtime evidence.

When an owner is ready to backfill real evidence, run `eng/Export-YoloVisionRealAssetOwnerProofInputTemplate.ps1` and fill `artifacts/user-acceptance/yolovision-real-asset-owner-proof-input.template.json`. Then validate and project it with `eng/Test-YoloVisionRealAssetOwnerProofInput.ps1 -Strict` and `eng/Import-YoloVisionRealAssetOwnerProofInput.ps1`. The importer writes `artifacts/user-acceptance/yolovision-real-asset-owner-sample-run-evidence.candidate.json`; it is a real-model-runtime candidate only after all hashes, host metadata, owner review, run logs and `YoloVision Passed=True` lines are real. It never becomes `package-consumer-runtime` proof.

## Required Assets For A Full Demo

- `model.onnx`: a YOLO-family model with documented input tensor name, output tensor names, layout, and supported image sizes.
- `labels.txt`: class label list matching the model.
- `input.*`: one or more redistributable test images.
- Postprocessing metadata: confidence threshold, IoU threshold, output decoding format, and NMS/plugin requirements.

## Asset Metadata Checklist

- Model source URL, license, SHA256, opset, and export command.
- Input tensor name, shape, layout, dtype, resize, padding, and normalization.
- Output tensor name, shape, layout, class count, and objectness rule.
- Label file source, license, and line count.
- Test image source, license, and expected detection notes.
- NMS location: graph, plugin, or application-side postprocessing.
- NMS mode: class-aware or class-agnostic.
- For segmentation: mask coefficient count, prototype count, prototype width/height, and mask scale/crop rule.
- For pose: keypoint count and keypoint stride.
- For OBB: angle channel index and unit, degrees or radians.
- YOLO family/task declaration and any model-specific decode notes.

## Managed Multi-Output Metadata

The command-line runner still executes a single-output TensorRT sample path. For models whose exported graph returns separate auxiliary tensors, use the managed helpers from tests or a host application:

- `YoloSampleRunner.DecodeSegmentationOutputs(...)`: detection rows plus mask coefficients and `[P,H,W]` or `[1,P,H,W]` prototype tensor.
- `YoloSampleRunner.DecodePoseOutputs(...)`: detection rows plus `[1,N,K*stride]` or `[1,K*stride,N]` keypoint tensor.
- `YoloSampleRunner.DecodeObbOutputs(...)`: detection rows plus `[1,N,1]` or `[1,1,N]` angle tensor.
- `YoloMultiOutputMetadata`: declares mask coefficient count, keypoint count/stride, angle unit, optional auxiliary channel start, and auxiliary tensor layout.

These helpers keep model-specific ownership outside TensorRT and are covered by managed tests. A real asset manifest must still record the exact output tensor names, shapes, layout, crop/scale rules, and evidence log before the sample is treated as a real demo pass.

The shared ONNX sample support now also has a multi-output snapshot path. `YoloVision` captures all float outputs, wraps them as `YoloRuntimeOutputTensor` values with explicit roles, then routes them through `YoloSampleRunner.DecodeRuntimeOutputs`. The command-line path remains compatible with single-output models by assigning the primary output role from the selected task and falling back to the single-output diagnostic decoder when no auxiliary metadata is supplied.

Use `--output <path>` or `--output-json <path>` to write a machine-readable `yolovision-output.v1` JSON report. The report includes copied output tensor shapes, per-output `valueSha256` and preview values, task/family metadata, postprocess thresholds, prediction summaries, labels path/class count/SHA256, model/input SHA256 values when files are available, and a strict boundary block that keeps the file out of runtime-proof promotion. When `--image` is used, the report also records the source image path/SHA256/size, the generated preprocessed tensor path/SHA256/element count, layout, color order, normalization scale, and letterbox/stretch metadata. It is intended for owner review, golden-output comparison, and sample-run evidence backfill; it still needs real logs, hashes, host metadata, and owner approval before any real-model-runtime decision.

Use `--visualization <path>` or `--visualization-svg <path>` to write a lightweight SVG visualization beside the JSON report. The SVG covers detection boxes, classification bars, segmentation masks/boxes, OBB rotation, pose keypoints, and semantic maps. This file is useful for article screenshots and owner review, but it is still derived evidence: it must be tied to a real model, real input image, real preprocessed tensor, output JSON hash, run log hash, and owner review before a real-model-runtime claim is allowed.

Minimal output examples for all six tasks live under `samples/YoloVision/examples`: det, cls, seg, OBB, pose, and semantic segmentation. Validate those examples, or owner-produced output JSON files, with:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionOutputReport.ps1 -Strict
```

The validator checks task-specific prediction metadata, copied output tensor summaries, `boundary.isRuntimeProof=false`, and the forbidden substitute list. It writes `artifacts/yolovision/yolovision-output-report-validation.json`; this validation artifact is still owner-review infrastructure, not runtime proof.

For multi-output models, declare output roles and auxiliary metadata explicitly:

```powershell
dotnet run --project .\samples\YoloVision -- `
  --model .\models\yolo-seg.onnx `
  --family v8 `
  --task seg `
  --output-role-map boxes:det,proto:mask-prototypes `
  --mask-coefficient-count 32 `
  --aux-layout boxes-first
```

Task-specific command skeletons should stay explicit in article drafts and owner proof records. Use the same runner and change only the family/task/output metadata that the selected model actually needs:

```powershell
# Detection: YOLO v5/v6/v7/v8/v9/v10/v11/v26/YOLOX/custom
dotnet run --project .\samples\YoloVision -- --model .\models\yolo-det.onnx --labels .\models\coco.names --image .\models\det.ppm --preprocessed-output .\models\det-fp32.bin --input-shape 1x3x640x640 --family v8 --task det --layout auto --has-objectness auto --nms-mode class-aware

# YOLOv10 end-to-end detection: [1,N,6] = x1,y1,x2,y2,score,classId; no second NMS
dotnet run --project .\samples\YoloVision -- --model .\models\yolov10n.onnx --labels .\models\coco.names --image .\models\det.ppm --preprocessed-output .\models\yolov10n-fp32.bin --input-shape 1x3x640x640 --family v10 --task det --layout end2end --class-count 80 --confidence 0.25 --output .\artifacts\yolovision\yolov10n-output.json

# Classification
dotnet run --project .\samples\YoloVision -- --model .\models\yolo-cls.onnx --labels .\models\labels.txt --input-data .\models\cls-fp32.bin --input-shape 1x3x224x224 --family custom --task cls --classification-output logits

# Segmentation with explicit source-image mask mapping
dotnet run --project .\samples\YoloVision -- --model .\models\yolo-seg.onnx --labels .\models\coco.names --image .\models\seg.ppm --preprocessed-output .\models\seg-fp32.bin --input-shape 1x3x640x640 --family v8 --task seg --output-role-map boxes:det,proto:mask-prototypes --mask-coefficient-count 32 --mask-threshold 0.5 --mask-spatial-transform --mask-coordinate-space model-input --mask-crop-to-box true

# Oriented bounding box
dotnet run --project .\samples\YoloVision -- --model .\models\yolo-obb.onnx --labels .\models\labels.txt --input-data .\models\obb-fp32.bin --input-shape 1x3x1024x1024 --family v8 --task obb --output-role-map boxes:det,angles:obb-angle --obb-angle-output angles

# Pose
dotnet run --project .\samples\YoloVision -- --model .\models\yolo-pose.onnx --labels .\models\labels.txt --input-data .\models\pose-fp32.bin --input-shape 1x3x640x640 --family v8 --task pose --output-role-map boxes:det,keypoints:pose-keypoints --pose-keypoint-count 17

# Semantic segmentation
dotnet run --project .\samples\YoloVision -- --model .\models\yolo-sem.onnx --labels .\models\labels.txt --input-data .\models\sem-fp32.bin --input-shape 1x3x512x512 --family custom --task sem --semantic-output semantic --class-count 21
```

These skeletons are documentation and proof-record scaffolding, not bundled proof. A real `real-model-runtime` record still needs the exact model SHA256, labels SHA256, image SHA256, preprocessed tensor SHA256, run log SHA256, stdout/stderr summaries, expected evidence lines, license notes, and owner approval.

Dedicated role options are also accepted: `--detection-output`, `--classification-output`, `--semantic-output`, `--mask-prototypes-output`, `--pose-keypoints-output`, and `--obb-angle-output`. If no explicit role is supplied, the runner uses conservative tensor-name heuristics such as `proto`, `keypoint`, `angle`, `semantic`, `logits`, `box`, or `detect`.

When the sample runs with the default synthetic tensor, it is pipeline evidence only. It does not prove real object detection quality.

## Real Case Proof Pack

The release-facing real case checklist is generated into `artifacts/final-release/real-case-proof-execution-pack.json` and `artifacts/final-release/real-case-proof-execution-pack.md`. The YoloVision portion covers detection, segmentation, OBB, pose, and classification cases, but the pack is still an owner execution plan rather than runtime proof.

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealCaseProofExecutionPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealCaseEvidenceRecordTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealCaseEvidenceRecord.ps1 -RecordPath .\artifacts\final-release\real-case-evidence-record-template.json
```

`Test-RealCaseEvidenceRecord.ps1` keeps the template at `blocked-owner-action-required` until a real owner supplies the model source, license, ONNX/engine/input/output SHA256 values, stdout/stderr logs, screenshot, host OS, GPU, driver, CUDA, TensorRT, runtime package metadata, and owner review. The sample README, article matrix, sidecar, screenshots, and build-only reports must keep `canPublishPublicly=false` and cannot substitute `real-model-runtime`, `package-consumer-runtime`, Linux runner, owner authorization, or post-publish verification proof.

See `docs/articles/zh-cn/yolovision-model-assets.md`, `docs/articles/zh-cn/yolovision-asset-candidates.md`, `docs/articles/zh-cn/yolovision-real-asset-walkthrough.md`, `docs/articles/zh-cn/yolovision-yolox-official-runtime-tutorial.md`, `samples/assets/yolovision-assets.template.json`, and `samples/assets/yolovision-yolox-s-example.json` for the release-facing asset checklist and the completed official YOLOX-S source-tree runtime path.

## Real Model Evidence Backfill

Generate sidecar templates before turning a candidate YOLO asset record into real model evidence:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OnnxEngineBuildEvidenceSidecarTemplate.ps1
```

For a generic YOLO-family model, start from `artifacts/user-acceptance/onnx-engine-build-evidence-sidecar.yolovision.template.json`. For the YOLOX-S candidate walkthrough, start from `artifacts/user-acceptance/onnx-engine-build-evidence-sidecar.yolox-s.template.json`. Copy the selected template into the `models/*-evidence.sidecar.json` path referenced by the asset manifest, then run the TensorRtExec build-only command with `--evidenceSidecar`.

Next, run `samples/YoloVision` with a real model, labels, input image, task/family profile, output tensor metadata, and postprocess settings. Record `YoloVision Passed=True`, stdout/stderr summaries, the run log SHA256, model SHA256, labels SHA256, image SHA256, and license notes in the manifest.

Validate the backfill:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-SampleRunEvidenceRecordTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-YoloVisionRealAssetOwnerBackfillPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-YoloVisionRealAssetOwnerProofInputTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionRealAssetOwnerProofInput.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-YoloVisionRealAssetOwnerProofInput.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OnnxEngineBuildEvidenceSidecar.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleAssetManifest.ps1
```

Use `artifacts/user-acceptance/sample-run-evidence-record.yolovision.template.json` for generic YOLO-family assets, `sample-run-evidence-record.yolox-s.template.json` for the YOLOX-S walkthrough, or `yolovision-real-asset-owner-backfill-sample-run-evidence.template.json` for the six YOLOv8n task/article cases. This record carries the real runner log path, log SHA256, stdout/stderr summaries, expected evidence lines, and `canPromoteRealModelRuntime` decision. The sidecar is a bridge between TensorRtExec/OnnxToEngine reports and the asset manifest; the sample run evidence record is the bridge between the real `YoloVision` runner log and the manifest. Neither one can promote a build-only report or sample manifest to `package-consumer-runtime`.

## Evidence Lines

- `YoloVision TensorRtLine=...`
- `Profile Family=... Task=... Layout=... Nms=... NmsMode=...`
- `InputSource=external InputFile=...` for real preprocessed tensors, or `InputSource=synthetic InputFile=ramp`
- `Input=... Output=...`
- `Postprocess Task=...;Detections=...;Classifications=...;Semantic=...`
- `OutputJson=...` when `--output` or `--output-json` is supplied
- `Visualization=...` when `--visualization` or `--visualization-svg` is supplied
- `Output=... Outputs=...` showing the primary output and captured output count
- `Detection Class=... Score=... BoxCxCyWh=...` or `Detections=0 ...`
- `Classification Class=... Score=...` for classification models
- `SemanticMap Classes=... Width=... Height=... Values=...` for semantic segmentation models
- `Segmentations=...`, `OrientedBoxes=...`, or `Poses=...` when a host application uses the managed multi-output helpers
- `YoloVision Passed=True`

## Managed Postprocess Tests

Project quality tests cover layout inference, objectness/class score handling, class-aware NMS, family/task parsing, and the asset boundary. These tests do not require a TensorRT runtime or bundled model assets.
