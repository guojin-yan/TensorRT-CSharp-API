# OnnxToEngine

English | [简体中文](README.zh-CN.md)

Status: runnable common sample with trtexec-like option parsing.

The parity target is tracked in `docs/articles/zh-cn/onnx-to-engine-trtexec-parity-roadmap.md`. The sample can parse and report many trtexec-like switches, but only a subset is currently applied to real TensorRT builder/runtime behavior. Parse-only, probe-only, build-only, dependency-probe-only, and precheck outputs must not be described as official trtexec parity or runtime proof.

This sample generates a minimal dynamic ONNX identity model in process, so it does not require external model files. It demonstrates:

- ONNX parser creation
- in-memory ONNX parsing
- optimization profile setup
- serialized engine build
- runtime deserialize
- tensor binding
- enqueue and output readback
- trtexec-like option model for ONNX path, engine path, precision, workspace, shape profiles, timing cache, plugins, builder optimization level, max auxiliary streams, DLA diagnostics, IO format diagnostics, calibration-cache diagnostics, layer-info requests, benchmark/output diagnostic switches, and build-only/skip-inference modes

Run from the repository root with development probing enabled:

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"
```

Run:

```powershell
dotnet .\applications\OnnxToEngine\bin\Release\net8.0\OnnxToEngine.dll --tensor-rt-line 10
```

External ONNX build-only example:

```powershell
dotnet run --project .\applications\OnnxToEngine -- `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model.plan `
  --minShapes input:1x3x640x640 `
  --optShapes input:1x3x640x640 `
  --maxShapes input:4x3x640x640 `
  --fp16 `
  --workspace 512 `
  --builderOptimizationLevel 4 `
  --maxAuxStreams 2 `
  --exportReport .\models\model-build-report.json `
  --buildOnly
```

Dry-run / preview-only example:

```powershell
dotnet run --project .\applications\OnnxToEngine -- `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model.plan `
  --minShapes input:1x3x640x640 `
  --optShapes input:1x3x640x640 `
  --maxShapes input:4x3x640x640 `
  --builderOptimizationLevel 4 `
  --maxAuxStreams 2 `
  --exportReport .\models\model-precheck-report.json `
  --previewOnly
```

Dry-run mode accepts missing ONNX or engine paths and writes a precheck report with `DryRun=true`, `ProofClassification=precheck`, `BuildEvidenceOnly=true`, and `NormalizedCommandSha256`. It does not probe CUDA/TensorRT, parse ONNX, build an engine, load plugins, or run inference.

Expected evidence includes:

- `Parsed=True`
- `ProfileIndex=0`
- `EngineFileRoundTrip=True`
- `BindingReport Ready=True`
- `OutputMatch=True`

For arbitrary external ONNX models, this stage treats the sample as build-only or skip-inference unless the user supplies explicit binding and output semantics. That keeps model-dependent execution evidence separate from parser/build evidence.

## MNIST real-model runtime

The explicit `--mnist` path is a model-specific runner. It reads a 28x28 P5 PGM input, applies the model preprocessing formula `1 - pixel / 255`, builds and deserializes the external MNIST ONNX model, binds the discovered input/output tensor names, enqueues inference, applies stable softmax, and verifies both the expected digit and a minimum confidence. `--visualization` writes an SVG containing the actual input pixels, predicted digit, confidence, and all ten class probabilities.

TensorRT's `data/mnist/README.md` attributes this opset 8 graph to ONNX Model Zoo. It is already an ONNX release artifact, so there
is no framework-to-ONNX conversion step. Copy it from the user-installed TensorRT sample-data directory to
`<workspace>\models\OnnxToEngine\MNIST\nvidia-tensorrt-10.11\mnist.onnx` and verify SHA256
`2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf`. The full acquisition boundary is documented in
`docs/articles/zh-cn/demo-model-acquisition-and-onnx-conversion.md`; the ONNX is not committed or published.

```powershell
$workspaceRoot = Split-Path -Parent $PWD
$model = Join-Path $workspaceRoot 'models\OnnxToEngine\MNIST\nvidia-tensorrt-10.11\mnist.onnx'
$assetRoot = Join-Path $workspaceRoot 'downloads\mnist-owner-generated'

powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\New-MnistOwnerGeneratedDigit.ps1

dotnet .\applications\OnnxToEngine\bin\Release\net8.0\OnnxToEngine.dll `
  --mnist `
  --tensor-rt-line 10 `
  --onnx $model `
  --mnistInput (Join-Path $assetRoot 'digit-7.pgm') `
  --expectedDigit 7 `
  --minimumConfidence 0.5 `
  --saveEngine (Join-Path $assetRoot 'digit-7.plan') `
  --exportReport (Join-Path $assetRoot 'digit-7-report.json') `
  --exportOutput (Join-Path $assetRoot 'digit-7-output.json') `
  --exportPreprocessedInput (Join-Path $assetRoot 'digit-7-input.fp32.bin') `
  --visualization (Join-Path $assetRoot 'digit-7-result.svg')
```

The current project-generated-input evidence is `samples/assets/onnxtoengine-mnist-owner-generated-runtime-evidence.json`. TensorRT 10.11 predicts digit 7 at confidence `0.99945575`; all 10 logits match the independent ONNX Runtime CPU reference within `1e-4`. A controlled run that changes only `--expectedDigit` to 6 exits 2 with `State=mnist-output-mismatch` and `OutputMatch=False`. The earlier TensorRT-supplied-input record remains at `samples/assets/onnxtoengine-mnist-real-model-runtime-evidence.json` for historical comparison.

The complete Chinese acquisition, Engine build, project-generated input, annotated inference result, real terminal screenshot, independent reference, and controlled-negative walkthrough is [OnnxToEngine 实战：自有数字图片、TensorRT 与 ONNX Runtime 双重验证](../../docs/articles/zh-cn/onnxtoengine-mnist-owner-generated-tutorial.md).

The model, engine, PGM, tensors, and raw logs remain outside Git. Only a completed external-model enqueue with matching digit and confidence can be classified as `real-model-runtime`. This remains a source-tree sample execution, not package-consumer, public-package, post-publish, redistribution, or release proof. Generic external ONNX execution remains build-only unless another explicit model runner defines its input and output semantics.

YoloVision model semantics are deliberately not inferred by this sample. Detection, classification, segmentation, OBB, pose, and semantic-segmentation output roles are governed by `applications/YoloVision/yolovision-task-output-contract.json`; OnnxToEngine can provide build/report artifacts for those models, but the contract plus YoloVision owner evidence is what defines output metadata and real-model-runtime promotion.

Reports can be exported as JSON or Markdown with `--exportReport`; `--report` is accepted as a compatibility alias and is normalized back to `--exportReport` in report output. Input aliases `--model` and `--onnxFile` normalize to `--onnx`. Engine file aliases `--plan` and `--engineFile` normalize to `--saveEngine` for ONNX/build-only commands and to `--loadEngine` only when no ONNX/build-only intent is present. The report records parser/build state, TensorRT line, engine path, precision flags, `ProofClassification`, `BuildEvidenceOnly`, `DryRun`, `NormalizedCommandLine`, `NormalizedCommandSha256`, `PreflightMetadata`, `BuilderConfigDeploymentSnapshot`, `ReportBoundary`, and whether inference actually ran. `BuilderConfigDeploymentSnapshot` is the copied readback after deployment options and optimization profile application; it makes requested values auditable against actual builder-config values, with version-specific unsupported fields kept in diagnostics. `ReportBoundary` explicitly classifies ONNX Parser / ParserRefitter copied diagnostics as `copied-parser-diagnostics` and `copied-parser-refitter-diagnostics`; those snapshots are troubleshooting and release-gate surface evidence only, not runtime proof. Build-only, dependency-probe-only, and precheck reports are not runtime execution proof. The console also prints `OnnxToEngine ReportPath=...`, `OnnxToEngine ProofClassification=...`, `OnnxToEngine NormalizedCommandSha256=...`, `OnnxToEngine BuilderConfigDeploymentSnapshot=...`, and `OnnxToEngine State=...` so scripts can collect the generated build/report evidence without treating it as release proof.

Use `--help-json` or `--capabilities-json` to print the shared machine-readable trtexec-like option capability surface as `trtexec-like-option-capabilities.v1` JSON without loading TensorRT, CUDA, an ONNX model, plugins, or an engine. The payload includes option groups, aliases, implementation classes, parse/report-only or blocked states, `releaseFrozen=true`, and `canPromoteRuntimeProof=false`; it is source-quality capability metadata only, not runtime proof or release proof.

Runtime/output artifacts from `--exportTimes`, `--exportOutput`, `--exportProfile`, `--saveProfile`, and `--dumpRawBindingsToFile` include structured proof-boundary fields: `ArtifactProofBoundary`, `RuntimeProofClass`, `HasTensorOutputProof`, `HasRawBindingProof`, `OutputCaptureAvailable`, `OutputValidated`, `IsBuildOnlyEvidence`, `IsDependencyProbeOnly`, `IsSyntheticRuntime`, `ModelSource`, `EnginePath`, and `PreflightMetadata`. Bounded runtime captures every float output tensor in engine order. `--dumpOutput` logs a bounded preview plus shape, byte count, and SHA256 for every output; `--exportOutput` writes the corresponding `OutputTensors` array. When any runtime artifact path is present, the writer also creates a neighboring `*.engine-readback.json` artifact with `LoadedEngineDiagnostics`, `ReadbackFingerprint`, `ReadbackSha256`, `ReadbackAvailable`, and skipped reason.

Raw binding dumps are deterministic capture artifacts. `--dumpOutput`, `--dumpRawBindingsToFile`, or `--exportOutput` requests generic bounded runtime even without `--loadInputs`; that path generates deterministic float input and logs `InputSource=deterministic-generated`. When output readback succeeds, the raw path contains all output float32 bytes concatenated in engine order, while `<raw-path>.manifest.json` records tensor names, shapes, element counts, byte offsets, byte lengths, individual hashes, and the combined hash. Capture does not imply validation: an external bounded run without a reference match writes bytes but keeps `OutputValidated=false`, `HasTensorOutputProof=false`, `HasRawBindingProof=false`, and `runtime-output-captured-unverified`. Build-only, dry-run, dependency-probe-only, and no-transfer paths still receive skipped JSON instead of fabricated bytes. `synthetic-input-runtime is not real-model-runtime` and is never package-consumer-runtime proof.

When `--loadEngine` is used, the sample first runs safe readonly diagnostics. If every engine input/output is float and every input has a concrete runtime shape, it then binds all inputs in engine order, using independently generated values or complete `--loadInputs left:left.bin,right:right.bin` mappings. Each worker owns an independent execution context, binding set, and CUDA stream. The scheduler warms up and measures as before. A one-input/one-output identity match retains the legacy synthetic result; other captured outputs remain unverified unless structured references are supplied. This path proves bounded enqueue/readback and scheduler execution only. It does not prove real-model, package-consumer, or post-publish behavior.

`--referenceOutputs sum:sum.reference.json,difference:difference.reference.json` enables all-output comparison. Each reference JSON uses `schemaVersion: 1` and contains `tensorName`, `shape`, `values`, and a traceable `sourceClassification`. `--referenceAbsTolerance` and `--referenceRelTolerance` control finite comparison; `--referenceNaNPolicy reject|equal` and `--referenceInfinityPolicy exact|reject` make special values explicit. Missing, duplicate, unknown, unreadable, name, shape, count, or numerical mismatches fail validation with per-tensor diagnostics. Only a complete pass sets `OutputValidated=true`; file hashes alone do not. Synthetic references remain `synthetic-input-runtime` rather than real-model or package-consumer proof.

`--builderOptimizationLevel` and `--maxAuxStreams` are applied to the TensorRT builder config when the runtime reaches the build stage. `--memPoolSize` known pool tokens (`workspace`, `dlaSRAM`, `dlaLocalDRAM`, `dlaGlobalDRAM`, `tacticDRAM`, `tacticSharedMem`) are applied through `SetMemoryPoolLimit` and verified with `GetMemoryPoolLimit`. `--device` runs the operation on a dedicated host thread and reads back the current CUDA device. Real builds set/read back DLA/fallback/tactic/direct-I/O/sparsity and engine packaging flags. `--versionCompatible` and `--excludeLeanRuntime` cover TRT8/10/11; `--stripWeights` uses `StripPlan` plus `RefitIdentical` by default on TRT10/11; `--refit` is read back from both config and a deserialized engine when runtime continues. TRT8 keeps strip/streaming and the version-compatible+refit conflict parse-only. I/O and layer policies retain their existing TRT8/10 setters and TRT11 guards. `--shapes` and `--inputShapes` seed min/opt/max profiles when no explicit triplet is supplied.

`--allowWeightStreaming` requires a strongly typed build. `--weightStreamingBudget` accepts official `-2` disabled, `-1` automatic, `0..100%`, and exact byte-size forms; TRT10/11 resolve and set the budget before any execution context is created, then record streamable weights, automatic/resolved/readback budget, and scratch bytes. Load-engine mode can set a budget on an engine already built for streaming. `--noDataTransfers`, `--useSpinWait`, boolean `--threads`, `--useCudaGraph`, and `--sleepTime` are applied by compatible bounded runtime. Sleep time uses one bridge-owned `cudaLaunchHostFunc` state and an event fan-out to all inference streams; it is not implemented with caller-thread sleep or a managed callback. `--safe`, `--consistency`, `--builderCache`, and `--noBuilderCache` remain parse/report-only. Applied controls never claim model accuracy, performance improvement, package-consumer runtime, or public release proof.

`--refitFromOnnx <path>` is a managed extension for the stripped-plan refit lifecycle. It requires an explicit `--onnx` build plus `--stripWeights --refit`; load-engine mode is rejected so the ONNX build source and refit source cannot be confused. `--saveRefittedEngine <path>` optionally persists the committed engine to a path that must differ from both the stripped plan and ONNX source. The service creates an explicit serialization config, clears and reads back `ExcludeWeights`, serializes the full-weight engine, disposes the committed owner, independently reloads the persisted bytes, checks copied engine metadata, and routes optional runtime through that reload. Default serialization of a stripped engine is not sufficient because it can continue excluding refittable weights. A TensorRT 10 full-weight reload need not remain refittable. This local lifecycle does not prove package-consumer runtime or public release readiness.

## Real Case Proof Pack

`OnnxToEngine` is represented in `artifacts/final-release/real-case-proof-execution-pack.json` as the `onnx-to-engine-build` case. That case binds the external ONNX path, shape profile, serialized engine, build report, sidecar, hash fields, and host metadata into an owner-fillable proof checklist, while keeping the classification as `blocked-owner-action-required`.

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealCaseProofExecutionPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealCaseEvidenceRecordTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealCaseEvidenceRecord.ps1 -RecordPath .\artifacts\final-release\real-case-evidence-record-template.json
```

The validator requires real owner inputs before any promotion: model source, license, ONNX SHA256, engine SHA256, build report or sidecar SHA256, command line, logs, screenshot when applicable, host OS, GPU, driver, CUDA, TensorRT, runtime package metadata, and owner review. `real-case-proof-execution-pack` and build-only reports must keep `canPublishPublicly=false`; they cannot replace YoloVision or Classification real sample runtime logs, package-consumer-runtime proof, Linux runner proof, owner authorization, or post-publish verification.

