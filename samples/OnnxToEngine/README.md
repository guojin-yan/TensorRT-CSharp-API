# OnnxToEngine

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
dotnet .\samples\OnnxToEngine\bin\Debug\net8.0\OnnxToEngine.dll --tensor-rt-line 10
```

External ONNX build-only example:

```powershell
dotnet run --project .\samples\OnnxToEngine -- `
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
dotnet run --project .\samples\OnnxToEngine -- `
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

The explicit `--mnist` path is a model-specific runner. It reads TensorRT's P5 PGM assets, applies the official sample preprocessing formula `1 - pixel / 255`, builds and deserializes the external MNIST ONNX model, binds the discovered input/output tensor names, enqueues inference, applies stable softmax, and verifies both the expected digit and a minimum confidence.

```powershell
dotnet .\samples\OnnxToEngine\bin\Debug\net8.0\OnnxToEngine.dll `
  --mnist `
  --tensor-rt-line 10 `
  --onnx ".\third_party\nvidia\TensorRT-10.11.0.33-cuda 12.9\data\mnist\mnist.onnx" `
  --mnistInput ".\third_party\nvidia\TensorRT-10.11.0.33-cuda 12.9\data\mnist\7.pgm" `
  --expectedDigit 7 `
  --minimumConfidence 0.9 `
  --saveEngine ".\artifacts\real-case\onnx-to-engine-mnist-trt10-runtime\mnist-trt10.plan" `
  --exportReport ".\artifacts\real-case\onnx-to-engine-mnist-trt10-runtime\mnist-trt10-runtime-report.json" `
  --exportOutput ".\artifacts\real-case\onnx-to-engine-mnist-trt10-runtime\mnist-trt10-output.json" `
  --exportPreprocessedInput ".\artifacts\real-case\onnx-to-engine-mnist-trt10-runtime\mnist-trt10-input-f32.bin"
```

Only a completed external-model enqueue with matching digit and confidence can be classified as `real-model-runtime`. This remains a source-tree sample execution, not `package-consumer-runtime`, post-publish proof, or release authorization. Generic external ONNX execution remains build-only unless another explicit model runner defines its input and output semantics.

YoloVision model semantics are deliberately not inferred by this sample. Detection, classification, segmentation, OBB, pose, and semantic-segmentation output roles are governed by `samples/YoloVision/yolovision-task-output-contract.json`; OnnxToEngine can provide build/report artifacts for those models, but the contract plus YoloVision owner evidence is what defines output metadata and real-model-runtime promotion.

Reports can be exported as JSON or Markdown with `--exportReport`; `--report` is accepted as a compatibility alias and is normalized back to `--exportReport` in report output. Input aliases `--model` and `--onnxFile` normalize to `--onnx`. Engine file aliases `--plan` and `--engineFile` normalize to `--saveEngine` for ONNX/build-only commands and to `--loadEngine` only when no ONNX/build-only intent is present. The report records parser/build state, TensorRT line, engine path, precision flags, `ProofClassification`, `BuildEvidenceOnly`, `DryRun`, `NormalizedCommandLine`, `NormalizedCommandSha256`, `PreflightMetadata`, `BuilderConfigDeploymentSnapshot`, `ReportBoundary`, and whether inference actually ran. `BuilderConfigDeploymentSnapshot` is the copied readback after deployment options and optimization profile application; it makes requested values auditable against actual builder-config values, with version-specific unsupported fields kept in diagnostics. `ReportBoundary` explicitly classifies ONNX Parser / ParserRefitter copied diagnostics as `copied-parser-diagnostics` and `copied-parser-refitter-diagnostics`; those snapshots are troubleshooting and release-gate surface evidence only, not runtime proof. Build-only, dependency-probe-only, and precheck reports are not runtime execution proof. The console also prints `OnnxToEngine ReportPath=...`, `OnnxToEngine ProofClassification=...`, `OnnxToEngine NormalizedCommandSha256=...`, `OnnxToEngine BuilderConfigDeploymentSnapshot=...`, and `OnnxToEngine State=...` so scripts can collect the generated build/report evidence without treating it as release proof.

Runtime/output artifacts from `--exportTimes`, `--exportOutput`, `--exportProfile`, `--saveProfile`, and `--dumpRawBindingsToFile` include structured proof-boundary fields: `ArtifactProofBoundary`, `RuntimeProofClass`, `HasTensorOutputProof`, `HasRawBindingProof`, `IsBuildOnlyEvidence`, `IsDependencyProbeOnly`, `IsSyntheticRuntime`, `ModelSource`, `EnginePath`, and `PreflightMetadata`. When any runtime artifact path is present, the writer also creates a neighboring `*.engine-readback.json` artifact with `LoadedEngineDiagnostics`, `ReadbackFingerprint`, `ReadbackSha256`, `ReadbackAvailable`, and skipped reason. These fields are designed for auditability. `build-only` artifacts must not claim tensor or raw-binding proof; `dependency-probe-only` artifacts record preflight/readback metadata only; `runtime-output-captured-unverified` means enqueue/readback completed but no reference output matched; `synthetic-input-runtime is not real-model-runtime` and is never package-consumer-runtime proof.

Raw binding dumps remain conservative. The sample writes binary raw bytes only for the embedded identity synthetic runtime after inference ran, output matched, and raw bytes exist. For external ONNX build-only, dry-run, load-engine preflight, bounded runtime output without reference match, or missing binding semantics, the raw binding path receives skipped JSON with `HasRawBindingProof=false`.

When `--loadEngine` is used, the sample first runs safe readonly diagnostics. If the engine exposes exactly one float input and float outputs with a concrete runtime shape, it then attempts a bounded runtime path: set the input shape, fill a generated ramp tensor or read `--loadInputs tensor:file`, create one independent execution context/binding set/CUDA stream per effective `--infStreams` (or `--streams`) value, warm up for at least `--warmUp` milliseconds, then measure until both `--iterations` and `--duration` minima are satisfied. `--idleTime` is applied between measurement rounds, `--avgRuns` emits consecutive timing-average windows, and `--percentile` is calculated from raw GPU timing samples. If the single output matches the input exactly, the result remains `synthetic-input-runtime`; otherwise it is recorded as `runtime-output-captured-unverified` with `InferenceRan=true` and `OutputMatch=false`. This path proves bounded enqueue/readback and scheduler execution only. It does not prove model correctness, real-model-runtime, package-consumer-runtime, or post-publish verification.

`--builderOptimizationLevel` and `--maxAuxStreams` are applied to the TensorRT builder config when the runtime reaches the build stage. `--memPoolSize` known pool tokens (`workspace`, `dlaSRAM`, `dlaLocalDRAM`, `dlaGlobalDRAM`, `tacticDRAM`, `tacticSharedMem`) are also applied through `SetMemoryPoolLimit` and verified with `GetMemoryPoolLimit`; the build log records requested/readback bytes. Deployment-shaped arguments such as `--device`, `--useDLACore`, `--allowGPUFallback`, `--tacticSources`, `--inputIOFormats`, `--outputIOFormats`, `--calib`, `--directIO`, `--sparsity`, and `--stronglyTyped` remain parsed/report-only in this sample stage. `--shapes` and `--inputShapes` are accepted as aliases that seed min/opt/max shape profiles when no explicit profile triplet is supplied.

`--noDataTransfers`, `--useSpinWait`, `--threads`, `--sleepTime`, and `--useCudaGraph` remain parse/report-only because the bounded scheduler does not implement those official runtime mechanics. `--safe`, `--consistency`, `--builderCache`, and `--noBuilderCache` are also parse/report-only trtexec alignment switches; `--builderCache` and `--noBuilderCache` are intentionally mutually exclusive. Applied scheduler controls never claim DLA execution, calibrator callbacks, plugin loading, model-specific inference proof, safety runtime execution, builder cache lifecycle, or package-consumer-runtime proof.

## Real Case Proof Pack

`OnnxToEngine` is represented in `artifacts/final-release/real-case-proof-execution-pack.json` as the `onnx-to-engine-build` case. That case binds the external ONNX path, shape profile, serialized engine, build report, sidecar, hash fields, and host metadata into an owner-fillable proof checklist, while keeping the classification as `blocked-owner-action-required`.

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealCaseProofExecutionPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealCaseEvidenceRecordTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealCaseEvidenceRecord.ps1 -RecordPath .\artifacts\final-release\real-case-evidence-record-template.json
```

The validator requires real owner inputs before any promotion: model source, license, ONNX SHA256, engine SHA256, build report or sidecar SHA256, command line, logs, screenshot when applicable, host OS, GPU, driver, CUDA, TensorRT, runtime package metadata, and owner review. `real-case-proof-execution-pack` and build-only reports must keep `canPublishPublicly=false`; they cannot replace YoloVision or Classification real sample runtime logs, package-consumer-runtime proof, Linux runner proof, owner authorization, or post-publish verification.

