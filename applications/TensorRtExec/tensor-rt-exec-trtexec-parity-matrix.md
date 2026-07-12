# TensorRtExec Trtexec Parity Matrix Artifact

This file mirrors `tensor-rt-exec-trtexec-parity-matrix.json` for quick review. It is a machine-readable release-readiness artifact, not runtime proof.

| ID | trtexec option | Status | Proof boundary |
|---|---|---|---|
| `onnx-input` | `--onnx` / `--model` / `--onnxFile` | implemented | ONNX parsing/build report is build-only unless a sample runner records real model execution. |
| `save-engine` | `--saveEngine` / `--save-engine` / `--engine` / `--plan` / `--engineFile` | implemented | Serialized engine creation is build evidence, not inference proof. |
| `load-engine` | `--loadEngine` | bounded-runtime-output | Current load-engine path records preflight/readback metadata and may execute bounded one-float-input enqueue/readback. Unmatched output is runtime-output-captured-unverified and is not real-model or package-consumer proof. |
| `dynamic-shape` | `--minShapes/--optShapes/--maxShapes` | implemented-report | Shape profiles in reports do not prove every dynamic shape ran correctly. |
| `shape-profile` | `--minShapes input:... --optShapes input:... --maxShapes input:...` | implemented-report | Profile configuration is not output correctness proof. |
| `shape-alias-batch` | `--shapes --inputShapes --batch` | implemented-report | Shape aliases and batch values are normalized/reportable migration aids; they do not prove binding coverage, dynamic profile execution, or output correctness. |
| `yolovision-owner-backfill-shape-profiles` | `--minShapes/--optShapes/--maxShapes per YOLOv8n task` | documented-report | Task-specific shape profiles are build/report evidence only; real-model-runtime requires YoloVision run log, output JSON, hashes, and owner review. |
| `fp16` | `--fp16` | wrapper-ready | The switch can be represented; GPU/model support must be proven by smoke. |
| `int8` | `--int8 --calib` | parse-report-only-calibration-boundary | INT8 and calibration cache paths are CLI/GUI/report fields only; calibrator ownership and cache validity are not promoted by this application matrix. |
| `precision-shortcuts-debug-boundary` | `--fp8 --best --dumpRefit --allowWeightStreaming --markDebug --dumpDebugTensors` | parse-report-only | FP8/best precision shortcuts, refit dumps, weight-streaming intent, and debug tensor names are CLI/GUI/report fields only; they do not prove precision support, refit lifecycle, weight streaming execution, or debug tensor runtime output. |
| `workspace-memory-pool` | `--workspace --memPoolSize` | implemented-report | Recorded memory settings are build/report evidence only. |
| `timing-cache` | `--timingCacheFile --exportTimingCache` | parse-report-only | Timing cache path capture, normalized command output, and ParseOnlyOptions entries are not cache import/export lifecycle proof. |
| `gui-cli-field-map` | `CLI/WinForms parity checklist` | checklist-backed | GUI/CLI field parity proves option surface alignment only; command preview, GUI screenshots, dry-runs, and reports are not runtime proof. |
| `plugin-library-boundary` | `--plugins/--plugin/--dynamicPlugins/--setPluginsToSerialize` | diagnostic-alias-compatible | Plugin library paths, trtexec-compatible aliases, and copied inventory metadata do not prove plugin library load/register/deregister. |
| `profiling` | `--profilingVerbosity --dumpProfile --separateProfileRun --exportProfile --saveProfile` | implemented-report | Profiling artifacts are diagnostics unless a real enqueue run and log prove runtime execution. |
| `wait-idle-controls` | `--sleepTime --idleTime` | parse-report-only | Wait/idle controls are accepted and reported for command parity; they do not prove official benchmark scheduler semantics or runtime performance. |
| `layer-dump` | `--dumpLayerInfo --exportLayerInfo` | implemented-report | Layer info is diagnostic metadata, not output correctness proof; CLI and WinForms now route the same diagnostic switches. |
| `report-export-alias` | `--exportReport / --report` | implemented-report | Report aliases only choose JSON/Markdown output paths; reports remain build/report evidence and cannot promote runtime proof. |
| `verbose-logging` | `--verbose` | implemented-report | Verbose logs are supporting evidence only; logs need SHA256 and owner review. |
| `binding-metadata` | `--loadInputs --dumpOutput --dumpRawBindingsToFile --exportOutput` | synthetic-runtime-ready-external-guarded | Synthetic runtime may prove a tiny pipeline; bounded external/load-engine output can prove enqueue/readback only; external ONNX binding metadata still requires real model evidence. |
| `package-consumer-runtime-proof-boundary` | release proof validator | release-proof-records-only | Only clean external package consumer smoke validated as package-consumer-runtime can promote release proof. |

Every entry keeps `isRuntimeProof=false` in the JSON. The package proof boundary row points to release validators; it does not allow `TensorRtExec` itself to declare release proof.
