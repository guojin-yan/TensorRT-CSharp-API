# TensorRtExec GUI/CLI Field Map

`tensor-rt-exec-gui-cli-field-map.json` records the user-facing parity between the console entry and the WinForms entry. The map is a surface-contract artifact only: it is not runtime proof, not package-consumer-runtime proof, not post-publish proof, and not release approval.

## Boundary

- Command preview is not proof.
- GUI screenshots are not proof.
- Build-only reports are not runtime proof.
- Parse-report-only and precheck-only options must not be promoted to official trtexec parity.
- Real release proof still requires Owner-provided public package metadata, CleanConsumer logs, PostPublish logs, rollback review, and final release close approval.

## Field Groups

| Group | Examples | Boundary |
| --- | --- | --- |
| Model | `--onnx`, `--saveEngine`, `--loadEngine` | load-engine remains readonly diagnostics |
| Shape Profile | `--minShapes`, `--optShapes`, `--maxShapes` | shared normalized command |
| Plugin Paths | `--plugins` | diagnostic GUI/CLI parity only; load/register remains deferred |
| Precision | `--fp16`, `--int8`, `--fp8`, `--best` | INT8 and advanced precision remain guarded |
| Builder | `--workspace`, `--memPoolSize`, `--builderOptimizationLevel` | applied or diagnostic according to report |
| Timing/Profile | `--timingCacheFile`, `--exportTimingCache`, `--dumpProfile` | diagnostics/report evidence |
| Output Artifacts | `--loadInputs`, `--dumpOutput`, `--dumpRawBindingsToFile`, `--exportOutput`, `--exportTimes`, `--exportProfile`, `--saveProfile` | bounded artifacts only |
| Mode | `--buildOnly`, `--skipInference`, `--dryRun` | not runtime proof |
| Report | `--exportReport`, `--report`, `--evidenceSidecar` | report/sidecar only |

The WinForms `Command Preview` text is produced by `TensorRtExecOptions.ToArgumentLine()`, the same normalized command path used by the console. This keeps GUI and CLI semantics aligned while preserving proof boundaries.
