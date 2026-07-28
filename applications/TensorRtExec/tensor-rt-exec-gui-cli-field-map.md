# TensorRtExec GUI/CLI Field Map

`tensor-rt-exec-gui-cli-field-map.json` records the user-facing parity between the console entry and the WinForms entry. It covers all 84 normalized options emitted by `TensorRtExecOptions` plus the command-preview row. The map is a surface-contract artifact only: it is not runtime proof, not package-consumer-runtime proof, not post-publish proof, and not release approval.

## Boundary

- Command preview is not proof.
- GUI screenshots are not proof.
- Build-only reports are not runtime proof.
- Parse-report-only and precheck-only options must not be promoted to official trtexec parity.
- Real release proof still requires Owner-provided public package metadata, CleanConsumer logs, PostPublish logs, rollback review, and final release close approval.

## Field Groups

| Group | Examples | Boundary |
| --- | --- | --- |
| Model | `--onnx`, `--saveEngine`, `--loadEngine`, `--refitFromOnnx`, `--saveRefittedEngine` | ONNX refit persistence keeps stripped/refitted artifacts distinct, reloads a new engine owner, and has separate local-feed PackageReference output-hash evidence; public package/post-publish proof stays external |
| Shape Profile | `--minShapes`, `--optShapes`, `--maxShapes` | shared normalized command |
| Plugin Paths | `--plugins` | diagnostic GUI/CLI parity only; load/register remains deferred |
| Precision | `--fp16`, `--bf16`, `--noTF32`, `--int8`, `--calib`, `--fp8`, `--best` | INT8/calibration and advanced precision remain guarded |
| Builder | `--workspace`, `--memPoolSize`, `--builderOptimizationLevel`, `--maxNbTactics`, `--tilingOptimizationLevel` | scalar settings and known memory pools are version-guarded build/readback evidence |
| Deployment | `--device`, `--useDLACore`, `--allowGPUFallback`, `--tacticSources`, `--inputIOFormats`, `--layerPrecisions`, `--stronglyTyped` | version-guarded device/build readback; not DLA/model/package runtime proof |
| Packaging/Debug | `--versionCompatible`, `--stripWeights`, `--refit`, `--allowWeightStreaming`, `--dumpRefit`, `--markDebug` | applied options retain version guards; debug/refit dump intent stays parse-only |
| Timing/Profile | `--timingCacheFile`, `--exportTimingCache`, `--minTiming`, `--avgTiming`, `--profilingVerbosity` | build-cache, versioned builder readback, or diagnostics evidence only |
| Bounded Runtime | `--iterations`, `--streams`, `--infStreams`, `--threads`, `--useSpinWait`, `--useCudaGraph`, `--idleTime` | bounded scheduler behavior; not external model correctness or package proof |
| Device-side Delay | `--sleepTime` | parse/report-only until a faithful launch-to-compute GPU mechanism exists; CPU sleep is not a substitute |
| Safety/Cache Policy | `--safe`, `--consistency`, `--builderCache`, `--noBuilderCache` | parse/report-only intent |
| Output Artifacts | `--loadInputs`, `--dumpOutput`, `--dumpRawBindingsToFile`, `--exportOutput`, `--exportTimes`, `--exportProfile`, `--saveProfile` | bounded multi-output preview/JSON/raw+manifest capture; capture is not validation |
| Mode | `--buildOnly`, `--skipInference`, `--dryRun` | not runtime proof |
| Report | `--exportReport`, `--report`, `--evidenceSidecar` | report/sidecar only |

The WinForms `Command Preview` text is produced by `TensorRtExecOptions.ToArgumentLine()`, the same normalized command path used by the console. This keeps GUI and CLI semantics aligned while preserving proof boundaries.
