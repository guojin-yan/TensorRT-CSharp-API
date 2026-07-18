# OnnxToEngine trtexec Parity Matrix

`samples/OnnxToEngine/trtexec-parity-matrix.json` records how the sample aligns with key official `trtexec` model-conversion concepts. The goal is practical ONNX-to-engine coverage and reportability, not claiming a complete `trtexec` clone.

## Implemented Or Reported Areas

- Model paths: `--onnx`, `--saveEngine`, `--loadEngine`; `--loadEngine` now combines readonly readback with bounded one-float-input runtime output when shapes are concrete.
- Dynamic shapes: `--minShapes`, `--optShapes`, `--maxShapes`, `--shapes`, `--inputShapes`.
- Precision: `--fp16` implemented; `--fp8`, `--best`, `--int8`, and calibration remain parse/report-only or ownership-blocked until capability and model proof exist.
- Memory and builder controls: `--workspace` and known `--memPoolSize` pools are applied through `TensorRtBuilderConfig.SetMemoryPoolLimit` and immediately read back through `GetMemoryPoolLimit`; `--tacticSources` remains diagnostic. Timing cache uses the typed owner and records cache byte counts/SHA256 in the build report.
- Plugin diagnostics: `--plugins`, `--plugin`, `--dynamicPlugins`, and `--setPluginsToSerialize` are accepted as aliases/repeated lists and normalized into `Plugins`; they remain diagnostics and do not load, register, deregister, or serialize plugin libraries.
- Diagnostics and preflight: `--profilingVerbosity`, `--verbose`, `--previewOnly`, `--dryRun`.
- Runtime-shaped artifact switches: `--loadInputs`, `--dumpOutput`, `--dumpRawBindingsToFile`, `--exportOutput`, `--exportTimes`, `--exportProfile`, and `--saveProfile` are recorded as bounded artifacts. `--loadInputs` can feed bounded generic runtime, but it is not tensor correctness, raw binding, real-model-runtime, or package-consumer-runtime proof without expected output, real runner logs, and hashes.
- Engine packaging and device intent: `--refit`, `--dumpRefit`, `--allowWeightStreaming`, `--markDebug`, `--dumpDebugTensors`, `--versionCompatible`, `--useDLACore`, `--allowGPUFallback`.

Build-only, parse-only, preflight-only, sidecar-only, dependency-probe-only, runtime-output-captured-unverified, and synthetic runtime reports are not real-model-runtime proof and not package-consumer-runtime proof.
