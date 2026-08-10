# TensorRtExec Refitted Plan Persistence Validation

- Strict: `True`
- Checks: `52`
- Passed: `52`
- Failed: `0`

| Check | Passed | Actual |
| --- | --- | --- |
| `schema` | `True` | `trtexec-refitted-plan-persistence-evidence.v1` |
| `contract-requires-refit-source` | `True` | `--onnx,--stripWeights,--refit,--refitFromOnnx` |
| `trt10-persisted` | `True` | `True/True` |
| `trt10-artifact-differs` | `True` | `5594817d8b152a9478a57ae73ec2a8ed19e8c9e0794b448b89bcd738d99441eb` |
| `trt10-exclude-weights-present-before` | `True` | `3` |
| `trt10-exclude-weights-cleared-after` | `True` | `2` |
| `trt10-weights-included` | `True` | `True/True` |
| `trt10-original-disposed` | `True` | `True/True` |
| `trt10-same-process-reload` | `True` | `True/True` |
| `trt10-full-plan-refittable-fact` | `True` | `False/False` |
| `trt10-reload-metadata` | `True` | `2/5/1` |
| `trt10-context-gate` | `True` | `True/True` |
| `trt10-runtime-selected` | `True` | `True/True` |
| `trt10-same-process-inference` | `True` | `True/True` |
| `trt10-same-report-hash` | `True` | `f627e231af9ee570f6a3952d7a6404f9661039fc606e86193d1e70f8b297080d` |
| `trt10-same-output-artifact-hash` | `True` | `7112daa43f8d3a55a06a5ff1acb6b069e04866ede0203e5c4158fa849d2683d2` |
| `trt10-second-report-hash` | `True` | `bf65db9e214961cfda37ea1ca314699ed726f8c4499b7a9e09026b59f7c727a0` |
| `trt10-second-output-artifact-hash` | `True` | `75ae3fc7e1f85f993d57a7b4ea36bf6174beac49be81f6754b16f37748a013fc` |
| `trt10-baseline-report-hash` | `True` | `cf8a5eb8540ffc850f700bd4a230cb0e18e7da35fa2307d8615ff27af237d04e` |
| `trt10-baseline-output-artifact-hash` | `True` | `1182eba59e830dffcf81f2db6f1bf0c543fa35afebf7dfccd0f9f6589a5d6500` |
| `trt10-second-process-independent-command` | `True` | `--tensor-rt-line 10 --loadEngine E:\GitSpace\TensorRT-CSharp-API-4.0\TensorRtSharp4.0\artifacts\real-case\trtexec-refitted-plan-persistence\mnist-refitted-persisted.plan --workspace 64 --profilingVerbosity layer_names_only --exportReport E:\GitSpace\TensorRT-CSharp-API-4.0\TensorRtSharp4.0\artifacts\real-case\trtexec-refitted-plan-persistence\trt10-second-process-load-persisted.json --batch 2 --iterations 1 --warmUp 0 --duration 0 --streams 1 --builderOptimizationLevel 3 --loadInputs Input3:E:\GitSpace\TensorRT-CSharp-API-4.0\TensorRtSharp4.0\artifacts\real-case\onnx-to-engine-mnist-trt10-runtime\digit-7\mnist-trt10-7-input-f32.bin --exportOutput E:\GitSpace\TensorRT-CSharp-API-4.0\TensorRtSharp4.0\artifacts\real-case\trtexec-refitted-plan-persistence\trt10-second-process-output.json` |
| `trt10-second-process-reload` | `True` | `True/True/True` |
| `trt10-three-way-output-match` | `True` | `6f5771d6c5b056406c190a59e725cf9bb13c1f148c1ef06f99ed8acfb11b9041` |
| `trt10-process-exits` | `True` | `0/0/0` |
| `trt8-report-hash` | `True` | `b801bb53913065180e8c82392a638aafa317b5c99cb2deb1d0de1e5cf9175e18` |
| `trt8-dry-parse-only` | `True` | `dry-run-precheck/True` |
| `trt8-nondry-guard` | `True` | `2/TensorRT 8 has no ONNX parser-refitter API` |
| `trt11-same-report-hash` | `True` | `18ca7c2bd477ecf699406bcfd3b74d7fba07cb683e810a9ab12d1bdbcb8aa2e5` |
| `trt11-same-output-artifact-hash` | `True` | `d8d40cb3d174a742dc600eeff564b9df0b25228012546cb4f59af273946a7217` |
| `trt11-same-validation-hash` | `True` | `f57a229840d86ad938f268ae45f9f8ef3f24d8bd9fc7755bd0a04ddcf6ccb756` |
| `trt11-second-report-hash` | `True` | `5f45fd23017b35064d9cb8c2f7b753320d5764cc1bcf5f0eb6b61fd36c1f43cd` |
| `trt11-second-output-artifact-hash` | `True` | `4a981597505d1ca6682ae557230ac0fddf85e80a3628b6f73f22c1d1cbc7f469` |
| `trt11-second-validation-hash` | `True` | `b6144334d07d98e24d527a8f583cda8eaba116913875452c26d39d7e1a7ebd94` |
| `trt11-state` | `True` | `external-onnx-refit-reload-reference-validated-runtime/True/False` |
| `trt11-applied` | `True` | `--tensor-rt-line,--workspace,--builderOptimizationLevel,--loadEngine,--stripWeights,--refit,--refitFromOnnx,--saveRefittedEngine,--minShapes/--optShapes/--maxShapes,--saveEngine,--loadInputs,--exportOutput,--referenceOutputs,--referenceAbsTolerance,--referenceRelTolerance,--referenceNaNPolicy,--referenceInfinityPolicy,--iterations,--warmUp,--duration,--streams` |
| `trt11-plan-hashes` | `True` | `1ad0ba680913fc50e2278bd8c1d7ed357b9fb3e6b9e766d435f907911b8b86c7/c20d0797a16dcc553e714f9de719d9bcd9f6f6adbb3f4dab0eef089e6c2e04c9` |
| `trt11-exclude-weights` | `True` | `3/2` |
| `trt11-owner-dispose-reload` | `True` | `True/True/True` |
| `trt11-reload-metadata` | `True` | `2/5/1` |
| `trt11-runtime-selected` | `True` | `True/True` |
| `trt11-second-process-independent-command` | `True` | `--tensor-rt-line 11 --loadEngine E:\GitSpace\TensorRT-CSharp-API-4.0\TensorRtSharp4.0\artifacts\real-case\tensorrtexec-trt11-refit-lifecycle-20260810-091859\mnist-trt11-refitted-full-weight.plan --workspace 64 --profilingVerbosity layer_names_only --exportReport E:\GitSpace\TensorRT-CSharp-API-4.0\TensorRtSharp4.0\artifacts\real-case\tensorrtexec-trt11-refit-lifecycle-20260810-091859\second-process-reload\report.json --batch 2 --iterations 1 --warmUp 0 --duration 0 --streams 1 --builderOptimizationLevel 3 --loadInputs Input3:E:\GitSpace\TensorRT-CSharp-API-4.0\TensorRtSharp4.0\artifacts\real-case\onnx-to-engine-mnist-trt10-runtime\digit-7\mnist-trt10-7-input-f32.bin --exportOutput E:\GitSpace\TensorRT-CSharp-API-4.0\TensorRtSharp4.0\artifacts\real-case\tensorrtexec-trt11-refit-lifecycle-20260810-091859\second-process-reload\output.json --referenceOutputs Plus214_Output_0:E:\GitSpace\TensorRT-CSharp-API-4.0\TensorRtSharp4.0\artifacts\real-case\onnx-to-engine-mnist-trt10-runtime\digit-7\mnist-trt10-7.reference.json --referenceAbsTolerance 0.0001 --referenceRelTolerance 0.0001 --referenceNaNPolicy reject --referenceInfinityPolicy exact` |
| `trt11-second-process-reload` | `True` | `load-engine-reference-validated-runtime/True/True` |
| `trt11-output-match` | `True` | `3e4d0227be4c8e760f58a0b306ae7153df78b56586b5c3f5ddc658319e5c5b7c/3e4d0227be4c8e760f58a0b306ae7153df78b56586b5c3f5ddc658319e5c5b7c` |
| `trt11-strict-validations` | `True` | `tensor-rt-exec-report-ready/0/tensor-rt-exec-report-ready/0` |
| `trt11-process-exits` | `True` | `0/0` |
| `trt11-historical-probe-retained` | `True` | `dependency-probe-only/artifacts/real-case/trtexec-refitted-plan-persistence/trt11-save-refitted-dependency-probe.json` |
| `boundary-local` | `True` | `True` |
| `boundary-no-accuracy` | `True` | `False` |
| `boundary-trt11-local` | `True` | `True/True` |
| `boundary-no-package` | `True` | `False/False/False/False/False/False` |
| `boundary-no-publish` | `True` | `False/False` |
| `boundary-no-release-close` | `True` | `False` |
