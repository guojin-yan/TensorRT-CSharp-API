# TensorRtExec ONNX Refit Lifecycle Validation

- Strict: `True`
- Checks: `35`
- Passed: `35`
- Failed: `0`

| Check | Passed | Actual |
| --- | --- | --- |
| `schema` | `True` | `trtexec-onnx-refit-lifecycle-evidence.v1` |
| `contract-requires-onnx` | `True` | `--onnx,--stripWeights,--refit` |
| `contract-requires-strip` | `True` | `--onnx,--stripWeights,--refit` |
| `contract-requires-refit` | `True` | `--onnx,--stripWeights,--refit` |
| `trt10-strip-applied` | `True` | `True` |
| `trt10-refit-applied` | `True` | `True` |
| `trt10-refittable-before` | `True` | `True` |
| `trt10-parser-refit` | `True` | `True` |
| `trt10-engine-refit-commit` | `True` | `True` |
| `trt10-refittable-after` | `True` | `True` |
| `trt10-all-inventory` | `True` | `6/6` |
| `trt10-missing-zero` | `True` | `0` |
| `trt10-parser-errors-zero` | `True` | `0` |
| `trt10-context-gate` | `True` | `True/True` |
| `trt10-inference` | `True` | `True` |
| `trt10-output-shape` | `True` | `10/40` |
| `trt10-output-match` | `True` | `6f5771d6c5b056406c190a59e725cf9bb13c1f148c1ef06f99ed8acfb11b9041` |
| `trt8-precheck` | `True` | `dry-run-precheck/True` |
| `trt11-report-hash` | `True` | `18ca7c2bd477ecf699406bcfd3b74d7fba07cb683e810a9ab12d1bdbcb8aa2e5` |
| `trt11-output-artifact-hash` | `True` | `d8d40cb3d174a742dc600eeff564b9df0b25228012546cb4f59af273946a7217` |
| `trt11-strict-validation-hash` | `True` | `f57a229840d86ad938f268ae45f9f8ef3f24d8bd9fc7755bd0a04ddcf6ccb756` |
| `trt11-state` | `True` | `external-onnx-refit-reload-reference-validated-runtime/True/False` |
| `trt11-options-applied` | `True` | `--tensor-rt-line,--workspace,--builderOptimizationLevel,--loadEngine,--stripWeights,--refit,--refitFromOnnx,--saveRefittedEngine,--minShapes/--optShapes/--maxShapes,--saveEngine,--loadInputs,--exportOutput,--referenceOutputs,--referenceAbsTolerance,--referenceRelTolerance,--referenceNaNPolicy,--referenceInfinityPolicy,--iterations,--warmUp,--duration,--streams` |
| `trt11-refit-gates` | `True` | `onnx-refit-complete/True` |
| `trt11-refit-inventory` | `True` | `0/6/0` |
| `trt11-model-hash` | `True` | `26454/2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf` |
| `trt11-runtime-output` | `True` | `True/True/3e4d0227be4c8e760f58a0b306ae7153df78b56586b5c3f5ddc658319e5c5b7c` |
| `trt11-reference-comparison` | `True` | `0/1.9073486E-06` |
| `trt11-strict-validation` | `True` | `tensor-rt-exec-report-ready/0/69` |
| `trt11-historical-probe-retained` | `True` | `dependency-probe-only/artifacts/real-case/trtexec-onnx-refit-lifecycle/trt11-refit-dependency-probe.json` |
| `boundary-local` | `True` | `True` |
| `boundary-no-accuracy` | `True` | `False` |
| `boundary-local-persistence` | `True` | `True` |
| `boundary-no-package` | `True` | `False` |
| `boundary-no-publish` | `True` | `False/False` |
