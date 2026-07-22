# TensorRtExec ONNX Refit Lifecycle Validation

- Strict: `True`
- Checks: `24`
- Passed: `24`
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
| `trt11-dependency` | `True` | `dependency-probe-only/True` |
| `boundary-local` | `True` | `True` |
| `boundary-no-accuracy` | `True` | `False` |
| `boundary-no-persistence` | `True` | `False` |
| `boundary-no-package` | `True` | `False` |
| `boundary-no-publish` | `True` | `False/False` |
