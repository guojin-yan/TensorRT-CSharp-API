# Trtexec Engine Packaging Runtime Evidence Validation

- State: `passed`
- Checks: `20`
- Failures: `0`

| Check | Passed | Detail |
|---|---:|---|
| schema | True | trtexec-engine-packaging-runtime-evidence.v1 |
| trt8-version-compatible | True | TRT8 version-compatible flag readback |
| trt8-exclude-lean | True | TRT8 exclude-lean flag readback |
| trt8-refit-conflict-guard | True | TRT8 version-compatible+refit remains guarded |
| trt8-strip-guard | True | TRT8 strip remains guarded |
| trt8-streaming-guard | True | TRT8 streaming remains guarded |
| trt10-version-refit-runtime | True | TRT10 version-compatible/refit identity runtime |
| trt10-refittable-readback | True | Config and engine refit readback |
| trt10-runtime-host-code | True | Runtime host code enabled for version-compatible plan |
| trt10-strip-refit-identical | True | StripPlan plus RefitIdentical |
| trt10-weighted-model | True | Non-zero real model streamable weights |
| trt10-percentage-budget | True | 50 percent budget resolved and read back |
| trt10-streaming-scratch | True | Non-zero scratch bytes prove active streaming budget |
| trt10-context-order | True | Budget applied before execution context |
| trt10-weighted-enqueue | True | Weighted model enqueue with conservative output classification |
| trt10-load-diagnostics | True | Load-engine readonly diagnostics succeeded |
| trt10-load-auto-budget | True | Automatic budget resolved and read back |
| trt10-load-enqueue | True | Load-engine bounded runtime executed |
| trt11-dependency-boundary | True | TRT11 known structured exception remains dependency-only |
| proof-boundary | True | No proof or publish promotion |
