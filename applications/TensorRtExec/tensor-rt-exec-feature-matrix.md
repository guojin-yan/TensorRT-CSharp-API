# TensorRtExec Feature Matrix

`applications/TensorRtExec/tensor-rt-exec-feature-matrix.json` records CLI and WinForms coverage for the trtexec-like application.

| Feature | Modes | Status |
| --- | --- | --- |
| CLI argument parsing | CLI | implemented |
| WinForms parameter entry | WinForms | implemented |
| ONNX build-only report | CLI, WinForms | implemented |
| Dry-run/precheck report | CLI, WinForms | implemented |
| Shape profile configuration | CLI, WinForms | implemented |
| Timing iteration builder readback | CLI, WinForms | implemented-builder-config-readback |
| Precision switches | CLI, WinForms | wrapper-ready-plus-parse-report-only |
| I/O format and layer precision policies | CLI, WinForms | implemented-build-readback-with-version-guards |
| Timing and profiling options | CLI, WinForms | implemented-report |
| Load-engine bounded runtime | CLI, WinForms | bounded-runtime-output |
| Multi-output runtime artifacts | CLI, WinForms | implemented-bounded-multi-output-capture |
| Layer/profile diagnostic switches | CLI, WinForms | implemented-inspector-readback |
| Report alias compatibility | CLI | implemented-report |
| FP8, best, refit dump, and debug tensor diagnostics | CLI, WinForms | parse-report-only |
| Engine packaging, ONNX refit, and weight streaming | CLI, WinForms | implemented-build-refit-runtime-with-version-guards |
| Refitted plan persistence and independent reload | CLI, WinForms | implemented-trt10-persist-dispose-reload-runtime |
| Device and deployment policies | CLI, WinForms | implemented-build-readback-with-version-guards |
| Timing cache lifecycle | CLI, WinForms | implemented-build-cache-lifecycle |
| Full trtexec parity | CLI, WinForms | planned |
| Release proof promotion | CLI, WinForms | blocked by runtime proof |

The application can produce useful build/report evidence, but those reports are not runtime proof, post-publish proof, publish approval, release close approval, or package push.
