# TensorRtExec Feature Matrix

`applications/TensorRtExec/tensor-rt-exec-feature-matrix.json` records CLI and WinForms coverage for the trtexec-like application.

| Feature | Modes | Status |
| --- | --- | --- |
| CLI argument parsing | CLI | implemented |
| WinForms parameter entry | WinForms | implemented |
| ONNX build-only report | CLI, WinForms | implemented |
| Dry-run/precheck report | CLI, WinForms | implemented |
| Shape profile configuration | CLI, WinForms | implemented |
| Precision switches | CLI, WinForms | wrapper-ready-plus-parse-report-only |
| Timing and profiling options | CLI, WinForms | implemented-report |
| Load-engine bounded runtime | CLI, WinForms | bounded-runtime-output |
| Layer/profile diagnostic switches | CLI, WinForms | implemented-report |
| Report alias compatibility | CLI | implemented-report |
| Refit, weight streaming, and debug tensor diagnostics | CLI, WinForms | parse-report-only |
| DLA/device options | CLI, WinForms | diagnostic |
| Timing cache lifecycle | CLI, WinForms | implemented-build-cache-lifecycle |
| Full trtexec parity | CLI, WinForms | planned |
| Release proof promotion | CLI, WinForms | blocked by runtime proof |

The application can produce useful build/report evidence, but those reports are not runtime proof, post-publish proof, publish approval, release close approval, or package push.
