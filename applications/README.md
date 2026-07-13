# Applications

This directory contains user-facing tools built on top of the TensorRtSharp libraries and shared tool services.

| Directory | Purpose | Status |
| --- | --- | --- |
| `TensorRtExec` | trtexec-like command line and WinForms front end for ONNX-to-engine workflows, backed by `JYPPX.TensorRtSharp.Tools` | build/report capable; external-model inference still needs explicit binding semantics |

Application evidence is separate from release proof. A tool being buildable does not mean a public package was published or that asset-dependent runtime smoke passed.
# Applications

This directory is reserved for user-facing applications that are larger than a sample and closer to a product workflow.

## Current Application Tracks

| Directory | Purpose | Status |
| --- | --- | --- |
| `TensorRtExec` | trtexec-like ONNX to TensorRT engine tool with console workflow and WinForms track | in-progress |

`TensorRtExec` is the application lane for the user request to provide both command-line usage and a Windows form workflow. The roadmap lives in `docs/articles/zh-cn/tensorrtexec-console-winforms-application-roadmap.md`.
The GUI/CLI surface contract lives in `TensorRtExec/tensor-rt-exec-gui-cli-field-map.json` and `TensorRtExec/tensor-rt-exec-gui-cli-field-map.md`; it ties WinForms fields to normalized CLI options without promoting command previews, screenshots, reports, or sidecars to runtime proof.

## Proof Boundary

Application reports, command previews, GUI screenshots, dry-runs, build-only reports, load-engine diagnostics, timing cache files, and sidecars are not package-consumer-runtime proof. Real release proof still requires Owner-provided public package metadata, CleanConsumer restore/build/smoke logs, PostPublish logs, host runtime metadata, rollback review, and final release close approval.
