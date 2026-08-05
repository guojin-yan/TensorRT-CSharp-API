# Applications

This directory contains user-facing tools that are larger than a sample and represent complete deployment workflows built on TensorRtSharp libraries and shared tool services.

## Current application

| Directory | Purpose | Current status |
| --- | --- | --- |
| `TensorRtExec` | trtexec-like CLI and WinForms application for ONNX build, Engine load, refit, report export, binding, inference, and diagnostics | Release build, CLI/GUI workflow, real MNIST runtime, refitted-plan persistence/reload, and local package-consumer runtime validated on TensorRT 10.11; public package and post-publish validation remain pending |

The GUI/CLI field contract is `TensorRtExec/tensor-rt-exec-gui-cli-field-map.json`. User-facing walkthroughs are under `docs/articles/zh-cn`, including `tensorrtexec-gui-user-guide.md` and `tensorrtexec-refitted-plan-local-package-consumer.md`.

## Proof boundary

Application reports, command previews, screenshots, dry-runs, build-only results, local-file-feed consumers, and source-tree runtime records are not package-consumer-runtime proof and do not prove that a public package was published. Public package restore, post-publish smoke, compatible-host metadata, rollback review, and Owner release approval remain separate release gates. No application command in this directory authorizes package push, GitHub Release creation, model upload, or NVIDIA runtime redistribution.
