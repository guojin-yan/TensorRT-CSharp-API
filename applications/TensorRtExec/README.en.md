# TensorRtExec

English | [简体中文](README.md)

`applications/TensorRtExec` is an end-user ONNX-to-TensorRT Engine tool with both CLI and WinForms entry points. The two front ends share the trtexec-like command model and `TensorRtExecService` through the non-packable application Tools project. That project compiles against the published 4-series managed package, so model conversion, builder options, report export, and proof boundaries keep the same semantics without referencing the core source projects.

## Positioning

- CLI: automation, local build records, prechecks, and report archiving.
- WinForms: Windows desktop selection of ONNX/Engine files, shape profiles, report paths, and evidence sidecars.
- Shared service: CLI and GUI invoke the same build/report implementation.
- Proof boundary: external ONNX is build/report capable by default; model runtime proof requires explicit inputs, binding/output semantics, output validation, and a real run log.

## Common Commands

Build an external ONNX model without running inference:

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --save-engine .\models\model.plan `
  --minShapes input:1x3x640x640 `
  --optShapes input:1x3x640x640 `
  --maxShapes input:4x3x640x640 `
  --fp16 `
  --workspace 1GiB `
  --timingCache .\models\model.cache `
  --builderOptimizationLevel 4 `
  --maxAuxStreams 2 `
  --buildOnly `
  --exportReport .\artifacts\tensorrtexec\model-build-report.json
```

Normalize and preview options without reading ONNX or probing TensorRT:

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --save-engine .\models\model.plan `
  --minShapes input:1x3x640x640 `
  --optShapes input:1x3x640x640 `
  --maxShapes input:4x3x640x640 `
  --dryRun `
  --exportReport .\artifacts\tensorrtexec\model-precheck-report.md
```

The shared parser covers common build, profile, memory-pool, plugin, timing, engine-load, refit, output-capture, and reference-validation options. `--help-json` and `--capabilities-json` return the machine-readable implementation status without loading CUDA, TensorRT, an ONNX model, a plugin, or an Engine.

## WinForms

The Windows desktop front end edits the same normalized command consumed by the CLI. A publishable GUI article must show the actual application window, selected inputs, generated command, final status, and generated report. A screenshot proves the visible workflow only; it does not replace a successful TensorRT runtime log or validated model output.

See [TensorRtExec getting started](../../docs/articles/zh-cn/tensorrtexec-tool-getting-started.md) and the [GUI user guide](../../docs/articles/zh-cn/tensorrtexec-gui-user-guide.md) for the current workflow.

## Models And Reports

Keep ONNX, Engine, timing cache, tensors, and raw reports in the workspace-level `models` or a caller-selected artifact directory outside Git. Articles must record the model source and license, acquisition and ONNX conversion steps, input/output contract, exact command, hashes, host/runtime versions, a real terminal screenshot, and the GUI screenshot when the desktop front end is used.

Reports distinguish `precheck`, `build-only`, readonly Engine diagnostics, bounded synthetic runtime, real-model runtime, and package-consumer runtime. Build reports, GUI screenshots, local file feeds, ProjectReference runs, and direct `.nupkg` runs cannot be promoted to public-package or post-publish proof by themselves.

## Relationship To Other Applications

- `applications/OnnxToEngine` is the focused parser/build and MNIST reference workflow.
- `samples/ComputerVision/01.Classification` defines image-classification preprocessing and Top-K semantics.
- `applications/YoloVision` defines YOLO detection, classification, segmentation, OBB, pose, and semantic-segmentation output semantics.

Use TensorRtExec for build and deployment diagnostics, then use the model-specific runner for real input, task decoding, rendered output, and independent reference validation.

## Local Quality Checks

Restore and build locally before changing an Action workflow or publishing:

```powershell
dotnet build .\applications\TensorRtExec\TensorRtExec.csproj -c Release
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Release --filter TensorRtExec
```

CUDA, cuDNN, TensorRT, and NVRTC remain user-installed prerequisites. Missing host capability must be reported as an explicit environment blocker.
