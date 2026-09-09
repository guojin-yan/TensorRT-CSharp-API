# Samples

English | [简体中文](README.zh-CN.md)

`samples/` contains small, focused, runnable workflows. The directory name is part of the learning path: each
capability is a module and each case has a stable numeric prefix.

Only the numbered projects under `Cuda`, `Inference`, `Diagnostics`, `Performance`, and `ComputerVision` are user-facing samples.
`_shared` contains source files compiled into those projects, while `assets` contains lightweight manifests and
evidence metadata. Package-consumer templates and independent reference programs live under `tests/fixtures`; they
are test inputs, not additional samples.

## Case Series

| Module | Case | What it demonstrates | Article entrypoint |
| --- | --- | --- | --- |
| `Cuda` | `Cuda/01.RuntimeCompilation` | CUDA RTC source compilation, module loading, typed launch, and readback | [CUDA RTC article](../docs/articles/zh-cn/cuda-runtime-compilation-technical-article.md) |
| `Inference` | `Inference/01.Bindings` | TensorRT input/output bindings and host/device ownership | [Inference bindings tutorial](../docs/articles/zh-cn/inference-bindings-tutorial.md) |
| `Inference` | `Inference/02.DynamicShapes` | Explicit optimization profiles and dynamic shape execution | [Dynamic shape tutorial](../docs/articles/zh-cn/dynamic-shape-optimization-profile-tutorial.md) |
| `Inference` | `Inference/03.OnnxBuildAndRun` | Public-package ONNX parsing, engine build, one inference, and structured JSON | [Sample README](Inference/03.OnnxBuildAndRun/README.md) |
| `Inference` | `Inference/04.RefittedPlan` | ONNX initializer refit, plan persistence, disk reload, and output verification | [Sample README](Inference/04.RefittedPlan/README.md) |
| `Diagnostics` | `Diagnostics/01.CallbackLifecycle` | Logger, ProgressMonitor, Profiler, and DebugListener ownership and detach order | [Sample README](Diagnostics/01.CallbackLifecycle/README.md) |
| `Performance` | `Performance/01.MultiStream` | CUDA streams, events, and cross-stream ordering | [Multi-stream tutorial](../docs/articles/zh-cn/cuda-stream-event-multistream-tutorial.md) |
| `ComputerVision` | `ComputerVision/01.Classification` | Image preprocessing, TensorRT classification, Top-K, JSON, and an annotated result image | [Classification walkthrough](../docs/articles/zh-cn/classification-real-asset-walkthrough.md) |

Larger end-to-end workflows are under [`applications/`](../applications/README.md): `YoloVision` is the unified
six-task YOLO-family application, and `OnnxToEngine` is the ONNX-to-engine conversion application. `TensorRtExec`
is the desktop/CLI application.

## Package Boundary

Every runnable sample in this directory consumes the published managed package through
[`build/JYPPX.PublicSamplePackages.props`](../build/JYPPX.PublicSamplePackages.props). The repository keeps the
current 4-series dependency rule in that one file instead of repeating a package version in every project. This
central guard also prevents historical packages with an incompatible API surface from being selected.

The sample projects are executables and are explicitly `IsPackable=false`. `Classification` and `YoloVision` are
examples, not `JYPPX.TensorRT.CSharp.API.Classification` or `JYPPX.TensorRT.CSharp.API.YoloVision` packages.

For a new consumer project, install the managed API from the 4 series and one matching project-owned bridge package.
CUDA, cuDNN, TensorRT, and NVRTC remain user-installed prerequisites and are never downloaded by these samples.

```powershell
dotnet add package JYPPX.TensorRT.CSharp.API --version 4.0.0
dotnet add package JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge --version 4.0.0
```

The exact `4.0.0` release avoids the API-incompatible historical `4.0.6170` package. Repository projects use the
same shared version so local and CI restores remain reproducible.

## Model And Image Assets

ONNX models, converted tensors, weights, labels, and article screenshots are intentionally kept outside Git. The
temporary workspace is the repository sibling `<workspace-root>/models`; the acquisition URL, license, export command, ONNX input /
output contract, SHA256, and conversion tool must be recorded in the corresponding article and manifest. Source files
must never assume a machine-specific drive-letter or absolute directory.

For image-backed cases, pass an image path supplied by the user. `Classification` and `YoloVision` use the project-owned
`JYPPX.OpenCV.CSharp.API` package for JPEG/PNG decoding on Windows x64 and retain a managed BMP/PPM fallback for all
platforms currently covered by the repository. The OpenCV native runtime package is not bundled into TensorRT packages.

## Run

Run commands from the repository root. The offline commands do not require CUDA or TensorRT:

```powershell
dotnet run --project .\samples\Cuda\01.RuntimeCompilation -- --help
dotnet run --project .\samples\Inference\01.Bindings -- --help
dotnet run --project .\samples\Inference\02.DynamicShapes -- --help
dotnet run --project .\samples\Inference\03.OnnxBuildAndRun -- --help
dotnet run --project .\samples\Inference\04.RefittedPlan -- --help
dotnet run --project .\samples\Diagnostics\01.CallbackLifecycle -- --help
dotnet run --project .\samples\Performance\01.MultiStream -- --help
dotnet run --project .\samples\ComputerVision\01.Classification -- --help
dotnet run --project .\applications\YoloVision -- --list-capabilities
```

To run a real model, first follow the matching article's acquisition and conversion steps, then provide `--model`,
`--labels`, `--image` or `--input-data`, and the exact shape/layout/output metadata. A successful build, preflight,
or SVG/PNG report is not by itself model-runtime proof; the article must include the actual command output and a
rendered result image.

## Evidence Ladder For Asset-Dependent Samples

| Level | Meaning | Boundary |
| --- | --- | --- |
| `precheck` | Arguments, assets, shape, and output metadata are validated | not runtime proof |
| `build-only` | ONNX parsing or engine serialization produced a report | not inference proof |
| `synthetic-input-runtime` | The pipeline ran with generated input | not real-image proof |
| `real-model-runtime` | Real model, image, labels, hashes, output comparison, and rendered result agree | sample-level proof only |
| `package-consumer-runtime` | A clean external project restored published packages and ran successfully | release proof only |

`package-consumer-runtime` belongs to release proof records, not to an ordinary source-tree sample run. Any missing
CUDA/TensorRT/OpenCV native runtime is recorded as an environment blocker rather than silently reported as success.

## Adding A Case

1. Choose a capability module and the next numeric case directory.
2. Keep the project executable and `IsPackable=false`.
3. Consume the published managed package; do not add a source `ProjectReference` to `src/JYPPX.TensorRtSharp` or `src/JYPPX.CudaSharp`.
4. Add deterministic inputs, a structured report, a rendered result, and a focused bilingual article.
5. Add the project to `TensorRtSharp.sln`, this table, and the appropriate application/article index.
6. Build and run locally before changing an Action workflow or publishing a package.
