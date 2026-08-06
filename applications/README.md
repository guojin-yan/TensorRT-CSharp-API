# Applications

English | [简体中文](README.zh-CN.md)

`applications/` contains complete workflows that are larger than a focused sample and have their own user-facing
configuration, reports, and articles.

## Application Catalog

| Application | Scope | Package boundary | Main article series |
| --- | --- | --- | --- |
| [`YoloVision`](YoloVision/README.md) | YOLO-family `det`, `cls`, `seg`, `obb`, `pose`, and `sem`, including preprocessing, output-role routing, postprocessing, JSON, and rendered result images | Executable only; consumes the published TensorRT managed package and the project-owned OpenCV package; no sample NuGet package | [YOLO overview](../docs/articles/zh-cn/yolovision-sample-overview.md) |
| [`OnnxToEngine`](OnnxToEngine/README.md) | ONNX parser/build reports, dynamic profiles, MNIST reference flow, and conversion diagnostics | Executable only; core TensorRT/CUDA APIs come from the published 4-series package, while shared application Tools source is compiled locally and is not packaged | [ONNX conversion guide](../docs/articles/zh-cn/onnx-to-engine-quickstart.md) |
| [`TensorRtExec`](TensorRtExec/README.md) | Windows CLI and WinForms deployment tool for engine build, load, refit, bindings, inference, and reports | Executable only; TensorRT/CUDA vendor libraries are installed by the user | [TensorRtExec guide](../docs/articles/zh-cn/tensorrtexec-tool-getting-started.md) |

The historical isolated Refitted Plan validation is documented in
[the local package-consumer article](../docs/articles/zh-cn/tensorrtexec-refitted-plan-local-package-consumer.md).
Its project template now lives under `tests/fixtures/package-consumers`, outside the user-facing sample catalog.

## Public Dependencies

`YoloVision` uses the published 4-series `JYPPX.TensorRT.CSharp.API` dependency through its local
`Directory.Build.props`. JPEG/PNG input decoding uses `JYPPX.OpenCV.CSharp.API`; on Windows x64 the matching
`JYPPX.OpenCV.runtime.win-x64` package is restored. The applications never redistribute CUDA, cuDNN, TensorRT,
NVRTC, or OpenCV vendor binaries outside their NuGet package contracts.

Application projects are not NuGet libraries. In particular, `JYPPX.TensorRT.CSharp.API.YoloVision` and
`JYPPX.TensorRT.CSharp.API.Classification` are not published packages. Reuse the managed API from a consumer project;
copying an application's source is a separate, explicit choice.

`OnnxToEngine` and `TensorRtExec` reference `applications/_shared/JYPPX.TensorRtSharp.ApplicationTools`. That
non-packable application project links the repository's Tools implementation but compiles it against the published
managed package. Neither application has a `ProjectReference` to the core `src/JYPPX.CudaSharp` or
`src/JYPPX.TensorRtSharp` projects.

## Model And Article Workflow

Each model-backed application article must state the upstream project and license, acquisition URL or command, ONNX
conversion command, input/output contract, preprocessing, postprocessing, runtime prerequisites, and the exact
rendered result artifact. Model binaries remain in the sibling `models/` staging directory and are excluded from Git.
Articles should use repository-relative commands and placeholders such as `<model>` and `<image>`, never a machine's
absolute drive path.

## Environment Boundary

The host user installs matching CUDA, cuDNN, TensorRT, NVRTC, and (where required) OpenCV native runtime packages.
If the local machine cannot provide a runtime line, the application must report `blocked-by-cuda-driver` or another
explicit environment state and the article must say so. A build-only or preflight result is not runtime proof.
Build-only, preflight, and parse-only results are also not package-consumer-runtime proof.

Do not trigger an Action or publish a package until the affected application restores, builds, and runs locally with
the intended package graph. Release and NuGet publication remain a separate, owner-authorized step.
