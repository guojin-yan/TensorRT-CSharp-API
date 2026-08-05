# TensorRtSharp4.0

[![Build](https://github.com/guojin-yan/TensorRT-CSharp-API/actions/workflows/ci-validation.yml/badge.svg)](https://github.com/guojin-yan/TensorRT-CSharp-API/actions/workflows/ci-validation.yml)
[![Documentation](https://img.shields.io/badge/docs-DocFX-2f80ed)](https://guojin-yan.github.io/TensorRT-CSharp-API/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

TensorRtSharp4.0 provides a .NET API for TensorRT inference, CUDA runtime compilation, memory, streams, callbacks, and the TensorRtExec desktop workflow. The first public candidate is planned as <code>4.0.0-preview.1</code>; it is a source and package candidate, not a claim that every TensorRT feature has completed runtime validation.

## Introduction

The managed API keeps the public namespace roots stable:

- <code>JYPPX.TensorRtSharp</code> is the main TensorRT surface and the default home for shared types.
- <code>JYPPX.CudaSharp</code> is the CUDA surface.
- Native bridge loading is explicit through <code>jyppxtrtbridge</code> and the versioned bridge package.

CUDA, cuDNN, TensorRT, and NVRTC are user-installed prerequisites. NVIDIA runtime redistribution is retired: this repository publishes managed source/packages and project-owned bridge-only packages, never CUDA/cuDNN/TensorRT vendor archives.

## Release Highlights

- C# bindings and source are grouped by module under <code>src</code>.
- TensorRT engine building, execution contexts, bindings, dynamic-shape profiles, allocators, logging, profiling, progress monitoring, streams, events, CUDA graphs, and CUDA RTC are covered by the managed API.
- Runtime package roles are explicit: the `.Bridge` packages with `split_package_roles=bridge` contain only the project bridge for a fixed CUDA/TensorRT line.
- NuGet branding is fixed: <code>nuget/logo.jpg</code> is the package logo and the English root README is embedded as the package README.
- The release license is Apache-2.0.

## Get Started In 30 Seconds

Create a console project, reference the managed package, and install the bridge package that matches the CUDA/TensorRT installation on the target machine:

~~~powershell
dotnet new console -n TrtQuickstart
cd TrtQuickstart
dotnet add package JYPPX.TensorRT.CSharp.API --version 4.0.0-preview.1
dotnet add package JYPPX.TensorRT.CSharp.API.Bridge.win-x64-trt10.11-cuda12.9-cudnn9.22 --version 4.0.0-preview.1
~~~

Then create a runtime, load an engine, bind input/output tensors, execute, and read the result. The bridge package is not a replacement for the user-installed NVIDIA runtime. See the [inference bindings tutorial](docs/articles/zh-cn/inference-bindings-tutorial.md) and [Windows installation guide](docs/articles/zh-cn/windows-installation-and-troubleshooting-guide.md).

## Package Layout

| Package | Contents |
| --- | --- |
| <code>JYPPX.TensorRT.CSharp.API</code> | Managed TensorRT and CUDA-facing C# API |
| <code>JYPPX.TensorRT.CSharp.API.YoloVision</code> | Managed YOLO output decoders and image pipeline |
| <code>JYPPX.TensorRT.CSharp.API.Classification</code> | Classification helpers and sample contracts |
| <code>JYPPX.TensorRT.CSharp.API.Bridge.*</code> | Project-owned native bridge only, selected by installed CUDA/TensorRT versions |

Runtime packages do not bundle NVIDIA libraries. For local source builds, use the scripts in <code>eng</code> only through the documented entry points; most exporter and owner-proof scripts are internal engineering tools.

## Native Dependencies

Install the matching NVIDIA stack before running a bridge package:

| Example line | Expected user installation |
| --- | --- |
| Windows x64 TRT 10.11 / CUDA 12.9 | TensorRT 10.11, CUDA 12.9, cuDNN 9.22 |
| Linux x64 TRT 11.0 / CUDA 13.2 | TensorRT 11.0, CUDA 13.2, cuDNN 9.22 |

Use <code>TENSORRT_PATH</code>, <code>JYPPX_TENSORRT_ROOT</code>, and the platform loader path appropriate for your machine. The repository does not upload or package these vendor runtimes.

## Models And ONNX Conversion

Demo models are staged outside Git in the sibling <code>models</code> directory and are not included in source archives or packages. Each article records the official acquisition URL, pinned revision, license, conversion command, input/output contract, and SHA256.

| Demo | Official source and conversion |
| --- | --- |
| MNIST | Project-generated digits; export with the sample PyTorch/ONNX script, then build a TensorRT engine with <code>trtexec</code>. |
| ResNet18 | torchvision official weights; export with <code>torch.onnx.export</code> using NCHW 224x224 and ImageNet normalization. |
| YOLOv8n detection/classification/segmentation/pose/OBB | Ultralytics official checkpoints; export with the pinned Ultralytics command and validate output names/shapes before TensorRT build. |
| YOLOv10n | THU-MIG official checkpoint; export with the repository reference script and preserve end-to-end output contract. |
| YOLOX-S | Megvii official checkpoint; export through the pinned YOLOX/ONNX path and validate decode metadata. |
| LRASPP MobileNetV3 Large | torchvision v0.25.0 official weights; export to <code>[1,21,320,320]</code> with ImageNet mean/std and compare argmax maps. |

See the [demo model inventory](samples/assets/demo-model-inventory.json), [acquisition and conversion guide](docs/articles/zh-cn/demo-model-acquisition-and-onnx-conversion.md), and <code>eng/Sync-DemoOnnxModels.ps1</code>. Model files stay in the external model store until ModelZoo is available.

## Samples And Visual Results

Every complete image article includes a real program result, an annotated image, and a terminal or GUI capture:

| Sample | Result |
| --- | --- |
| YOLOv8n detection | ![YOLOv8n detection](docs/images/yolovision-yolov8n-det-annotated-cc0.webp) |
| ResNet18 classification | ![ResNet18 classification](docs/images/classification-resnet18-annotated-cc0.webp) |
| LRASPP semantic segmentation | ![Semantic segmentation](docs/images/yolovision-lraspp-semantic-annotated-cc0.webp) |
| TensorRtExec GUI | ![GUI configuration](docs/images/tensorrtexec-gui-runtime-config.png) ![GUI result](docs/images/tensorrtexec-gui-runtime-result.png) |

## Documentation

- [English documentation](docs/index.md)
- [Chinese article catalog and publication criteria](docs/articles/zh-cn/README.md)
- [Project overview](docs/articles/zh-cn/project-overview.md)
- [Source organization](docs/articles/zh-cn/source-organization.md)
- [Model acquisition and ONNX conversion](docs/articles/zh-cn/demo-model-acquisition-and-onnx-conversion.md)
- [Inference bindings](docs/articles/zh-cn/inference-bindings-tutorial.md)
- [TensorRtExec GUI](docs/articles/zh-cn/tensorrtexec-gui-user-guide.md)
- [Release candidate gate](docs/articles/zh-cn/release-candidate-gate.md)
- [Release proof sample article](docs/articles/zh-cn/release-proof-sample-article-closure.md)
- [Release proof owner dashboard](docs/articles/zh-cn/release-proof-owner-input-dashboard.md)
- [TensorRtExec report boundary](docs/articles/zh-cn/tensorrtexec-report-proof-boundary.md)
- [ONNX to engine report boundary](docs/articles/zh-cn/onnx-to-engine-trtexec-proof-boundary.md)
- [YOLOVision asset evidence guide](docs/articles/zh-cn/yolovision-owner-asset-evidence-guide.md)
- [Callback/allocator safety gates](docs/articles/zh-cn/callback-allocator-listener-readonly-safety-gates.md)
- [API readiness audit](artifacts/interface-coverage/release-api-readiness-audit.json)
- [YOLOVision model matrix](samples/YoloVision/yolo-model-matrix.json)
- [TensorRtExec feature matrix](applications/TensorRtExec/tensor-rt-exec-feature-matrix.json)
- [ONNX-to-engine parity matrix](samples/OnnxToEngine/trtexec-parity-matrix.json)
- [Article roadmap 30-plus](docs/articles/zh-cn/article-roadmap-30plus.md)

## Build From Source

~~~powershell
dotnet restore TensorRtSharp.sln
dotnet build TensorRtSharp.sln -c Release
dotnet test tests/JYPPX.ProjectQuality.Tests/JYPPX.ProjectQuality.Tests.csproj -c Release --no-restore
~~~

Pack locally with <code>JYPPXPackageVersion=4.0.0-preview.1</code>. Inspect the generated nupkg before any upload; it must contain the package README and <code>logo.jpg</code> and must not contain CUDA, cuDNN, or TensorRT vendor binaries.

## Release And Action Policy

Workflows are manual-only to conserve Actions quota. The <code>grape-yan</code> repository is validation-only: it may receive one candidate Action after local checks pass and never publishes. The formal <code>guojin-yan</code> workflow is dispatched once, with explicit Owner approval, for the fixed first version <code>4.0.0-preview.1</code>.

Until that run completes, release artifacts remain <code>blocked</code>. The state is <code>owner-action-required</code>. <code>clean-consumer-proof-execution-bundle</code> and <code>clean-consumer-external-proof-closure-pack</code> are <code>non-proof</code> Owner actions; they do not run runtime smoke, are not runtime proof, and are not post-publish proof. Build-only output, local feeds, ProjectReference, dry-runs, templates, and dashboards are rejected by <code>FailOnNotProof</code> and cannot be promoted to publication or issue close.

NuGet publication requires push permission for `JYPPX.TensorRT.CSharp.API`, `JYPPX.TensorRT.CSharp.API.YoloVision`, and `JYPPX.TensorRT.CSharp.API.Classification`. A nuget.org `403` is an authorization failure, not a retryable build failure.

## Repository Layout

- <code>src</code>: managed interfaces grouped by CUDA, TensorRT, runtime, memory, and shared modules.
- <code>native</code>: project bridge source and ABI exports.
- <code>samples</code>: runnable C# demonstrations and model metadata.
- <code>applications/TensorRtExec</code>: desktop engine builder and runner.
- <code>pack</code>: managed and bridge-only package definitions.
- <code>docs</code>: DocFX site and technical articles.
- <code>eng</code>: build, acquisition, validation, and release engineering scripts.

## License

Licensed under Apache-2.0. See [LICENSE](LICENSE).

## Support

Please include the package version, CUDA/cuDNN/TensorRT versions, GPU model, operating system, and the failing command when opening an issue. Do not upload proprietary model weights or NVIDIA runtime archives.
*** End Patch
