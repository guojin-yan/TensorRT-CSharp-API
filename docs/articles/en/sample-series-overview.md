# TensorRtSharp4.0 Sample Series Learning Path

English | [简体中文](../zh-cn/sample-series-overview.md)

Use the [`samples` catalog](https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/samples/README.md) for focused runnable cases and the [`applications` catalog](https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/applications/README.md) for complete model and desktop workflows. Every user-facing README has an English and Simplified Chinese version.

All sample and application projects are executables with `IsPackable=false`. They consume the published 4-series `JYPPX.TensorRT.CSharp.API` package; Classification and YoloVision are examples, not extension NuGet packages. The host user installs CUDA, cuDNN, TensorRT, and NVRTC and selects a project-owned Bridge package matching the target platform and vendor-runtime versions.

## Package Setup

Use the `4.0.0-*` floating rule for the maintained preview line. A bare `--prerelease` may select the API-incompatible historical `4.0.6170` package:

```powershell
dotnet add package JYPPX.TensorRT.CSharp.API --version "4.0.0-*"
dotnet add package JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge --version "4.0.0-*"
```

The Bridge ID is an environment example. Choose the package matching the target OS, architecture, TensorRT, CUDA, and cuDNN line. Bridge packages do not install NVIDIA vendor runtimes.

Vision samples use the project owner's [OpenCV-CSharp-API](https://github.com/guojin-yan/OpenCV-CSharp-API):

```powershell
dotnet add package JYPPX.OpenCV.CSharp.API --prerelease
dotnet add package JYPPX.OpenCV.runtime.win-x64 --prerelease
```

## Learning Modules

| Order | Module and case | Learning goal | Main article |
| --- | --- | --- | --- |
| 01 | `Cuda/01.RuntimeCompilation` | Compile CUDA C, load a module, launch a typed kernel, and read results | [CUDA RTC](../zh-cn/cuda-runtime-compilation-technical-article.md) |
| 02 | `Inference/01.Bindings` | Tensor ownership, host/device memory, enqueue, and output readback | [Inference bindings](../zh-cn/inference-bindings-tutorial.md) |
| 03 | `Inference/02.DynamicShapes` | Optimization profiles and runtime dynamic shapes | [Dynamic shapes](../zh-cn/dynamic-shape-optimization-profile-tutorial.md) |
| 04 | `Performance/01.MultiStream` | CUDA streams, events, dependencies, and timing boundaries | [Multi-stream](../zh-cn/cuda-stream-event-multistream-tutorial.md) |
| 05 | `ComputerVision/01.Classification` | OpenCV decoding, ImageNet preprocessing, Top-K, JSON, and an annotated result | [ResNet18 classification](../zh-cn/classification-real-asset-walkthrough.md) |

Complete applications continue with:

| Application | Scope | Main article |
| --- | --- | --- |
| `YoloVision` | Detection, classification, instance segmentation, OBB, pose, semantic segmentation, reports, and visualizations | [All-task overview](../zh-cn/yolovision-all-task-overview.md) |
| `OnnxToEngine` | ONNX parsing, builder configuration, dynamic profiles, Engine serialization, and MNIST validation | [ONNX to Engine](../zh-cn/onnx-to-engine-quickstart.md) |
| `TensorRtExec` | CLI and WinForms model build/deployment workflow | [TensorRtExec GUI](../zh-cn/tensorrtexec-gui-user-guide.md) |

All three applications consume the published TensorRT managed package. `OnnxToEngine` and `TensorRtExec` share a
non-packable application Tools project that links the repository implementation and compiles it against that public
package; neither application references the core CUDA or TensorRT source projects.

## Model Staging

Converted ONNX files are staged in the workspace-level `models` directory beside the source repository. They are excluded from Git, NuGet, and Release assets until a separate model zoo is available.

Every model article must record the upstream project and license, pinned revision, acquisition command, framework-to-ONNX conversion command, tool versions, opset, input/output contract, preprocessing and postprocessing, and SHA256 values. Commands use workspace variables, repository-relative paths, or placeholders such as `<model>` and `<image>` instead of a contributor's drive letter and user name.

## Complete Article Standard

A publishable case article follows the full workflow rather than listing APIs:

1. Introduce the project, target reader, and responsibility of every dependency.
2. Acquire the model, license, labels, and test image; verify hashes.
3. Export ONNX and document versions, options, tensor names, shapes, layouts, and hashes.
4. Install public packages and prepare a matching user-owned runtime environment.
5. Explain the important code and run the exact command.
6. Preserve real stdout, compare against an independent reference, and run a controlled negative.
7. Render classification, detection, segmentation, OBB, pose, or semantic results onto the source image.
8. Include the real terminal window and, for desktop applications, the actual software window.

Preflight, build-only output, a generated report, or a screenshot cannot be promoted to real-model runtime evidence. When the local host lacks a compatible runtime, document the environment blocker and the installation requirement instead of presenting a simulated success.
