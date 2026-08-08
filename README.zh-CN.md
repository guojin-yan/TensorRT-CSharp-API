<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/readme/hero-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/images/readme/hero-light.svg">
  <img alt="TensorRtSharp4.0 - 面向 C# 与 .NET 的 TensorRT 和 CUDA 接口" src="docs/images/readme/hero-light.svg" width="100%">
</picture>

<h1 align="center">TensorRtSharp4.0</h1>

<p align="center">
  面向 C# 与 .NET 的 TensorRT、CUDA 托管接口、项目自有桥接包、可运行视觉示例和 TensorRtExec 桌面工具。
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/github/license/guojin-yan/TensorRT-CSharp-API.svg" alt="仓库许可证" /></a>
  <a href="https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/"><img src="https://img.shields.io/nuget/vpre/JYPPX.TensorRT.CSharp.API.svg" alt="NuGet 预览版本" /></a>
  <a href="https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/"><img src="https://img.shields.io/nuget/dt/JYPPX.TensorRT.CSharp.API.svg" alt="NuGet 下载量" /></a>
  <a href="https://github.com/guojin-yan/TensorRT-CSharp-API/releases"><img src="https://img.shields.io/github/v/release/guojin-yan/TensorRT-CSharp-API?include_prereleases&label=Release" alt="GitHub Release" /></a>
  <a href="https://github.com/guojin-yan/TensorRT-CSharp-API/stargazers"><img src="https://img.shields.io/github/stars/guojin-yan/TensorRT-CSharp-API?style=flat&amp;label=Stars" alt="GitHub Stars" /></a>
</p>

<p align="center">
  <a href="https://github.com/guojin-yan/TensorRT-CSharp-API/actions/workflows/ci-validation.yml"><img src="https://github.com/guojin-yan/TensorRT-CSharp-API/actions/workflows/ci-validation.yml/badge.svg?branch=TensorRtSharp4.0" alt="托管代码检查" /></a>
  <a href="https://github.com/guojin-yan/TensorRT-CSharp-API/actions/workflows/release-quality-gate.yml"><img src="https://github.com/guojin-yan/TensorRT-CSharp-API/actions/workflows/release-quality-gate.yml/badge.svg?branch=TensorRtSharp4.0" alt="发布质量门禁" /></a>
</p>

<p align="center"><a href="README.md">English</a> | <strong>简体中文</strong></p>

# TensorRtSharp4.0

[![构建](https://github.com/guojin-yan/TensorRT-CSharp-API/actions/workflows/ci-validation.yml/badge.svg)](https://github.com/guojin-yan/TensorRT-CSharp-API/actions/workflows/ci-validation.yml)
[![文档](https://img.shields.io/badge/docs-DocFX-2f80ed)](https://guojin-yan.github.io/TensorRT-CSharp-API/)
[![许可证](https://img.shields.io/github/license/guojin-yan/TensorRT-CSharp-API.svg)](LICENSE)

TensorRtSharp4.0 是面向 .NET 的 TensorRT / CUDA 桥接项目，包含推理执行、CUDA 运行时编译、显存、流与事件、回调、分配器以及 TensorRtExec 桌面工具。首个 4 系列预览版已经发布，当前开发重点转向使用公开包的系列案例、完整应用和配套技术文章。

## 项目简介

- 主 TensorRT 顶层命名空间为 JYPPX.TensorRtSharp，未单独归类的共享类型也归入该命名空间。
- CUDA 顶层命名空间为 JYPPX.CudaSharp。
- 原生桥接库为 jyppxtrtbridge，按 CUDA/TensorRT 版本提供项目自有 bridge-only 包。

CUDA、cuDNN、TensorRT 和 NVRTC 由使用者自行安装。仓库不重新分发 CUDA、cuDNN、TensorRT 厂商运行库：只发布托管源码、托管包和项目自有桥接包。

## 首版重点

- src 按 CUDA、TensorRT、运行时、显存和共享模块归类。
- 覆盖 engine 构建、执行上下文、binding、动态 shape、分配器、日志、性能分析、进度监视、stream、event、CUDA Graph 和 CUDA RTC。
- 运行库包角色明确：windows_split_package_roles=bridge 的 .Bridge 包只包含项目桥接库。
- NuGet 包固定使用 nuget/logo.jpg，并嵌入当前英文 README。
- 许可证为 Apache-2.0。

## 本次更新：4.0.0-preview.1

- 确立 `JYPPX.TensorRtSharp` 与 `JYPPX.CudaSharp` 顶层命名空间，并提供首个预览版托管 API。
- 提供托管包和按版本区分的 Windows/Linux bridge-only 包，CUDA、cuDNN、TensorRT 仍由用户自行安装。
- 补充演示模型的可复现获取与 ONNX 转换记录，模型二进制继续保存在 Git 仓库外。

查看 [4.0.0-preview.1 详细说明](docs/releases/4.0.0-preview.1.md)，或浏览 [全部版本列表](docs/releases/README.md)。

## 30 秒开始

在目标机器安装匹配的 NVIDIA 运行库，然后创建控制台项目并添加托管包和对应 bridge 包：

~~~powershell
dotnet new console -n TrtQuickstart
cd TrtQuickstart
dotnet add package JYPPX.TensorRT.CSharp.API --version "4.0.0-*"
dotnet add package JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge --version "4.0.0-*"
~~~

`4.0.0-*` 只跟随当前维护的预览线，不会误选 API 不兼容的历史 4.x 包（例如 `4.0.6170`）。请根据目标机器安装的 RID 和 NVIDIA 运行时矩阵替换 Bridge 包 ID。

程序创建 runtime、加载 engine、绑定输入输出并执行推理。bridge 包不包含 CUDA、cuDNN 或 TensorRT。请先阅读 [推理绑定教程](docs/articles/zh-cn/inference-bindings-tutorial.md) 和 [Windows 安装排错](docs/articles/zh-cn/windows-installation-and-troubleshooting-guide.md)。

## 包结构

| 包 | 内容 |
| --- | --- |
| JYPPX.TensorRT.CSharp.API | TensorRT/CUDA 托管 API |
| JYPPX.TensorRT.CSharp.API.Bridge.* | 仅项目自有原生桥接库 |

位于 `samples/ComputerVision/01.Classification` 的 `Classification` 和位于 `applications/YoloVision` 的 `YoloVision` 都是可运行案例。它们使用已发布的 4 系列托管包，但自身不进入任何公开包源或 Release 资产。

## 系列案例

| 系列 | 项目 | 主要内容 |
| --- | --- | --- |
| CUDA | `Cuda/01.RuntimeCompilation` | CUDA RTC 编译、模块加载、kernel 启动与结果回读 |
| 推理基础 | `Inference/01.Bindings`、`Inference/02.DynamicShapes` | binding、显存归属和动态 profile |
| 性能 | `Performance/01.MultiStream` | CUDA stream、event 与跨流顺序 |
| 计算机视觉 | `Classification` | 图像预处理、Top-K、JSON 和识别结果图 |
| 完整应用 | `YoloVision`、`OnnxToEngine`、`TensorRtExec` | 多步骤工作流与高级用法 |

可运行命令和对应文章见 [系列案例目录](samples/README.md) 与 [完整应用目录](applications/README.md)。

CUDA RTC 路线图：[English](docs/articles/en/cuda-runtime-compilation-roadmap.md) | [简体中文](docs/articles/zh-cn/cuda-runtime-compilation-roadmap.md) | [技术文章](docs/articles/zh-cn/cuda-runtime-compilation-technical-article.md)

## 公开包与 Release 资产

首个 4 系列公开预览版已经可用。NuGet 包 README 使用根目录英文 README，包图标固定为 <code>nuget/logo.jpg</code>，核心托管包使用 Apache-2.0 SPDX 许可证表达式。

| 包 | 版本 | NuGet.org | GitHub Packages | 用途 |
| --- | --- | --- | --- | --- |
| <code>JYPPX.TensorRT.CSharp.API</code> | [![版本](https://img.shields.io/nuget/vpre/JYPPX.TensorRT.CSharp.API.svg?label=version)](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/) | [包页面](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/) | [包源](https://github.com/users/guojin-yan/packages/nuget/package/jyppx.tensorrt.csharp.api) | 核心 TensorRT/CUDA 托管 API |

| 发布渠道 | 链接 | 资产 |
| --- | --- | --- |
| GitHub Release | [TensorRtSharp4.0 Releases](https://github.com/guojin-yan/TensorRT-CSharp-API/releases) | 源码压缩包、核心托管 <code>.nupkg</code> 和项目自有 Bridge <code>.nupkg</code> 包 |
| GitHub Packages | [NuGet 包源](https://github.com/users/guojin-yan/packages?repo_name=TensorRT-CSharp-API) | 核心托管包以及已发布的项目自有 Bridge 矩阵 |
| NuGet.org | [NuGet 包页面](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/) | 核心托管包；项目自有 Bridge 包以官方推送完成后的页面状态为准 |

### Bridge 包矩阵

下面列出每个已发布 bridge 包及其匹配的用户自装 CUDA、cuDNN、TensorRT 版本。`.Bridge` 包只包含 `jyppxtrtbridge`，不会携带 NVIDIA 厂商运行库。`published-preview.1` 表示当前已在 GitHub Packages、GitHub Release 和 NuGet.org 提供。

| 包 ID | Runtime key | CUDA | cuDNN | TensorRT | 发布状态 |
| --- | --- | --- | --- | --- | --- |
| `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt8.6.cuda11.8.cudnn8.9.Bridge`<br>[![版本](https://img.shields.io/nuget/vpre/JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt8.6.cuda11.8.cudnn8.9.Bridge.svg?label=version)](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt8.6.cuda11.8.cudnn8.9.Bridge/) | `win-x64-trt8.6-cuda11.8-cudnn8.9` | 11.8 | 8.9 | 8.6 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt8.6.cuda12.1.cudnn8.9.Bridge`<br>[![版本](https://img.shields.io/nuget/vpre/JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt8.6.cuda12.1.cudnn8.9.Bridge.svg?label=version)](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt8.6.cuda12.1.cudnn8.9.Bridge/) | `win-x64-trt8.6-cuda12.1-cudnn8.9` | 12.1 | 8.9 | 8.6 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda11.8.cudnn8.9.Bridge`<br>[![版本](https://img.shields.io/nuget/vpre/JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda11.8.cudnn8.9.Bridge.svg?label=version)](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda11.8.cudnn8.9.Bridge/) | `win-x64-trt10.11-cuda11.8-cudnn8.9` | 11.8 | 8.9 | 10.11 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge`<br>[![版本](https://img.shields.io/nuget/vpre/JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge.svg?label=version)](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge/) | `win-x64-trt10.11-cuda12.9-cudnn9.22` | 12.9 | 9.22 | 10.11 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda12.9.cudnn9.22.Bridge`<br>[![版本](https://img.shields.io/nuget/vpre/JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda12.9.cudnn9.22.Bridge.svg?label=version)](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda12.9.cudnn9.22.Bridge/) | `win-x64-trt11.0-cuda12.9-cudnn9.22` | 12.9 | 9.22 | 11.0 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda13.2.cudnn9.22.Bridge`<br>[![版本](https://img.shields.io/nuget/vpre/JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda13.2.cudnn9.22.Bridge.svg?label=version)](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda13.2.cudnn9.22.Bridge/) | `win-x64-trt11.0-cuda13.2-cudnn9.22` | 13.2 | 9.22 | 11.0 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu20.04.trt8.6.cuda11.8.cudnn8.9.Bridge`<br>[![版本](https://img.shields.io/nuget/vpre/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu20.04.trt8.6.cuda11.8.cudnn8.9.Bridge.svg?label=version)](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu20.04.trt8.6.cuda11.8.cudnn8.9.Bridge/) | `linux-x64-ubuntu20.04-trt8.6-cuda11.8-cudnn8.9` | 11.8 | 8.9 | 8.6 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu20.04.trt8.6.cuda12.1.cudnn8.9.Bridge`<br>[![版本](https://img.shields.io/nuget/vpre/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu20.04.trt8.6.cuda12.1.cudnn8.9.Bridge.svg?label=version)](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu20.04.trt8.6.cuda12.1.cudnn8.9.Bridge/) | `linux-x64-ubuntu20.04-trt8.6-cuda12.1-cudnn8.9` | 12.1 | 8.9 | 8.6 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu20.04.trt10.11.cuda11.8.cudnn8.9.Bridge`<br>[![版本](https://img.shields.io/nuget/vpre/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu20.04.trt10.11.cuda11.8.cudnn8.9.Bridge.svg?label=version)](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu20.04.trt10.11.cuda11.8.cudnn8.9.Bridge/) | `linux-x64-ubuntu20.04-trt10.11-cuda11.8-cudnn8.9` | 11.8 | 8.9 | 10.11 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt8.6.cuda11.8.cudnn8.9.Bridge`<br>[![版本](https://img.shields.io/nuget/vpre/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt8.6.cuda11.8.cudnn8.9.Bridge.svg?label=version)](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt8.6.cuda11.8.cudnn8.9.Bridge/) | `linux-x64-ubuntu22.04-trt8.6-cuda11.8-cudnn8.9` | 11.8 | 8.9 | 8.6 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt8.6.cuda12.1.cudnn8.9.Bridge`<br>[![版本](https://img.shields.io/nuget/vpre/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt8.6.cuda12.1.cudnn8.9.Bridge.svg?label=version)](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt8.6.cuda12.1.cudnn8.9.Bridge/) | `linux-x64-ubuntu22.04-trt8.6-cuda12.1-cudnn8.9` | 12.1 | 8.9 | 8.6 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt10.11.cuda11.8.cudnn8.9.Bridge`<br>[![版本](https://img.shields.io/nuget/vpre/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt10.11.cuda11.8.cudnn8.9.Bridge.svg?label=version)](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt10.11.cuda11.8.cudnn8.9.Bridge/) | `linux-x64-ubuntu22.04-trt10.11-cuda11.8-cudnn8.9` | 11.8 | 8.9 | 10.11 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt10.11.cuda12.9.cudnn9.22.Bridge`<br>[![版本](https://img.shields.io/nuget/vpre/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt10.11.cuda12.9.cudnn9.22.Bridge.svg?label=version)](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt10.11.cuda12.9.cudnn9.22.Bridge/) | `linux-x64-ubuntu22.04-trt10.11-cuda12.9-cudnn9.22` | 12.9 | 9.22 | 10.11 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt11.0.cuda12.9.cudnn9.22.Bridge`<br>[![版本](https://img.shields.io/nuget/vpre/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt11.0.cuda12.9.cudnn9.22.Bridge.svg?label=version)](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt11.0.cuda12.9.cudnn9.22.Bridge/) | `linux-x64-ubuntu22.04-trt11.0-cuda12.9-cudnn9.22` | 12.9 | 9.22 | 11.0 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt11.0.cuda13.2.cudnn9.22.Bridge`<br>[![版本](https://img.shields.io/nuget/vpre/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt11.0.cuda13.2.cudnn9.22.Bridge.svg?label=version)](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt11.0.cuda13.2.cudnn9.22.Bridge/) | `linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22` | 13.2 | 9.22 | 11.0 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu24.04.trt10.11.cuda12.9.cudnn9.22.Bridge`<br>[![版本](https://img.shields.io/nuget/vpre/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu24.04.trt10.11.cuda12.9.cudnn9.22.Bridge.svg?label=version)](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu24.04.trt10.11.cuda12.9.cudnn9.22.Bridge/) | `linux-x64-ubuntu24.04-trt10.11-cuda12.9-cudnn9.22` | 12.9 | 9.22 | 10.11 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu24.04.trt11.0.cuda12.9.cudnn9.22.Bridge`<br>[![版本](https://img.shields.io/nuget/vpre/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu24.04.trt11.0.cuda12.9.cudnn9.22.Bridge.svg?label=version)](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu24.04.trt11.0.cuda12.9.cudnn9.22.Bridge/) | `linux-x64-ubuntu24.04-trt11.0-cuda12.9-cudnn9.22` | 12.9 | 9.22 | 11.0 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu24.04.trt11.0.cuda13.2.cudnn9.22.Bridge`<br>[![版本](https://img.shields.io/nuget/vpre/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu24.04.trt11.0.cuda13.2.cudnn9.22.Bridge.svg?label=version)](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu24.04.trt11.0.cuda13.2.cudnn9.22.Bridge/) | `linux-x64-ubuntu24.04-trt11.0-cuda13.2-cudnn9.22` | 13.2 | 9.22 | 11.0 | published-preview.1 |

## 模型获取与 ONNX 转换

演示模型统一暂存于仓库外的 models 目录，不上传 GitHub，也不放入 NuGet。每篇完整技术文章都要写明官方获取 URL、固定 revision、许可证、转换命令、输入输出契约和 SHA256。

| 演示 | 获取和转换方式 |
| --- | --- |
| MNIST | 使用项目生成的数字图像，运行示例 PyTorch/ONNX 脚本导出，再用 trtexec 构建 TensorRT engine。 |
| ResNet18 | 获取 torchvision 官方权重，按 NCHW 224x224 和 ImageNet 归一化执行 torch.onnx.export。 |
| YOLOv8n 检测/分类/分割/姿态/OBB | 获取 Ultralytics 官方 checkpoint，使用固定版本导出命令并校验输出名称和形状。 |
| YOLOv10n | 获取 THU-MIG 官方 checkpoint，使用仓库 reference 脚本导出并保留端到端输出契约。 |
| YOLOX-S | 获取 Megvii 官方 checkpoint，按固定 YOLOX/ONNX 流程转换并校验 decode 元数据。 |
| LRASPP MobileNetV3 Large | 获取 torchvision v0.25.0 官方权重，使用 mean/std 导出到 [1,21,320,320] 并比较逐像素 argmax。 |

详见 [演示模型清单](samples/assets/demo-model-inventory.json)、[模型获取与 ONNX 转换文章](docs/articles/zh-cn/demo-model-acquisition-and-onnx-conversion.md) 和 eng/Sync-DemoOnnxModels.ps1。后续模型会迁移到 ModelZoo。

## 文档入口

- [英文文档](docs/index.md)
- [中文文章目录与完整性标准](docs/articles/zh-cn/README.md)
- [项目概览](docs/articles/zh-cn/project-overview.md)
- [源码组织](docs/articles/zh-cn/source-organization.md)
- [模型获取与 ONNX 转换](docs/articles/zh-cn/demo-model-acquisition-and-onnx-conversion.md)
- [推理绑定教程](docs/articles/zh-cn/inference-bindings-tutorial.md)
- [TensorRtExec GUI 使用](docs/articles/zh-cn/tensorrtexec-gui-user-guide.md)
- [候选发布门禁](docs/articles/zh-cn/release-candidate-gate.md)
- [发布证明与发布后验证指南](docs/articles/zh-cn/release-proof-and-post-publish-verification-guide.md)
- [Owner 输入交叉哈希审计](docs/articles/zh-cn/owner-input-cross-hash-audit.md)
- [TensorRtExec 报告边界](docs/articles/zh-cn/tensorrtexec-report-proof-boundary.md)
- [ONNX 转 engine 报告边界](docs/articles/zh-cn/onnx-to-engine-trtexec-proof-boundary.md)
- [YOLOVision 资产证据指南](docs/articles/zh-cn/yolovision-owner-asset-evidence-guide.md)
- [回调和分配器安全门](docs/articles/zh-cn/callback-allocator-listener-readonly-safety-gates.md)
- [API readiness audit](artifacts/interface-coverage/release-api-readiness-audit.json)
- [YOLOVision 模型矩阵](applications/YoloVision/yolo-model-matrix.json)
- [TensorRtExec 功能矩阵](applications/TensorRtExec/tensor-rt-exec-feature-matrix.json)
- [ONNX 转换 parity 矩阵](applications/OnnxToEngine/trtexec-parity-matrix.json)
- [article-roadmap-30plus](docs/articles/zh-cn/publishing/article-roadmap-30plus.md)

## 源码构建

~~~powershell
dotnet restore TensorRtSharp.sln
dotnet build TensorRtSharp.sln -c Release
dotnet test tests/JYPPX.ProjectQuality.Tests/JYPPX.ProjectQuality.Tests.csproj -c Release --no-restore
~~~

后续发布时将 JYPPXPackageVersion 设置为批准的 4 系列版本。上传前必须检查每个 nupkg：包含 README 和 logo.jpg，且不包含 CUDA、cuDNN、TensorRT 厂商二进制。

## 发布与 Action 规则

所有 workflow 仅手工触发，用于节省 Action 额度。grape-yan 仓库只做验证，不发布任何包。先在本地完成 restore、build、定向测试、包内容检查和干净消费者验证，只有 Owner 批准后才触发远程验证或正式发布。

NuGet 发布需要核心包权限，以及每个项目自有 `.Bridge` ID 的按包 push 权限。`JYPPX.TensorRT.CSharp.API.YoloVision` 和 `JYPPX.TensorRT.CSharp.API.Classification` 是仅供 samples 使用的 ID，严禁上传。nuget.org `403` 表示授权失败，不应重复上传。

## 目录结构

- src：按模块归类的托管接口。
- native：原生桥接和 ABI 导出。
- samples：可运行 C# 演示和模型元数据。
- applications/TensorRtExec：桌面 engine 构建与运行工具。
- pack：托管包和 bridge-only 包定义。
- docs：DocFX 站点和技术文章。
- eng：构建、获取、校验和发布脚本。

## 许可证

本项目使用 Apache-2.0，详见 [LICENSE](LICENSE)。

## 问题反馈

请附上包版本、CUDA/cuDNN/TensorRT 版本、GPU、操作系统和失败命令。不要上传有再分发限制的模型权重或 NVIDIA 运行库。
