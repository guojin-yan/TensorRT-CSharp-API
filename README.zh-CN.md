<p align="center">
  <img src="https://socialify.git.ci/guojin-yan/TensorRT-CSharp-API/image?description=1&descriptionEditable=TensorRT%20and%20CUDA%20bindings%20for%20C%23%20and%20.NET&forks=1&issues=1&name=1&owner=1&pattern=Circuit%20Board&pulls=1&stargazers=1&theme=Light" alt="TensorRtSharp4.0" width="100%" />
</p>

<h1 align="center">TensorRtSharp4.0</h1>

<p align="center">
  面向 C# 与 .NET 的 TensorRT、CUDA 托管接口、项目自有桥接包、可运行视觉示例和 TensorRtExec 桌面工具。
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache-2.0 许可证" /></a>
  <a href="https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/"><img src="https://img.shields.io/nuget/vpre/JYPPX.TensorRT.CSharp.API.svg" alt="NuGet 预览版本" /></a>
  <a href="https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/"><img src="https://img.shields.io/nuget/dt/JYPPX.TensorRT.CSharp.API.svg" alt="NuGet 下载量" /></a>
  <a href="https://github.com/users/guojin-yan/packages/nuget/package/jyppx.tensorrt.csharp.api"><img src="https://img.shields.io/badge/GitHub%20Packages-package%20feed-24292f" alt="GitHub Packages 源" /></a>
  <a href="https://github.com/guojin-yan/TensorRT-CSharp-API/releases"><img src="https://img.shields.io/github/v/release/guojin-yan/TensorRT-CSharp-API?include_prereleases&label=Release" alt="GitHub Release" /></a>
  <a href="https://dotnet.microsoft.com/"><img src="https://img.shields.io/badge/.NET-Framework%204.6--4.8.1%20%7C%20Core%203.1%20%7C%205--10-512BD4" alt="支持的 .NET 版本" /></a>
  <a href="https://developer.nvidia.com/tensorrt"><img src="https://img.shields.io/badge/TensorRT-%E7%94%A8%E6%88%B7%E8%87%AA%E8%A1%8C%E5%AE%89%E8%A3%85-76B900" alt="TensorRT 由用户安装" /></a>
</p>

<p align="center">
  <a href="https://github.com/guojin-yan/TensorRT-CSharp-API/actions/workflows/ci-validation.yml"><img src="https://github.com/guojin-yan/TensorRT-CSharp-API/actions/workflows/ci-validation.yml/badge.svg?branch=TensorRtSharp4.0" alt="托管代码检查" /></a>
  <a href="https://github.com/guojin-yan/TensorRT-CSharp-API/actions/workflows/release-quality-gate.yml"><img src="https://github.com/guojin-yan/TensorRT-CSharp-API/actions/workflows/release-quality-gate.yml/badge.svg?branch=TensorRtSharp4.0" alt="发布质量门禁" /></a>
</p>

<p align="center"><a href="README.md">English</a> | <strong>简体中文</strong></p>

# TensorRtSharp4.0

[![构建](https://github.com/guojin-yan/TensorRT-CSharp-API/actions/workflows/ci-validation.yml/badge.svg)](https://github.com/guojin-yan/TensorRT-CSharp-API/actions/workflows/ci-validation.yml)
[![文档](https://img.shields.io/badge/docs-DocFX-2f80ed)](https://guojin-yan.github.io/TensorRT-CSharp-API/)
[![许可证](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

TensorRtSharp4.0 是面向 .NET 的 TensorRT / CUDA 桥接项目，包含推理执行、CUDA 运行时编译、显存、流与事件、回调、分配器以及 TensorRtExec 桌面工具。首个公开候选版本按 4.0.0-preview.1 规划；它是源码和包的候选版本，不代表所有 TensorRT 功能都已经完成真实运行时验证。

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

## 30 秒开始

在目标机器安装匹配的 NVIDIA 运行库，然后创建控制台项目并添加托管包和对应 bridge 包：

~~~powershell
dotnet new console -n TrtQuickstart
cd TrtQuickstart
dotnet add package JYPPX.TensorRT.CSharp.API --version 4.0.0-preview.1
dotnet add package JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge --version 4.0.0-preview.1
~~~

程序创建 runtime、加载 engine、绑定输入输出并执行推理。bridge 包不包含 CUDA、cuDNN 或 TensorRT。请先阅读 [推理绑定教程](docs/articles/zh-cn/inference-bindings-tutorial.md) 和 [Windows 安装排错](docs/articles/zh-cn/windows-installation-and-troubleshooting-guide.md)。

## 包结构

| 包 | 内容 |
| --- | --- |
| JYPPX.TensorRT.CSharp.API | TensorRT/CUDA 托管 API |
| JYPPX.TensorRT.CSharp.API.Bridge.* | 仅项目自有原生桥接库 |

`samples/YoloVision` 和 `samples/Classification` 是可运行演示。它们的项目文件和本地 package-consumer 验证脚本保留在源码中用于开发检查，但对应的示例包 ID 明确不进入任何公开包源和 Release 资产。

## 公开包与 Release 资产

首个公开候选版本为 <code>4.0.0-preview.1</code>。NuGet 包 README 使用根目录英文 README，包图标固定为 <code>nuget/logo.jpg</code>，核心托管包使用 Apache-2.0 SPDX 许可证表达式。

| 包 | 版本 | NuGet.org | GitHub Packages | 用途 |
| --- | --- | --- | --- | --- |
| <code>JYPPX.TensorRT.CSharp.API</code> | [![版本](https://img.shields.io/nuget/vpre/JYPPX.TensorRT.CSharp.API.svg?label=4.0.0-preview.1)](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/) | [包页面](https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/) | [包源](https://github.com/users/guojin-yan/packages/nuget/package/jyppx.tensorrt.csharp.api) | 核心 TensorRT/CUDA 托管 API |

| 发布渠道 | 链接 | 资产 |
| --- | --- | --- |
| GitHub Release | [TensorRtSharp4.0 Releases](https://github.com/guojin-yan/TensorRT-CSharp-API/releases) | 源码压缩包、核心托管 <code>.nupkg</code> 和项目自有 Bridge <code>.nupkg</code> 包 |
| GitHub Packages | [NuGet 包源](https://github.com/users/guojin-yan/packages?repo_name=TensorRT-CSharp-API) | 核心托管包以及已发布的项目自有 Bridge 矩阵 |

### Bridge 包矩阵

下面列出每个已规划 bridge 包及其匹配的用户自装 CUDA、cuDNN、TensorRT 版本。`.Bridge` 包只包含 `jyppxtrtbridge`，不会携带 NVIDIA 厂商运行库。`published-preview.1` 表示已在 GitHub Packages 和预发行版中提供；`runner-blocked` 表示因缺少兼容的 Windows 自托管 runner 而尚未发布。

| 包 ID | Runtime key | CUDA | cuDNN | TensorRT | 发布状态 |
| --- | --- | --- | --- | --- | --- |
| `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt8.6.cuda11.8.cudnn8.9.Bridge` | `win-x64-trt8.6-cuda11.8-cudnn8.9` | 11.8 | 8.9 | 8.6 | runner-blocked |
| `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt8.6.cuda12.1.cudnn8.9.Bridge` | `win-x64-trt8.6-cuda12.1-cudnn8.9` | 12.1 | 8.9 | 8.6 | runner-blocked |
| `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda11.8.cudnn8.9.Bridge` | `win-x64-trt10.11-cuda11.8-cudnn8.9` | 11.8 | 8.9 | 10.11 | runner-blocked |
| `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge` | `win-x64-trt10.11-cuda12.9-cudnn9.22` | 12.9 | 9.22 | 10.11 | runner-blocked |
| `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda12.9.cudnn9.22.Bridge` | `win-x64-trt11.0-cuda12.9-cudnn9.22` | 12.9 | 9.22 | 11.0 | runner-blocked |
| `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda13.2.cudnn9.22.Bridge` | `win-x64-trt11.0-cuda13.2-cudnn9.22` | 13.2 | 9.22 | 11.0 | runner-blocked |
| `JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu20.04.trt8.6.cuda11.8.cudnn8.9.Bridge` | `linux-x64-ubuntu20.04-trt8.6-cuda11.8-cudnn8.9` | 11.8 | 8.9 | 8.6 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu20.04.trt8.6.cuda12.1.cudnn8.9.Bridge` | `linux-x64-ubuntu20.04-trt8.6-cuda12.1-cudnn8.9` | 12.1 | 8.9 | 8.6 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu20.04.trt10.11.cuda11.8.cudnn8.9.Bridge` | `linux-x64-ubuntu20.04-trt10.11-cuda11.8-cudnn8.9` | 11.8 | 8.9 | 10.11 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt8.6.cuda11.8.cudnn8.9.Bridge` | `linux-x64-ubuntu22.04-trt8.6-cuda11.8-cudnn8.9` | 11.8 | 8.9 | 8.6 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt8.6.cuda12.1.cudnn8.9.Bridge` | `linux-x64-ubuntu22.04-trt8.6-cuda12.1-cudnn8.9` | 12.1 | 8.9 | 8.6 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt10.11.cuda11.8.cudnn8.9.Bridge` | `linux-x64-ubuntu22.04-trt10.11-cuda11.8-cudnn8.9` | 11.8 | 8.9 | 10.11 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt10.11.cuda12.9.cudnn9.22.Bridge` | `linux-x64-ubuntu22.04-trt10.11-cuda12.9-cudnn9.22` | 12.9 | 9.22 | 10.11 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt11.0.cuda12.9.cudnn9.22.Bridge` | `linux-x64-ubuntu22.04-trt11.0-cuda12.9-cudnn9.22` | 12.9 | 9.22 | 11.0 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt11.0.cuda13.2.cudnn9.22.Bridge` | `linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22` | 13.2 | 9.22 | 11.0 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu24.04.trt10.11.cuda12.9.cudnn9.22.Bridge` | `linux-x64-ubuntu24.04-trt10.11-cuda12.9-cudnn9.22` | 12.9 | 9.22 | 10.11 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu24.04.trt11.0.cuda12.9.cudnn9.22.Bridge` | `linux-x64-ubuntu24.04-trt11.0-cuda12.9-cudnn9.22` | 12.9 | 9.22 | 11.0 | published-preview.1 |
| `JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu24.04.trt11.0.cuda13.2.cudnn9.22.Bridge` | `linux-x64-ubuntu24.04-trt11.0-cuda13.2-cudnn9.22` | 13.2 | 9.22 | 11.0 | published-preview.1 |

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
- [发布证明示例文章](docs/articles/zh-cn/release-proof-sample-article-closure.md)
- [Owner 输入看板](docs/articles/zh-cn/release-proof-owner-input-dashboard.md)
- [TensorRtExec 报告边界](docs/articles/zh-cn/tensorrtexec-report-proof-boundary.md)
- [ONNX 转 engine 报告边界](docs/articles/zh-cn/onnx-to-engine-trtexec-proof-boundary.md)
- [YOLOVision 资产证据指南](docs/articles/zh-cn/yolovision-owner-asset-evidence-guide.md)
- [回调和分配器安全门](docs/articles/zh-cn/callback-allocator-listener-readonly-safety-gates.md)
- [API readiness audit](artifacts/interface-coverage/release-api-readiness-audit.json)
- [YOLOVision 模型矩阵](samples/YoloVision/yolo-model-matrix.json)
- [TensorRtExec 功能矩阵](applications/TensorRtExec/tensor-rt-exec-feature-matrix.json)
- [ONNX 转换 parity 矩阵](samples/OnnxToEngine/trtexec-parity-matrix.json)
- [article-roadmap-30plus](docs/articles/zh-cn/article-roadmap-30plus.md)

## 源码构建

~~~powershell
dotnet restore TensorRtSharp.sln
dotnet build TensorRtSharp.sln -c Release
dotnet test tests/JYPPX.ProjectQuality.Tests/JYPPX.ProjectQuality.Tests.csproj -c Release --no-restore
~~~

本地打包时使用 JYPPXPackageVersion=4.0.0-preview.1。上传前检查 nupkg 必须包含 README 和 logo.jpg，不能包含 CUDA、cuDNN、TensorRT 厂商二进制。

## 发布与 Action 规则

所有 workflow 仅手工触发，用于节省 Action 额度。grape-yan 仓库只做候选验证，不发布任何包；本地门禁通过后最多执行一次候选 Action。正式 guojin-yan 仓库在 Owner 明确批准后只执行一次正式发布流程，版本固定为 4.0.0-preview.1。

在正式流程完成前，发布状态保持 blocked，状态为 owner-action-required。clean-consumer-proof-execution-bundle 和 clean-consumer-external-proof-closure-pack 属于 non-proof 的 Owner action，不会运行 runtime smoke，也不是 runtime proof 或 post-publish proof。build-only、local feed、ProjectReference、dry-run、template 和 dashboard 都不能晋级为发布证明或 issue close；FailOnNotProof 会拒绝这些替代物。

NuGet 发布只需要核心 package ID `JYPPX.TensorRT.CSharp.API` 的 push 权限。`JYPPX.TensorRT.CSharp.API.YoloVision` 和 `JYPPX.TensorRT.CSharp.API.Classification` 是仅供 samples 使用的 ID，严禁上传。nuget.org `403` 表示授权失败，不应重复上传。

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
*** End Patch
