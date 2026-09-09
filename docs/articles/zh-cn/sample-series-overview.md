# TensorRtSharp4.0 系列案例学习路线

[English](../en/sample-series-overview.md) | 简体中文

案例源码入口见 [`samples/README.zh-CN.md`](https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/samples/README.zh-CN.md)，完整应用入口见 [`applications/README.zh-CN.md`](https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/applications/README.zh-CN.md)。

TensorRtSharp4.0 的案例分成“小型能力案例”和“完整应用”两层。`samples` 中的项目用于学习一个明确能力，
`applications` 中的项目用于展示模型准备、参数配置、执行、结果校验和可视化组成的完整工作流。

所有可运行案例都是应用程序，均设置为 `IsPackable=false`。它们引用已发布的
`JYPPX.TensorRT.CSharp.API` 4 系列包，不发布 Classification、YoloVision 等案例专用 NuGet 包。
CUDA、cuDNN、TensorRT 和 NVRTC 由用户安装；运行时还需要选择与本机环境匹配的项目自有 `.Bridge` 包。

## 依赖安装

新建仓库外项目时，使用已核验的精确正式版本：

```powershell
dotnet add package JYPPX.TensorRT.CSharp.API --version "4.0.0"
dotnet add package JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge --version "4.0.0"
```

第二个包只是一种环境示例。实际包 ID 必须匹配目标机器的操作系统、CUDA、cuDNN 和 TensorRT 版本。
`.Bridge` 包只包含 `jyppxtrtbridge`，不会安装 NVIDIA 厂商运行库。

视觉案例使用项目作者维护的 [OpenCV-CSharp-API](https://github.com/guojin-yan/OpenCV-CSharp-API)：

```powershell
dotnet add package JYPPX.OpenCV.CSharp.API --prerelease
dotnet add package JYPPX.OpenCV.runtime.win-x64 --prerelease
```

仓库内项目通过 `build/JYPPX.PublicSamplePackages.props` 和 `build/JYPPX.OpenCvSamplePackages.props`
集中维护浮动范围，避免每个 `.csproj` 重复版本。命令行、文章和用户代码不依赖源码 `ProjectReference`。

## 第一部分：CUDA 基础

| 顺序 | 案例 | 学习目标 | 配套文章 |
| --- | --- | --- | --- |
| 01 | `Cuda/01.RuntimeCompilation` | 编译 CUDA C 源码、加载 module、启动 kernel、读取结果 | [CUDA RTC 完整文章](cuda-runtime-compilation-technical-article.md) |

这一部分适合先确认 CUDA 驱动、NVRTC、context、module 和 device memory 的最小闭环。

## 第二部分：TensorRT 推理基础

| 顺序 | 案例 | 学习目标 | 配套文章 |
| --- | --- | --- | --- |
| 01 | `Inference/01.Bindings` | 输入输出 tensor、host/device ownership、enqueue 和回读 | [推理绑定教程](inference-bindings-tutorial.md) |
| 02 | `Inference/02.DynamicShapes` | optimization profile、动态 shape 和运行时尺寸检查 | [动态 Shape 教程](dynamic-shape-optimization-profile-tutorial.md) |

建议按顺序运行。第二个案例建立在第一个案例的资源生命周期与 binding 概念上。

## 第三部分：性能与并发

| 顺序 | 案例 | 学习目标 | 配套文章 |
| --- | --- | --- | --- |
| 01 | `Performance/01.MultiStream` | 多 stream、event、跨流依赖和计时 | [CUDA 多流教程](cuda-stream-event-multistream-tutorial.md) |

性能文章必须同时记录模型或 kernel、输入、预热轮次、测量轮次、同步点和硬件环境，不能只展示一个耗时数字。

## 第四部分：计算机视觉

| 顺序 | 案例 | 学习目标 | 配套文章 |
| --- | --- | --- | --- |
| 01 | `ComputerVision/01.Classification` | OpenCV 图片解码、ImageNet 归一化、Top-K、JSON 和结果图 | [ResNet18 分类文章](classification-real-asset-walkthrough.md) |

视觉文章必须写明模型获取、许可证、固定 revision、ONNX 转换命令、输入输出合同和 SHA256。
结果部分至少包含一次真实程序运行页面和一张将识别结果绘制回原图的图片。

## 第五部分：完整应用与高级用法

| 应用 | 主要能力 | 入口 |
| --- | --- | --- |
| `YoloVision` | 检测、分类、实例分割、OBB、姿态、语义分割、报告与可视化 | [全任务总览](yolovision-all-task-overview.md) |
| `OnnxToEngine` | ONNX 解析、构建配置、动态 profile、engine 序列化 | [ONNX 转 Engine](onnx-to-engine-quickstart.md) |
| `TensorRtExec` | CLI 与 WinForms 构建/执行工具、报告和高级运行参数 | [TensorRtExec GUI](tensorrtexec-gui-user-guide.md) |

完整应用位于 `applications`。YoloVision 使用公开 TensorRT 4 系列包及 OpenCV-CSharp-API；
OnnxToEngine 和 TensorRtExec 也通过公共 TensorRT 包取得核心 API。两者共用的 Tools 实现由
`applications/_shared/JYPPX.TensorRtSharp.ApplicationTools` 链接并编译，不发布案例或 Tools NuGet 包，
也不引用核心 CUDA/TensorRT 源码项目。

## 模型与图片约定

转换后的 ONNX 暂存在源码仓库同级的 `models` 目录，不上传 GitHub，不进入 NuGet 或 Release。
文章使用 `$repoRoot`、`$workspaceRoot`、`$modelRoot` 等变量组织路径，不写某台电脑的盘符和用户名。

模型文章的完整流程固定为：

1. 介绍项目、依赖职责和目标任务。
2. 获取权重、许可证与测试图片并校验哈希。
3. 转换 ONNX，记录工具版本、参数、输入输出和 SHA256。
4. 安装公开 NuGet 包，准备匹配的用户运行环境。
5. 编译并运行案例，保存结构化输出和真实 stdout。
6. 使用独立参考验证输出，并执行至少一个受控负例。
7. 将分类、检测、分割、OBB 或姿态结果绘制回原图。
8. 在文章中展示结果图和真实程序运行页面。

本机缺少对应 CUDA/TensorRT/OpenCV 原生环境时，应在文章中明确标为环境限制，不把预检查或模拟输出写成运行成功。
