# 应用程序

[English](README.md) | 简体中文

`applications/` 保存比单功能示例更完整的用户工作流。每个应用都有独立配置、结构化报告、运行结果和配套文章；小而专注的学习案例位于 [`samples/`](../samples/README.zh-CN.md)。

## 应用目录

| 应用 | 用途 | 发布边界 | 文章入口 |
| --- | --- | --- | --- |
| [`YoloVision`](YoloVision/README.zh-CN.md) | YOLO 系列检测、分类、实例分割、OBB、Pose 和语义分割，包含预处理、输出路由、后处理、JSON 和标注结果图 | 仅可执行应用；消费 TensorRT 托管包和项目自有 OpenCV 包；不发布案例 NuGet | [YOLO 系列总览](../docs/articles/zh-cn/yolovision-sample-overview.md) |
| [`OnnxToEngine`](OnnxToEngine/README.zh-CN.md) | ONNX 解析与 Engine 构建、动态 profile、MNIST 独立参考和转换诊断 | 仅可执行应用；TensorRT/CUDA 核心 API 来自已发布 4 系列包，应用共享 Tools 源码只在本地编译且不打包 | [ONNX 转换快速入门](../docs/articles/zh-cn/onnx-to-engine-quickstart.md) |
| [`TensorRtExec`](TensorRtExec/README.md) | Windows CLI 和 WinForms 工具，覆盖 Engine 构建、加载、refit、binding、推理和报告 | 仅可执行应用；CUDA/TensorRT 运行库由用户安装 | [TensorRtExec 入门](../docs/articles/zh-cn/tensorrtexec-tool-getting-started.md) |

## 公共依赖

应用通过 NuGet 使用当前 4 系列 `JYPPX.TensorRT.CSharp.API`。`YoloVision` 的 JPEG/PNG 解码使用 `JYPPX.OpenCV.CSharp.API`；应用不会把 CUDA、cuDNN、TensorRT 或 NVRTC 打进项目包。

`JYPPX.TensorRT.CSharp.API.YoloVision` 和 `JYPPX.TensorRT.CSharp.API.Classification` 不是计划发布的 NuGet。需要复用时，应在自己的项目中安装托管 API 和匹配平台的 Bridge 包，再根据需要参考应用源码。

`OnnxToEngine` 和 `TensorRtExec` 引用 `applications/_shared/JYPPX.TensorRtSharp.ApplicationTools`。该不可打包的应用共享项目链接仓库内 Tools 实现，但使用已经发布的托管包进行编译；两个应用都不再引用核心 `src/JYPPX.CudaSharp` 或 `src/JYPPX.TensorRtSharp` 项目。

## 模型、运行与文章

模型文件统一暂存在源码仓库同级的 `<workspace-root>/models`，不上传 GitHub。每个模型案例必须在文章中写清上游项目、许可证、获取地址、转换命令、输入输出契约、预处理、后处理和 SHA256。

正式文章必须展示真实运行的原始控制台输出和截图。检测、分割、OBB、Pose、分类等视觉任务还要把结果绘制到原图；WinForms 工作流要提供真实软件页面截图。文章命令使用相对路径和占位符，不出现开发机盘符。

缺少目标 CUDA、cuDNN 或 TensorRT 环境时，应明确记录 `blocked-by-*` 环境状态并说明用户安装要求。`precheck`、`build-only`、GUI 截图和生成报告都不能单独声明真实模型运行成功。

在应用本地还原、构建和目标路径运行通过之前，不触发 Action；包和 Release 发布属于单独的授权流程。
