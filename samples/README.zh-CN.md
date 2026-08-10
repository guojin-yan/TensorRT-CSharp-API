# 示例

[English](README.md) | 简体中文

`samples/` 只保存小而专注、可直接运行的学习案例。目录按照“功能模块/编号案例”组织，编号用于固定学习顺序；完整工具和大型视觉工作流放在 [`applications/`](../applications/README.zh-CN.md)。

`_shared` 是案例共用源码，`assets` 是轻量清单和证据元数据。包消费者夹具、独立参考程序和发布验证输入统一放在 `tests/fixtures`，不再混入用户案例目录。

## 案例系列

| 模块 | 案例 | 主要内容 | 配套文章 |
| --- | --- | --- | --- |
| `Cuda` | `Cuda/01.RuntimeCompilation` | CUDA RTC 源码编译、模块加载、类型化启动和结果读回 | [CUDA RTC 技术文章](../docs/articles/zh-cn/cuda-runtime-compilation-technical-article.md) |
| `Inference` | `Inference/01.Bindings` | TensorRT 输入输出、主机/设备内存所有权与 binding | [推理绑定教程](../docs/articles/zh-cn/inference-bindings-tutorial.md) |
| `Inference` | `Inference/02.DynamicShapes` | 显式优化配置和动态 Shape 推理 | [动态 Shape 教程](../docs/articles/zh-cn/dynamic-shape-optimization-profile-tutorial.md) |
| `Inference` | `Inference/03.OnnxBuildAndRun` | 公开包解析 ONNX、构建 engine、单次推理和结构化 JSON | [示例说明](Inference/03.OnnxBuildAndRun/README.zh-CN.md) |
| `Inference` | `Inference/04.RefittedPlan` | ONNX initializer refit、plan 持久化、磁盘重载和输出验证 | [示例说明](Inference/04.RefittedPlan/README.zh-CN.md) |
| `Diagnostics` | `Diagnostics/01.CallbackLifecycle` | Logger、ProgressMonitor、Profiler、DebugListener 的所有权和解除顺序 | [示例说明](Diagnostics/01.CallbackLifecycle/README.zh-CN.md) |
| `Performance` | `Performance/01.MultiStream` | CUDA Stream、Event 与跨流同步 | [多流与 Event 教程](../docs/articles/zh-cn/cuda-stream-event-multistream-tutorial.md) |
| `ComputerVision` | `ComputerVision/01.Classification` | 图像预处理、TensorRT 分类、Top-K、JSON 和标注结果图 | [真实分类模型实战](../docs/articles/zh-cn/classification-real-asset-walkthrough.md) |

大型案例包括 [`YoloVision`](../applications/YoloVision/README.zh-CN.md)、[`OnnxToEngine`](../applications/OnnxToEngine/README.zh-CN.md) 和 [`TensorRtExec`](../applications/TensorRtExec/README.md)。

## NuGet 与运行库边界

所有可运行示例通过 [`build/JYPPX.PublicSamplePackages.props`](../build/JYPPX.PublicSamplePackages.props) 消费已发布的 4 系列托管包，不直接引用 `src` 项目，也不在每个项目中写死版本号。案例项目均设置 `IsPackable=false`，不会发布 `Classification` 或 `YoloVision` 示例包。

新建外部项目时，安装托管 API 和一个与目标系统、TensorRT、CUDA、cuDNN 版本匹配的 Bridge 包：

```powershell
dotnet add package JYPPX.TensorRT.CSharp.API --version "4.0.0-*"
dotnet add package JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge --version "4.0.0-*"
```

CUDA、cuDNN、TensorRT 和 NVRTC 由用户自行安装，Bridge 包不携带 NVIDIA 运行库。

## 模型与图片

ONNX、权重、标签、预处理张量和临时运行产物统一暂存在源码仓库同级的 `<workspace-root>/models`，不上传 GitHub。每个使用深度学习模型的案例和文章必须记录：

- 上游项目、模型名称、版本或提交、许可证和获取地址；
- 权重到 ONNX 的完整转换命令、工具版本和 opset；
- 输入输出名称、Shape、布局、数据类型和动态维度；
- resize/crop、RGB/BGR、scale、mean/std 等预处理；
- 模型、标签、输入和结果产物的 SHA256；
- TensorRT 真实运行命令、终端输出截图和任务结果图。

文章和源码使用仓库相对路径或 `<model>`、`<image>` 等占位符，不写开发者机器上的盘符绝对路径。`Classification` 和 `YoloVision` 使用项目自有的 `JYPPX.OpenCV.CSharp.API` 读取 JPEG/PNG，并为 BMP/PPM 保留托管回退路径。

## 运行

从仓库根目录执行帮助命令不需要 CUDA 或 TensorRT：

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

运行真实模型时，先按照配套文章获取并转换模型，再传入 `--model`、`--labels`、`--image` 或 `--input-data` 以及准确的布局和输出元数据。仅构建成功、预检查通过或生成 SVG/PNG，都不能代替真实 TensorRT 推理结果。

## 文章与结果规范

每个正式案例文章应按以下顺序编写：项目背景与依赖、模型获取、ONNX 转换、环境准备、关键代码、完整命令、原始控制台结果、结果解释、结果图或软件界面截图、常见问题和证据边界。图像任务必须把框、类别、置信度、掩码、关键点或旋转框绘制到原图上；桌面程序还要提供真实软件运行页面截图。

证据按 `precheck`、`build-only`、`synthetic-input-runtime`、`real-model-runtime` 和 `package-consumer-runtime` 分级。缺少本机 CUDA/TensorRT 环境时应明确记录环境阻塞，不得把预检查或截图描述成真实运行成功。

## 新增案例

1. 选择模块和下一个编号目录。
2. 保持项目可执行且 `IsPackable=false`。
3. 使用已发布 NuGet，不新增到核心源码项目的 `ProjectReference`。
4. 提供确定性输入、结构化报告、真实结果图和双语 README。
5. 编写配套技术文章，并加入解决方案和案例索引。
6. 本地构建和运行通过后，才考虑修改 Action 或执行发布。
