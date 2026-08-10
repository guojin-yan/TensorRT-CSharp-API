# 开发中更新

本页记录 `4.0.0-preview.1` 发布后的仓库改进。这里的内容尚未形成新的 NuGet、GitHub Package、GitHub Release 或版本标签，不代表已经发布。

## 案例与应用

- `samples` 按 CUDA、Inference、Performance 和 ComputerVision 模块整理，案例项目统一设置为不可打包。
- 新增 `Inference/03.OnnxBuildAndRun`：通过公开包解析 ONNX、构建 engine、执行一次推理并输出结构化 JSON；`--synthetic` 提供确定性的 Identity 模型 smoke，`--help` 保持离线可用。
- `YoloVision`、Classification 和基础案例使用已发布的 `JYPPX.TensorRT.CSharp.API` 4 系列包。
- `OnnxToEngine` 与 `TensorRtExec` 已移除到核心 `src/JYPPX.CudaSharp`、`src/JYPPX.TensorRtSharp` 的项目引用。
- 新增不可打包的 `applications/_shared/JYPPX.TensorRtSharp.ApplicationTools`，链接 Tools 实现并使用公共 TensorRT 包编译。
- 两个工具应用的离线 `--help` 入口已在公共包依赖图下运行通过。

## OpenCV

- Classification 与 YoloVision 使用项目作者发布的 `JYPPX.OpenCV.CSharp.API` 和 Windows x64 runtime 包。
- JPEG/PNG 通过 OpenCV 包解码，BMP/PPM 保留托管回退。
- OpenCV-CSharp-API 源码仓库仅用于核对包 ID 和用法，本项目不修改该仓库。

## 文档

- 用户可见案例和应用 README 已补齐中英文版本与语言切换。
- 新增中英文案例系列学习路线，统一模型获取、ONNX 转换、SHA256、真实终端截图和结果图规范。
- 用户安装命令不写死具体包版本，说明使用当前 4 系列预览包；已发布版本说明继续保留精确 SemVer。
- 修复 README 和 DocFX 导航中的失效或缺失入口。

## 依赖规则

- TensorRT 案例版本集中维护在 `build/JYPPX.PublicSamplePackages.props`。当前必须限制在维护中的 `4.0.0-preview.*` 线，以避免 NuGet 源中的历史不兼容 4.x 包被宽泛规则选中。
- OpenCV 案例使用共享的 5 系列浮动规则，并在当前环境解析到公开预览包。
- CUDA、cuDNN、TensorRT 和 NVRTC 继续由用户安装，Bridge 包不携带 NVIDIA 厂商运行库。

## 本地验证

- `OnnxToEngine`、`TensorRtExec`、ApplicationTools：Release 构建 0 warning、0 error。
- 完整解决方案 Release 构建：0 warning、0 error。
- 案例、应用包消费和双语 README 定向测试：102/102 通过。
- 应用输出中的 TensorRT/CUDA 托管程序集 SHA256 与 NuGet 缓存中的公开包程序集一致。
- Classification、YoloVision 输出中的 OpenCV 托管程序集 SHA256 与公开 OpenCV NuGet 包一致。
- 本机真实运行回归已覆盖 CUDA RTC、Inference Bindings、Dynamic Shapes、MultiStream、Classification、YOLOv8 Detection、YOLOv8 Instance Segmentation、YOLOv8 OBB、LRASPP Semantic Segmentation 和 OnnxToEngine MNIST。
- 真实模型结果摘要：Classification 在 dog 图片上输出 French bulldog Top-1；Detection 输出 4 个 person + 1 个 bus；OBB 输出 13 个旋转框；Semantic 输出 `21x320x320` class-index map；MNIST 输入 7 的预测为 7，置信度 `0.999993`。
- 视觉案例的 JSON、Tensor、mask、class-index map 和 SVG 结果写入本地 `artifacts`，不进入 Git；文章继续使用脱敏的真实终端截图和原图叠加结果图。
- 技术文章严格目录：10/10 通过。
- DocFX：0 warning、0 error。
- managed 包 dry-run 已确认包含根目录 `README.md` 和 `logo.jpg`，不包含 NVIDIA vendor runtime；当前机器的 TensorRT 10.11/CUDA 12.9 bridge-only dry-run 也只包含项目 bridge DLL、README 和 logo。

本机已验证 PowerShell 7.6.4 的 `pwsh`，并保留 KI-005 引入的 `PowerShellHost` 解析与 Windows PowerShell `powershell.exe` 回退。只有明确依赖 PowerShell 7 语法的脚本才强制要求 `pwsh`。CUDA、cuDNN、TensorRT 和 NVRTC 仍由使用者自行安装，Linux 与未列入本机矩阵的 CUDA/TensorRT 组合尚未验证。

## 发布状态

- 未触发 GitHub Actions。
- 未创建或更新版本标签。
- 未发布新的 NuGet、GitHub Package 或 GitHub Release。
- 后续版本号和发布时间由项目维护者在发布前确认。
