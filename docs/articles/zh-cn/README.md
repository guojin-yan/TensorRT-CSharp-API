# TensorRT CSharp API v4.0 中文公开文章

`docs/articles/zh-cn` 既保存面向使用者的中文文章，也保留大量历史工程文档。公开阅读请从七个稳定模块进入：

| 模块 | 定位 | 首批文章 |
| --- | --- | --- |
| [版本发布](01-release/README.md) | 正式版本、升级与发布解读 | [`REL-001`：4.0.0 正式发布](01-release/2026/2026-08-10-tensorrtsharp-4.0.0.md) |
| [系列案例](02-samples/README.md) | 小型可运行案例与连续学习路线 | [`SMP-001`：总览](02-samples/smp-001-sample-series-overview.md)、[`SMP-002`：Bindings](02-samples/smp-002-inference-bindings.md)、[`SMP-003`：Dynamic Shape](02-samples/smp-003-dynamic-shapes.md)、[`SMP-004`：ONNX](02-samples/smp-004-onnx-build-and-run.md)、[`SMP-005`：Refit](02-samples/smp-005-refitted-plan.md)、[`SMP-006`：CUDA RTC](02-samples/smp-006-cuda-runtime-compilation.md)、[`SMP-007`：多流](02-samples/smp-007-cuda-multistream.md)、[`SMP-008`：Callback](02-samples/smp-008-callback-lifecycle.md)、[`SMP-009`：ResNet18](02-samples/smp-009-resnet18-classification.md) |
| [完整应用](03-applications/README.md) | YoloVision、OnnxToEngine、TensorRtExec | [`APP-YV-001`：YoloVision 总览](03-applications/yolovision/app-yv-001-yolovision-overview.md)、[`APP-YV-002`：检测](03-applications/yolovision/app-yv-002-yolov8n-detection.md)、[`APP-YV-003`：分类](03-applications/yolovision/app-yv-003-yolov8n-classification.md)、[`APP-YV-004`：实例分割](03-applications/yolovision/app-yv-004-yolov8n-instance-segmentation.md)、[`APP-YV-005`：Pose](03-applications/yolovision/app-yv-005-yolov8n-pose.md)、[`APP-YV-006`：OBB](03-applications/yolovision/app-yv-006-yolov8n-obb.md)、[`APP-YV-007`：LRASPP](03-applications/yolovision/app-yv-007-lraspp-semantic-segmentation.md)、[`APP-YV-008`：YOLOv10](03-applications/yolovision/app-yv-008-yolov10n-end-to-end.md)、[`APP-YV-009`：YOLOX](03-applications/yolovision/app-yv-009-yolox-s-detection.md)、[`APP-ONNX-001`：OnnxToEngine](03-applications/onnxtoengine/app-onnx-001-onnx-to-engine-getting-started.md)、[`APP-ONNX-002`：MNIST](03-applications/onnxtoengine/app-onnx-002-mnist-runtime-validation.md)、[`APP-ONNX-003`：高级构建](03-applications/onnxtoengine/app-onnx-003-advanced-build-options.md)、[`APP-EXEC-001`：TensorRtExec](03-applications/tensorrtexec/app-exec-001-tensorrtexec-getting-started.md)、[`APP-EXEC-002`：GUI](03-applications/tensorrtexec/app-exec-002-gui-onnx-build.md)、[`APP-EXEC-003`：CLI](03-applications/tensorrtexec/app-exec-003-cli-parameter-guide.md)、[`APP-EXEC-004`：性能与校验](03-applications/tensorrtexec/app-exec-004-performance-and-output-validation.md)、[`APP-EXEC-005`：Refit 与持久化](03-applications/tensorrtexec/app-exec-005-refit-and-engine-persistence.md) |
| [API 使用](04-api/README.md) | 按类和功能介绍 TensorRT/CUDA 接口 | [`MSC-007`：对象模型](04-api/tensorrt/msc-007-tensorrt-object-model.md)、[`API-001`：构建对象](04-api/tensorrt/api-001-builder-network-config-profile.md)、[`API-002`：运行对象](04-api/tensorrt/api-002-runtime-engine-context-bindings.md)、[`API-003`：CUDA 资源](04-api/cuda/api-003-cuda-device-memory-stream-event-graph.md)、[`API-004`：ONNX Parser](04-api/onnx/api-004-onnx-parser-parser-refitter-diagnostics.md)、[`API-005`：回调与诊断](04-api/diagnostics/api-005-callbacks-logger-profiler-progress-debug-listener.md)、[`API-006`：序列化与 Inspector](04-api/engine/api-006-serialization-engine-inspector-error-boundary.md) |
| [安装与运行环境](05-installation/README.md) | 按平台安装、包选择、运行库矩阵和排错 | [`MSC-003`：Windows 安装](05-installation/windows/msc-003-windows-installation.md)、[`INS-001`：Linux 安装](05-installation/linux/ins-001-linux-installation-runtime-validation.md)、[`INS-002`：WSL](05-installation/wsl/ins-002-wsl-gpu-passthrough-runtime-validation.md)、[`INS-003`：容器](05-installation/container/ins-003-container-deployment-runtime-boundary.md)、[`INS-004`：GPU CI](05-installation/ci/ins-004-gpu-ci-runner-validation.md)、[`MSC-004`：包选择](05-installation/packages/msc-004-managed-and-bridge-package-selection.md)、[`MSC-005`：运行库矩阵](05-installation/runtime/msc-005-windows-linux-bridge-matrix.md)、[`MSC-008`：Native 排障](05-installation/troubleshooting/msc-008-cuda-error35-native-load.md) |
| [源码编译](06-source-build/README.md) | CMake、绑定生成、Bridge 编译和本地打包 | [`MSC-006`：C++ Bridge 编译](06-source-build/bridge/msc-006-build-cpp-bridge-from-source.md)、[`BLD-001`：托管编译与本地包验证](06-source-build/managed/bld-001-managed-source-build-test-and-package-validation.md)、[`BLD-002`：绑定生成](06-source-build/bindings/bld-002-binding-generation-diff-audit.md)、[`BLD-003`：CMake 与 Native 调试](06-source-build/native/bld-003-cmake-presets-native-debugging.md)、[`BLD-004`：Runtime Bridge 打包](06-source-build/runtime/bld-004-runtime-bridge-packaging.md) |
| [项目背景与其他主题](07-misc/README.md) | 项目总览、架构背景和模型资产 | [`MSC-001`：项目总览](07-misc/overview/msc-001-what-is-tensorrtsharp4.md)、[`MSC-002`：P/Invoke 与 Bridge](07-misc/architecture/msc-002-beyond-pinvoke.md)、[`MSC-009`：模型资产](07-misc/models/msc-009-model-acquisition-and-onnx-governance.md) |

新模块使用 [`article-index.json`](article-index.json) 记录稳定 ID、canonical 源、版本、状态和外部发布冻结字段。`ready` 只表示正文经过技术校验；只有真实发布后填写 URL、时间、提交和 SHA256，才可以改为 `published` 与 `immutable=true`。

截至 2026-08-14，索引共登记 50 篇 canonical 文章，其中 48 篇为 `ready`、2 篇为 `review`。`04-api` 模块的 7 篇文章已经全部完成仓库正文复核，API-001 至 API-006 的实机批次证据见 [`04-api/api-runtime-evidence-20260813.json`](04-api/api-runtime-evidence-20260813.json)；`APP-ONNX-001` 至 `APP-ONNX-003` 的 MNIST、ORT、动态 Profile 和高级构建批次证据见 [`03-applications/onnxtoengine/onnxtoengine-runtime-evidence-20260813.json`](03-applications/onnxtoengine/onnxtoengine-runtime-evidence-20260813.json)；`APP-EXEC-001` 至 `APP-EXEC-005` 的 CLI、GUI、benchmark、Reference 和 Refit 批次证据见 [`03-applications/tensorrtexec/tensorrtexec-runtime-evidence-20260813.json`](03-applications/tensorrtexec/tensorrtexec-runtime-evidence-20260813.json)；`BLD-001` 至 `BLD-004` 的生成、Windows Native Bridge 与本地包批次证据见 [`06-source-build/source-build-evidence-20260813.json`](06-source-build/source-build-evidence-20260813.json)；`INS-001`、`INS-003` 的 Ubuntu 24.04 容器 Linux Bridge 与真实 GPU 包消费者证据，以及 `INS-002`、`INS-004` 的继续阻塞边界，见 [`05-installation/installation-runtime-evidence-20260814.json`](05-installation/installation-runtime-evidence-20260814.json)。

根目录历史文档没有被批量迁移或删除。它们继续包含：

- 用户教程和模型案例；
- API、ABI、生命周期和兼容性设计说明；
- CI/证据 schema/发布门禁的内部工程记录；
- Owner 输入、候选 proof 和 post-publish 模板；
- `publishing` 子目录中的文章规划与发布治理材料。

2026-08-11 整理前基线复核为 361 个 Markdown；当前公开入口按七个模块组织，50 篇 canonical 长文由 `article-index.json` 单独管理。数量代表项目过程材料多，不代表所有历史 Markdown 都是完整公开文章。

## 1. 完整文章标准

技术文章只有同时满足以下条件，才可以标记为 `complete-technical-article`：

1. 标题后先写“前言”，介绍 TensorRT CSharp API v4.0 的项目定位、核心优势、GitHub 源码、稳定版 NuGet、Runtime Bridge 和本文对应代码入口；让读者即使从单篇文章进入，也能找到项目和完整源码。
2. 从环境、模型与图片获取开始，一步步完成项目创建、依赖配置、代码编写、编译和运行。
3. 有真实执行结果，不把模板、build-only、local feed 或指标卡片单独写成程序运行结果。
4. 图像模型文章至少有两张 PNG/JPEG/WebP：一张原图叠加识别结果，一张真实终端或软件运行页面截图。
5. 模型案例必须写明权重/ONNX 获取 URL、固定 revision、许可证、转换命令、ONNX 输入输出合同、SHA256 和外层 `models` 暂存路径。
6. 配图不得嵌入没有公开再分发授权的测试图片或第三方素材；图片来源和许可证必须可复核。
7. 正文命令使用工作区变量和相对路径，不堆叠盘符、用户名或某台机器的绝对路径。
8. 证据哈希、失败条件和 proof boundary 放在结果与复查部分，不代替教程主体。

## 2. 文章规范

完整的标题、链接、程序输出、图片宽度、模块边界和文章声明约束见 [`publishing/public-article-writing-spec.md`](publishing/public-article-writing-spec.md)。提交前运行 `eng/Test-PublicArticleIndex.ps1`，规范文章不在结尾链接未发布的下一篇。
9. 通过 `eng/Test-TechnicalArticleCompleteness.ps1`。

当前 canonical 文章还必须通过 `eng/Test-PublicArticleIndex.ps1`。该检查会验证“前言”、项目源码链接、`JYPPX.TensorRT.CSharp.API 4.0.0` 链接，以及 Samples 文章的 GitHub 案例源码入口。

## 3. 分类

| 分类 | 含义 | 可作为完整对外文章 |
| --- | --- | --- |
| `complete-technical-article` | 正文、真实结果、配图、来源和边界齐全 | 内容完整，但公开发布仍需 Owner 授权 |
| `documentation-ready` | 适合作为项目文档，可能缺执行配图或独立叙事 | 否 |
| `internal-engineering-record` | 设计、门禁、schema、proof 或审计记录 | 否 |
| `draft-needs-runtime-or-images` | 教程骨架存在，但缺真实执行或配图 | 否 |
| `owner-input-template` | 等待外部/Owner 回填的模板 | 否 |

本目录盘点到 419 个 Markdown 文件。真实模型文章的机器可读完整性目录位于 `publication-catalog.json`；新模块长文的 canonical 与发布冻结状态位于 `article-index.json`。未进入任何公开索引的历史文件默认归类为 `project-documentation-not-publication-ready`，只作为项目文档或迁移素材，不因 DocFX 能构建就自动成为公开文章。

当前严格目录 10/10 通过的文章有：

- [使用 TensorRT CSharp API v4.0 在 C# 中运行 ResNet18 图像分类](classification-real-asset-walkthrough.md)。
- [使用项目自有数字图片完成 MNIST TensorRT 与 ONNX Runtime 双重验证](onnxtoengine-mnist-owner-generated-tutorial.md)。
- [使用 TensorRT CSharp API v4.0 在 C# 中运行 YOLOv8n 目标检测](yolovision-yolov8n-detection-tutorial.md)。
- [使用 TensorRT CSharp API v4.0 在 C# 中运行 LRASPP 语义分割](yolovision-lraspp-semantic-segmentation-tutorial.md)。
- [使用 TensorRT CSharp API v4.0 在 C# 中运行 YOLOv8n 图像分类](yolovision-yolov8n-classification-tutorial.md)。
- [使用 TensorRT CSharp API v4.0 在 C# 中运行 YOLOv8n 姿态估计](yolovision-yolov8n-pose-tutorial.md)。
- [使用 TensorRT CSharp API v4.0 在 C# 中运行 YOLOv8n OBB 旋转目标检测](yolovision-yolov8n-obb-tutorial.md)。
- [使用 TensorRT CSharp API v4.0 在 C# 中运行 YOLOv8n 实例分割](yolovision-yolov8n-instance-segmentation-tutorial.md)。

上述 6 篇 YoloVision 文章使用不含 `local-package-consumer` 的稳定 canonical 路径。旧文件名继续保留完整历史正文，供已有链接和证据合同兼容，但不再作为严格目录或公开导航入口。
- [使用 TensorRT CSharp API v4.0 在 C# 中运行 YOLOv10n End-to-End 目标检测](yolovision-yolov10n-real-asset-tutorial.md)。
- [使用 TensorRT CSharp API v4.0 在 C# 中运行官方 YOLOX-S 目标检测](yolovision-yolox-official-runtime-tutorial.md)。

非模型实机文章不需要原图叠加结果，按独立标准检查真实运行窗口、依赖获取、完整代码流程、失败诊断和证据边界：

- [在 C# 中使用 TensorRT CSharp API v4.0 动态编译并运行 CUDA Kernel](cuda-runtime-compilation-technical-article.md)。
- [使用 TensorRT CSharp API v4.0 完成 Dynamic Shape 推理](dynamic-shape-optimization-profile-tutorial.md)。
- [使用 TensorRT CSharp API v4.0 管理推理输入、显存绑定与 GPU 输出读回](inference-bindings-tutorial.md)。
- [使用 TensorRT CSharp API v4.0 在 C# 中实现 CUDA 多流与 Event 同步](cuda-stream-event-multistream-tutorial.md)。
- [用本地 NuGet 包验证 TensorRT IProgressMonitor：真实构建进度与安全取消](progress-monitor-local-package-consumer-tutorial.md)。
- [用本地 NuGet 包验证 TensorRT IProfiler：即时计时、延迟上报与异常隔离](profiler-local-package-consumer-tutorial.md)。
- [用本地 NuGet 包验证 TensorRT ILogger：真实日志、生命周期与异常隔离](logger-local-package-consumer-tutorial.md)。
- [用本地 NuGet 包验证 TensorRT IStreamReaderV2：安全所有权、真实读取与失败闭环](stream-reader-local-package-consumer-tutorial.md)。
- [使用 TensorRT CSharp API v4.0 验证 TensorRT 10 同主版本兼容宿主](tensorrt10-compatible-host-source-runtime.md)。

模型转换和桌面工具文章需要真实模型构建、程序窗口、报告校验和清楚的 build/runtime 边界；只有声称完成推理的文章才必须提供输出语义校验。未经授权的输入图片不嵌入仓库：

- [使用 TensorRT CSharp API v4.0 将 MNIST ONNX 转换为 TensorRT Engine 并推理](onnx-to-engine-quickstart.md)。
- [使用项目自有数字图片完成 MNIST TensorRT 与 ONNX Runtime 双重验证](onnxtoengine-mnist-owner-generated-tutorial.md)。
- [使用 TensorRtExec GUI 将 ONNX 构建为 TensorRT Engine](tensorrtexec-gui-user-guide.md)。
- [使用 TensorRT CSharp API v4.0 从本地 NuGet 包加载 Refitted Plan 并完成 MNIST 推理](tensorrtexec-refitted-plan-local-package-consumer.md)。

其余模型教程即使已有真实 TensorRT 证据，在补齐原图叠加结果和真实程序窗口截图前，也不会标记为完整技术文章。

DocFX 构建成功只说明链接和站点生成正确，不等于外部文章已经发布，也不替代 public-package、post-publish、Owner acceptance 或 release proof。
