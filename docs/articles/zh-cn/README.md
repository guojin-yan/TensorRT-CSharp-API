# 中文文章目录说明

`docs/articles/zh-cn` 是 DocFX 中文概念文档源目录，不等同于“已经可以发布到公众号或博客的技术文章库”。这里同时包含：

- 用户教程和模型案例；
- API、ABI、生命周期和兼容性设计说明；
- CI/证据 schema/发布门禁的内部工程记录；
- Owner 输入、候选 proof 和 post-publish 模板；
- `publishing` 子目录中的文章规划与发布治理材料。

2026-08-04 盘点到 487 个 Markdown 文件。当前有 7 篇文章同时具备真实程序运行窗口截图和原图叠加识别结果；其余文件仍以项目文档、内部记录或待完善稿为主。数量代表项目过程材料多，不代表有 487 篇完整成稿。

## 完整文章标准

技术文章只有同时满足以下条件，才可以标记为 `complete-technical-article`：

1. 先介绍项目、使用到的库、各依赖职责和目标读者。
2. 从环境、模型与图片获取开始，一步步完成项目创建、依赖配置、代码编写、编译和运行。
3. 有真实执行结果，不把模板、build-only、local feed 或指标卡片单独写成程序运行结果。
4. 图像模型文章至少有两张 PNG/JPEG/WebP：一张原图叠加识别结果，一张真实终端或软件运行页面截图。
5. 模型案例必须写明权重/ONNX 获取 URL、固定 revision、许可证、转换命令、ONNX 输入输出合同、SHA256 和外层 `models` 暂存路径。
6. 配图不得嵌入没有公开再分发授权的测试图片或第三方素材；图片来源和许可证必须可复核。
7. 正文命令使用工作区变量和相对路径，不堆叠盘符、用户名或某台机器的绝对路径。
8. 证据哈希、失败条件和 proof boundary 放在结果与复查部分，不代替教程主体。
9. 通过 `eng/Test-TechnicalArticleCompleteness.ps1`。

## 分类

| 分类 | 含义 | 可作为完整对外文章 |
| --- | --- | --- |
| `complete-technical-article` | 正文、真实结果、配图、来源和边界齐全 | 内容完整，但公开发布仍需 Owner 授权 |
| `documentation-ready` | 适合作为项目文档，可能缺执行配图或独立叙事 | 否 |
| `internal-engineering-record` | 设计、门禁、schema、proof 或审计记录 | 否 |
| `draft-needs-runtime-or-images` | 教程骨架存在，但缺真实执行或配图 | 否 |
| `owner-input-template` | 等待外部/Owner 回填的模板 | 否 |

机器可读发布目录位于 `publication-catalog.json`。只有被显式列入 `articles` 且通过严格门禁的文件，才算内容完整；所有未列入文件默认是 `project-documentation-not-publication-ready`。

当前完成门禁的文章有：

- [使用 TensorRtSharp4.0 在 C# 中运行 ResNet18 图像分类](classification-real-asset-walkthrough.md)。
- [使用 TensorRtSharp4.0 在 C# 中运行 YOLOv8n 目标检测](yolovision-yolov8n-det-local-package-consumer-tutorial.md)。
- [使用 TensorRtSharp4.0 在 C# 中运行 LRASPP 语义分割](yolovision-lraspp-semantic-local-package-consumer-tutorial.md)。
- [使用 TensorRtSharp4.0 在 C# 中运行 YOLOv8n 图像分类](yolovision-yolov8n-cls-local-package-consumer-tutorial.md)。
- [使用 TensorRtSharp4.0 在 C# 中运行 YOLOv8n 姿态估计](yolovision-yolov8n-pose-local-package-consumer-tutorial.md)。
- [使用 TensorRtSharp4.0 在 C# 中运行 YOLOv8n OBB 旋转目标检测](yolovision-yolov8n-obb-local-package-consumer-tutorial.md)。
- [使用 TensorRtSharp4.0 在 C# 中运行 YOLOv8n 实例分割](yolovision-yolov8-seg-local-package-consumer-tutorial.md)。

非模型实机文章不需要原图叠加结果，按独立标准检查真实运行窗口、依赖获取、完整代码流程、失败诊断和证据边界：

- [在 C# 中使用 TensorRtSharp4.0 动态编译并运行 CUDA Kernel](cuda-runtime-compilation-technical-article.md)。
- [使用 TensorRtSharp4.0 完成 Dynamic Shape 推理](dynamic-shape-optimization-profile-tutorial.md)。
- [使用 TensorRtSharp4.0 管理推理输入、显存绑定与 GPU 输出读回](inference-bindings-tutorial.md)。
- [使用 TensorRtSharp4.0 在 C# 中实现 CUDA 多流与 Event 同步](cuda-stream-event-multistream-tutorial.md)。

其余模型教程即使已有真实 TensorRT 证据，在补齐原图叠加结果和真实程序窗口截图前，也不会标记为完整技术文章。

DocFX 构建成功只说明链接和站点生成正确，不等于外部文章已经发布，也不替代 public-package、post-publish、Owner acceptance 或 release proof。
