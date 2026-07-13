# 技术与宣发文章矩阵 30+

本文是面向微信公众号、博客和项目主页的发布矩阵。它不是 API 文档目录，也不是 runtime proof；它用于安排有头有尾、可配图、可落地到仓库样例的技术与宣传文章。

## 规划原则

- 不为了凑数量牺牲质量。
- 每篇文章必须有目标读者、文章类型、样例路径、图片需求和 proof 边界。
- 案例文章必须说明模型获取、许可证、SHA256、输入输出 metadata 和运行日志要求。
- 涉及发布的文章必须明确 local feed、ProjectReference、direct `.nupkg`、template、dry-run、build-only 都不是 proof。

## 文章矩阵

| # | 标题 | 目标读者 | 类型 | 样例/代码路径 | 真实资产需求 | Proof 边界 | 优先级 |
|---|---|---|---|---|---|---|---|
| 1 | TensorRtSharp4.0 项目总览 | C# AI 推理开发者 | 宣发 | `README.zh-CN.md` | 否 | 不是 runtime proof | P0 |
| 2 | TensorRT C# Binding 为什么不能只是 P/Invoke | 库维护者 | 深度解析 | `native/src/tensorrt` | 否 | ABI 设计说明 | P0 |
| 3 | 从 missing 清零到 deferred 边界提升 | 项目评审者 | 技术解析 | `artifacts/interface-coverage` | 否 | 完成度说明，不是可用性全证明 | P0 |
| 4 | TensorRT 8/10/11 跨版本封装策略 | TensorRT 用户 | 深度解析 | `native/manifests/tensorrt` | 否 | version guard 说明 | P0 |
| 5 | NuGet 安装与 runtime 包选择 | NuGet 消费者 | 教程 | `pack/runtime-split` | 否 | 安装教程不是 package proof | P0 |
| 6 | Windows CUDA/TensorRT/.NET 环境搭建 | Windows 用户 | 教程 | `docs/articles/zh-cn/windows-local-dev-environment.md` | 截图 | 环境搭建不是 runtime proof | P1 |
| 7 | Linux runner handoff 与兼容主机 proof | Linux 用户 | 教程 | `docs/articles/zh-cn/linux-runner-evidence-checklist.md` | 是 | 需要 owner log/hash | P1 |
| 8 | Package Consumer Runtime Proof 为什么不能用本地 feed 替代 | 发布负责人 | 深度解析 | `eng/Test-PackageConsumerRuntimeProofRecord.ps1` | 是 | 必须 clean external consumer | P0 |
| 9 | Plugin Registry Inventory 只读 API | plugin 用户 | 接口介绍 | `TensorRtPluginRegistryInventory.cs` | 否 | copied metadata diagnostic | P1 |
| 10 | ErrorRecorder Snapshot 与 InterfaceInfo | 调试用户 | 接口介绍 | `TensorRtErrorRecorderSnapshot.cs` | 否 | copied snapshot diagnostic | P1 |
| 11 | Engine 与 ExecutionContext 部署诊断 | 推理服务开发者 | 接口介绍 | `TensorRtEngineDeploymentSnapshot` | 否 | 诊断不是输出 proof | P1 |
| 12 | OnnxToEngine 快速入门 | 模型部署工程师 | 教程 | `samples/OnnxToEngine` | 否 | build-only | P0 |
| 13 | OnnxToEngine 与官方 trtexec 参数对照 | trtexec 用户 | 深度解析 | `samples/OnnxToEngine/trtexec-parity-matrix.json` | 否 | parity 不是 runtime proof | P0 |
| 14 | TensorRtExec CLI：C# 版 trtexec-like 工具 | 工具用户 | 教程 | `applications/TensorRtExec` | 否 | build/report 不是 proof | P0 |
| 15 | TensorRtExec WinForms 图形界面教程 | Windows 桌面用户 | 教程 | `applications/TensorRtExec/WinForms` | 截图 | GUI 截图不是 proof | P1 |
| 16 | TensorRtExec parity matrix 解读 | 工具用户 | 深度解析 | `tensor-rt-exec-trtexec-parity-matrix.json` | 否 | parse-only 不晋级 | P0 |
| 17 | YoloVision 总览：统一 YOLO 系列样例 | CV 开发者 | 宣发+教程 | `samples/YoloVision` | 否 | README 不是 proof | P0 |
| 18 | YOLOv8 Detection 真实资产教程 | CV 开发者 | 案例 | `samples/assets/yolovision-yolov8-det-candidate.template.json` | 是 | owner-action-required | P0 |
| 19 | YOLOv8 Segmentation 真实资产教程 | CV 开发者 | 案例 | `samples/assets/yolovision-yolov8-seg-candidate.template.json` | 是 | owner-action-required | P0 |
| 20 | YOLO Pose 输出与 keypoint metadata | CV 开发者 | 案例 | `YoloPoseDecoder.cs` | 是 | real-model-runtime 需 log/hash | P1 |
| 21 | YOLO OBB 旋转框与 angle tensor | CV 开发者 | 案例 | `YoloObbDecoder.cs` | 是 | real-model-runtime 需 log/hash | P1 |
| 22 | YOLO Classification 与 Semantic Segmentation | CV 开发者 | 案例 | `YoloSemanticMap.cs` | 是 | real-model-runtime 需 log/hash | P1 |
| 23 | Classification 样例真实资产 walkthrough | 分类模型用户 | 案例 | `samples/Classification` | 是 | sample proof，不是 package proof | P1 |
| 24 | Dynamic shape profile 最佳实践 | 推理服务开发者 | 教程 | `samples/DynamicShape` | 可选 | profile 配置不是 proof | P1 |
| 25 | MultiStream CUDA stream/event 教程 | 性能用户 | 教程 | `samples/MultiStream` | 否 | smoke 需单独记录 | P1 |
| 26 | CUDA Graph 能力与边界 | 性能用户 | 深度解析 | `smoke/CudaGraphSmokeRunner` | 否 | graph smoke 不等于模型质量 | P2 |
| 27 | EngineInspector 与 layer dump | 调试用户 | 教程 | `TensorRtEngineInspector` | 否 | layer dump 是 diagnostic | P2 |
| 28 | Refit weights 使用指南 | 模型维护者 | 教程 | `smoke/RefitWeightsSmokeRunner` | 可选 | refit proof 需具体模型 | P2 |
| 29 | INT8/calibration 边界与后续计划 | 性能用户 | 深度解析 | `BuilderConfigScalarControlsTests.cs` | 是 | calibrator ownership 未完成 | P2 |
| 30 | 常见安装与 DLL 加载问题排查 | 新用户 | 排障 | `docs/articles/zh-cn/troubleshooting-index.md` | 截图 | 排障不是 proof | P0 |
| 31 | 常见模型转换问题排查 | 模型部署工程师 | 排障 | `TensorRtExec` / `OnnxToEngine` | 可选 | build error 不是 proof | P0 |
| 32 | 版本升级与 CUDA/TensorRT 组合选择 | 维护者 | 指南 | `artifacts/interface-coverage` | 否 | 矩阵不是 runtime proof | P1 |
| 33 | Release proof evidence ladder | 发布负责人 | 深度解析 | `onnxtoengine-tensorrtexec-yolovision-evidence-ladder.md` | 是 | validator 才能晋级 | P0 |
| 34 | 发布流程与 owner action checklist | 发布负责人 | 发布说明 | `docs/articles/zh-cn/release-final-owner-action-sequence.md` | 是 | 不执行 publish | P0 |
| 35 | 案例合集：从 ONNX 到 YOLO 实战 | 决策者/开发者 | 宣发+案例 | `samples` / `applications` | 是 | 文章合集不是 proof | P1 |
| 36 | ONNX Parser 与 ParserRefitter 诊断 release gate | 模型转换维护者、发布负责人 | 发布证据说明 | `TensorRtOnnxParserDiagnosticSnapshot.cs` / `Test-RuntimePackageReadiness.ps1` | 否 | copied diagnostics，不是 runtime proof | P0 |

## 配图计划

- 架构图：managed wrapper、C ABI bridge、native TensorRT。
- 证据梯度图：tutorial -> app report -> sample run -> real-model-runtime -> package-consumer-runtime。
- YOLO 后处理图：det/seg/pose/obb/sem 的输出 tensor 到结果对象。
- TensorRtExec 界面截图：CLI output、WinForms report 配置。
- NuGet 包结构图：managed package 与 runtime package。

## 下一步

优先完成 P0 文章正文，并且每篇案例文章都对应仓库中的样例路径、命令、资产模板和 proof boundary。需要真实模型资产的文章先以 owner-action-required 模板发布内部草稿，等待 owner 回填后再标记为 ready。
