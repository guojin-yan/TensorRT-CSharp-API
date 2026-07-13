# 30+ 篇技术与宣发文章规划

rticle-roadmap-30plus.json 是面向微信公众号、博客和项目文档的机器可读文章规划。它不是 proof，不批准公开发布，也不关闭 release issue。

## Summary

| Field | Value |
| --- | --- |
| roadmapState | `release-readiness-planning` |
| articleCount | `40` |
| minimumArticleCount | `30` |
| canPublishPublicly | `False` |
| canCloseReleaseIssue | `False` |

## Boundary

The 30+ article roadmap is publication planning only: not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push. failedBlockerCount=0 is not ready.

Rows marked `near-ready-owner-proof-input` are article-body and owner proof input guidance only. They still require Owner-provided real assets, public package metadata, stdout/stderr logs, SHA256 values, host metadata, and strict validators before any proof promotion.

## 文章矩阵

| Id | Title | Audience | Status | Target |
| --- | --- | --- | --- | --- |
| `1` | TensorRtSharp4.0 项目总览：把 TensorRT/CUDA 带到 .NET | C# 工程师、AI 推理平台开发者 | `planned` | docs/articles/zh-cn/publishing/tensorrtsharp4-0-项目总览-把-tensorrt-cuda-带到-net.md |
| `2` | 为什么 TensorRT C# Binding 不能只是简单 P/Invoke | 系统开发者、库维护者 | `planned` | docs/articles/zh-cn/publishing/为什么-tensorrt-c-binding-不能只是简单-p-invoke.md |
| `3` | 从 missing 清零到 deferred 边界提升：真实完成度怎么看 | 项目评审者、贡献者 | `ready` | docs/articles/zh-cn/publishing/从-missing-清零到-deferred-边界提升-真实完成度怎么看.md |
| `4` | TensorRT 8/10/11 跨版本封装策略 | TensorRT 用户、库维护者 | `planned` | docs/articles/zh-cn/publishing/tensorrt-8-10-11-跨版本封装策略.md |
| `5` | Windows 本地开发环境：CUDA、TensorRT、.NET 与 CMake | Windows 用户 | `needs screenshot` | docs/articles/zh-cn/publishing/windows-本地开发环境-cuda-tensorrt-net-与-cmake.md |
| `6` | NuGet 安装与 runtime 包选择指南 | NuGet 消费者 | `planned` | docs/articles/zh-cn/publishing/nuget-安装与-runtime-包选择指南.md |
| `7` | Package Consumer Runtime Proof 为什么不能用本地 feed 替代 | 发布负责人 | `planned` | docs/articles/zh-cn/publishing/package-consumer-runtime-proof-为什么不能用本地-feed-替代.md |
| `8` | Plugin Registry Inventory 只读 API 使用指南 | TensorRT plugin 用户 | `needs smoke` | docs/articles/zh-cn/publishing/plugin-registry-inventory-只读-api-使用指南.md |
| `9` | TensorRT Engine 与 ExecutionContext 部署诊断 | 推理服务开发者 | `planned` | docs/articles/zh-cn/publishing/tensorrt-engine-与-executioncontext-部署诊断.md |
| `10` | OnnxToEngine 快速入门：从 ONNX 到 TensorRT Engine | 模型部署工程师 | `ready` | docs/articles/zh-cn/publishing/onnxtoengine-快速入门-从-onnx-到-tensorrt-engine.md |
| `11` | OnnxToEngine 与官方 trtexec 参数对照 | TensorRT 命令行用户 | `ready` | docs/articles/zh-cn/publishing/onnxtoengine-与官方-trtexec-参数对照.md |
| `12` | TensorRtExec CLI：面向用户的 trtexec-like 工具 | 命令行用户 | `ready` | docs/articles/zh-cn/publishing/tensorrtexec-cli-面向用户的-trtexec-like-工具.md |
| `13` | TensorRtExec WinForms：Windows 桌面模型转换工作流 | Windows 桌面用户 | `needs screenshot` | docs/articles/zh-cn/publishing/tensorrtexec-winforms-windows-桌面模型转换工作流.md |
| `14` | TensorRtExec 能力矩阵：哪些已实现，哪些仍是 planned | 用户和维护者 | `ready` | docs/articles/zh-cn/publishing/tensorrtexec-能力矩阵-哪些已实现-哪些仍是-planned.md |
| `15` | YoloVision 总览：一个样例覆盖 YOLO 全系列 | 视觉模型部署用户 | `ready` | docs/articles/zh-cn/publishing/yolovision-总览-一个样例覆盖-yolo-全系列.md |
| `16` | YoloVision Detection 教程 | 检测模型用户 | `near-ready-owner-proof-input` | docs/articles/zh-cn/publishing/yolovision-detection-教程.md |
| `17` | YoloVision Segmentation 教程 | 分割模型用户 | `near-ready-owner-proof-input` | docs/articles/zh-cn/publishing/yolovision-segmentation-教程.md |
| `18` | YoloVision Pose 教程 | 姿态估计用户 | `needs sample` | docs/articles/zh-cn/publishing/yolovision-pose-教程.md |
| `19` | YoloVision OBB 教程 | 遥感/旋转框检测用户 | `needs sample` | docs/articles/zh-cn/publishing/yolovision-obb-教程.md |
| `20` | YoloVision Classification 与 Semantic Segmentation | 分类/语义分割用户 | `needs sample` | docs/articles/zh-cn/publishing/yolovision-classification-与-semantic-segmentation.md |
| `21` | YOLO 模型资产怎么选：许可证、hash 与可再分发边界 | 样例维护者、发布负责人 | `planned` | docs/articles/zh-cn/publishing/yolo-模型资产怎么选-许可证-hash-与可再分发边界.md |
| `22` | YoloVision 预处理与后处理 Metadata 指南 | 视觉模型开发者 | `ready` | docs/articles/zh-cn/publishing/yolovision-预处理与后处理-metadata-指南.md |
| `23` | Classification 样例：自备 ONNX 分类模型接入 | 分类模型用户 | `needs sample` | docs/articles/zh-cn/publishing/classification-样例-自备-onnx-分类模型接入.md |
| `24` | Dynamic Shape 与 Optimization Profile 实战 | 部署工程师 | `ready` | docs/articles/zh-cn/publishing/dynamic-shape-与-optimization-profile-实战.md |
| `25` | InferenceBindings：让 TensorRT 输入输出绑定更像 C# | C# 开发者 | `ready` | docs/articles/zh-cn/publishing/inferencebindings-让-tensorrt-输入输出绑定更像-c.md |
| `26` | MultiStream：CUDA Stream/Event 入门 | CUDA 初学者 | `ready` | docs/articles/zh-cn/publishing/multistream-cuda-stream-event-入门.md |
| `27` | CUDA error 35 排查：驱动、runtime 与 TensorRT 版本 | 环境部署用户 | `planned` | docs/articles/zh-cn/publishing/cuda-error-35-排查-驱动-runtime-与-tensorrt-版本.md |
| `28` | Callback 与 Allocator 为什么要谨慎提升 | 高级用户、贡献者 | `planned` | docs/articles/zh-cn/publishing/callback-与-allocator-为什么要谨慎提升.md |
| `29` | Release Evidence Bundle 怎么读 | 发布负责人 | `planned` | docs/articles/zh-cn/publishing/release-evidence-bundle-怎么读.md |
| `30` | 真实公开发布前最后一步：Owner 输入与 StrictClose | 发布 Owner | `planned` | docs/articles/zh-cn/publishing/真实公开发布前最后一步-owner-输入与-strictclose.md |
| `31` | TensorRtSharp4.0 常见问题排查合集 | 所有用户 | `planned` | docs/articles/zh-cn/publishing/tensorrtsharp4-0-常见问题排查合集.md |
| `32` | 从样例到博客：如何准备一篇可复现的模型部署文章 | 项目维护者、技术作者 | `planned` | docs/articles/zh-cn/publishing/从样例到博客-如何准备一篇可复现的模型部署文章.md |
| `33` | 发布前 Proof 边界：样例、工具报告与真实运行证据的分层 | 发布负责人、项目评审者、贡献者 | `ready` | docs/articles/zh-cn/publishing/发布前-proof-边界-样例-工具报告与真实运行证据的分层.md |
| `34` | TensorRtExec 报告为什么不能替代真实模型运行 Proof | 工具用户、发布 Owner | `ready` | docs/articles/zh-cn/publishing/tensorrtexec-报告为什么不能替代真实模型运行-proof.md |
| `35` | OnnxToEngine 与 trtexec-like 转换：能力、报告和 Proof 边界 | 模型部署工程师、TensorRT 命令行用户 | `ready` | docs/articles/zh-cn/publishing/onnxtoengine-与-trtexec-like-转换-能力-报告和-proof-边界.md |
| `36` | YoloVision 真实资产证据包：模型、图片、labels、hash 与输出校验 | 视觉样例维护者、发布负责人 | `near-ready-owner-proof-input` | docs/articles/zh-cn/publishing/yolovision-真实资产证据包-模型-图片-labels-hash-与输出校验.md |
| `37` | CUDA 初始化 Proof Scaffold：SetValidDevices、InitDevice 与 ChooseDevice 的本地 smoke 分层 | CUDA wrapper 维护者、发布负责人 | `ready` | docs/articles/zh-cn/publishing/cuda-初始化-proof-scaffold-本地-smoke-分层.md |
| `38` | CUDA Graph Event Node borrowed handle 安全边界：HasEvent 替代 GetEvent | CUDA Graph API 维护者、安全评审者 | `ready` | docs/articles/zh-cn/publishing/cuda-graph-event-node-borrowed-handle-安全边界.md |
| `39` | Package Consumer Proof 分层边界：local smoke、package-feed substitute 与 clean external evidence | 发布负责人、包验证维护者 | `ready` | docs/articles/zh-cn/publishing/package-consumer-分层边界-local-smoke-package-feed-substitute-clean-external-evidence.md |
| `40` | YoloVision 真实资产证据链：从样例矩阵到 owner proof | 视觉样例维护者、发布负责人 | `near-ready-owner-proof-input` | docs/articles/zh-cn/publishing/yolovision-真实资产证据链-从样例矩阵到-owner-proof.md |

## Must Avoid Claims

- 不把 local feed、ProjectReference、direct nupkg、dry-run、dashboard、runbook、candidate、draft 或 build-only 写成 proof。
- 不把 failedBlockerCount=0 写成 ready。
- 不恢复旧样例公开入口名。
