# TensorRtSharp 4.0 发布故事：从 Deferred 边界提升到可审计 Release Gate

## 适用读者

这篇文章适合社区读者、潜在用户、贡献者和技术管理者。它用发布故事的方式解释项目为什么反复强调 deferred、proof、owner input 和 release gate。

## 解决问题

很多开源项目在发布前只展示功能列表，很少解释哪些能力已经真实可用，哪些只是规划或占位。TensorRtSharp 4.0 的发布故事不是“接口数量很大”，而是“每个接口、样例、工具和 proof 都能被追踪到真实边界”。

## 从 Missing 接口清零到 Deferred 边界提升

早期主线是补齐缺失接口，让 manifest/source 对齐。随后项目转向 deferred 边界提升：只读查询型 API 优先落地，高 ownership 风险 API 谨慎设计。这个转向让项目避免用占位入口制造完成度，也让 C# wrapper 能跟上 native 实现。

## 从样例到应用

`samples/OnnxToEngine` 负责 ONNX-to-engine 常用路径，目标是尽量贴近官方 trtexec 的模型转换能力。`samples/YoloVision` 替代早期过窄的检测样例命名，统一承载 YOLO 多 family、多 task 教程。`applications/TensorRtExec` 则把 trtexec 风格能力做成 CLI 和 WinForms 双入口，适合演示、排查和 build report。

## 从 Build 到 Proof

项目坚持区分 build、diagnostics、sample evidence、package consumer proof 和 post-publish verification。`artifacts/final-release` 下的 checklist、handoff、dashboard、owner execution pack 都不是为了“看起来能发布”，而是为了让 owner 知道还缺什么，避免把错误证据带到 release close。

## 对外发布时怎么讲

对公众号和博客读者，可以这样描述：TensorRtSharp 4.0 正在把 TensorRT/CUDA 的 C++ 推理能力整理成 C# 可用的稳定 API，同时提供 OnnxToEngine、YoloVision、TensorRtExec 等真实用户入口；项目对 proof 的要求很严格，因此 release close 只接受 clean consumer runtime proof 和 post-publish verification，不接受模板、截图或 build report 替代。

## 边界说明

发布故事不是 runtime proof。`build-only`、`dry-run`、`template`、local feed、ProjectReference、direct `.nupkg`、GUI screenshot、TensorRtExec report、YoloVision matrix、OnnxToEngine report、sidecar、readonly diagnostics 都不能替代 package-consumer-runtime proof、post-publish proof 或 release close approval。

## 下一步

后续文章应继续扩展真实案例：YOLOv8 det/seg/pose/obb/cls、OnnxToEngine 与 trtexec 对照、TensorRtExec GUI/CLI 使用、NuGet runtime package 安装、Windows/Linux 排查，以及真实 owner proof 完成后的 post-publish verification 复盘。
