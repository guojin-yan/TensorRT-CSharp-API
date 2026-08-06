# TensorRtSharp4.0 是什么

TensorRtSharp4.0 是面向 .NET 的 TensorRT / CUDA 高层封装项目。它的目标不是把 NVIDIA C++ API 原样暴露给 C#，而是在 C ABI bridge、托管 interop 和高层 wrapper 之间建立一条可验证的部署链路，让 .NET 用户能够用更稳定的对象模型完成 engine build、deserialize、tensor binding、enqueue、CUDA memory/stream 操作和 package 化部署。

当前项目已经越过“扫描头文件并补齐入口”的阶段。接口覆盖报告可以证明 manifest/source 的入口数量已经追平，但真实可用性仍以非 deferred 实现、高层 C# wrapper、smoke、package consumer 和 readiness 证据为准。

## 目标读者

- 希望在 .NET 项目中使用 TensorRT / CUDA 的 C# 工程师。
- 正在评估 TensorRtSharp4.0 是否适合进入内部 PoC、工具链或推理服务的技术负责人。
- 需要理解 `native/`、`src/JYPPX.TensorRtSharp`、`src/JYPPX.CudaSharp`、`samples/`、`smoke/` 和 `artifacts/` 如何共同证明完成度的维护者。
- 需要区分 API 覆盖、runtime smoke、package consumer proof 和 release close proof 的发布负责人。

## 项目分层

项目主要分为四层：

1. native bridge：位于 `native/`，通过 C ABI 暴露 TensorRT/CUDA 能力。
2. generated interop：位于 `src/*/Internal/Interop/Generated`，由 manifest 和 generator 生成。
3. high-level wrapper：位于 `src/JYPPX.TensorRtSharp` 和 `src/JYPPX.CudaSharp`，面向普通 C# 用户。
4. samples/smoke/package gates：位于 `samples/`、`smoke/`、`eng/` 和 `artifacts/`，用于证明 API 能被构建、消费、运行或明确阻塞。

这四层必须同时成立，才适合把某个接口宣传为用户可用 API。只有 native entrypoint 或 manifest row，不等于高层可用。

## 当前可用路径

当前仓库中更适合作为入门路径的样例包括：

- `samples/Performance/01.MultiStream`：CUDA stream/event 和跨 stream ordering。
- `samples/Inference/02.DynamicShapes`：TensorRT dynamic shape、optimization profile 和 inference binding。
- `samples/Inference/01.Bindings`：最小 identity network 的输入输出绑定和 enqueue。
- `applications/OnnxToEngine`：ONNX parser、serialized engine build、deserialize 和 round-trip。

`samples/ComputerVision/01.Classification` 与 `applications/YoloVision` 是面向真实模型的用户侧样例，但模型、labels、图片不随仓库分发，需要用户自行准备可再分发资产。

更多发布前完成度请查看 `artifacts/interface-coverage/release-api-readiness-audit.json`。它会明确 safe read-only candidates、blocked ownership risk APIs 和跨版本 guard，而不是只看 manifest/source 是否匹配。

## 可复制命令

先从文档和样例入口确认项目形状：

```powershell
dotnet build .\TensorRtSharp.sln -c Debug --no-restore /m:1 /p:UseSharedCompilation=false /nr:false /v:minimal
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~ReleaseReadinessSmokeClosureTests|FullyQualifiedName~ReleaseArticleMatrixTests" /p:UseSharedCompilation=false /nr:false /v:minimal
```

如果要从用户路径试用，优先读这些入口：

- `applications/OnnxToEngine`
- `applications/YoloVision`
- `applications/TensorRtExec`
- `src/JYPPX.TensorRtSharp`
- `src/JYPPX.CudaSharp`

## 当前发布证据

当前 `win-x64-trt11.0-cuda13.2-cudnn9.22` package readiness 已显示：

- `Overall=ready`
- `split components: ready 3/3`
- `readiness blockers: 0`
- `vendor blockers: none`

但 full package consumer runtime smoke 在当前机器上被 CUDA driver/runtime compatibility 阻塞为 `blocked-by-cuda-driver`，错误来自 `cudaRuntimeGetVersion` 的 CUDA error 35。这说明 package 布局、restore/build/native-copy 和 TensorRT bridge 探测已经走到 runtime 边界，但当前机器不是 CUDA 13.2 runtime smoke 的兼容执行环境。

## 必须保留的边界

不要把以下内容当成项目 100% runtime 可用：

- manifest/source 入口数量追平。
- readiness blockers 清零。
- bridge-only dependency probe 通过。
- compile-only package consumer 通过。
- safety gate、design gate、precheck 或 copied-state 证据。

这些材料都是工程 readiness 的一部分，但它们仍然是 not runtime proof、not public package proof、not post-publish proof、not package push、not release close approval。

特别是 callback 相关接口仍保持谨慎边界。在 full package consumer smoke 输出完整 `EvidenceKind=real-callback-runtime`、`RealCallbackRuntime=True` 和 required markers 前，`IGpuAllocator::*`、`IGpuAsyncAllocator::*`、`IOutputAllocator::*` 和 `IDebugListener::processDebugTensor` direct callback rows 必须继续 deferred。

## 适合谁使用

TensorRtSharp4.0 适合：

- 希望在 .NET 中部署 TensorRT engine 的开发者。
- 需要 CUDA memory/stream/event 基础能力的 C# 项目。
- 想要用 NuGet/runtime package 交付 TensorRT/CUDA 依赖的团队。
- 需要清晰诊断 package restore、native asset copy、driver/runtime compatibility 的工程团队。

如果你的需求是直接扩展 TensorRT plugin、注册真实 allocator/debug-listener callback 或暴露 native borrowed pointer，那么当前仍应先阅读 callback boundary 文档，并等待对应 proof 完成。

## 截图与图示建议

- 架构图：`native bridge -> generated interop -> high-level wrapper -> samples/smoke/package gates`。
- 完成度漏斗图：manifest/source match、non-deferred implementation、C# wrapper、smoke、package-consumer proof、post-publish proof。
- 用户路径图：`applications/OnnxToEngine`、`applications/YoloVision`、`applications/TensorRtExec`、`docs/articles/zh-cn`。

## 下一步

- 模型用户从 `applications/OnnxToEngine` 和 `applications/TensorRtExec` 开始，先完成 ONNX build/report。
- 视觉用户从 `applications/YoloVision` 开始，准备模型、labels、输入资产、license 和 SHA256。
- 维护者继续按 `artifacts/interface-coverage/tensorrt-interface-comparison.csv` 筛选安全只读 deferred API，优先完成 manifest/source/interop/wrapper/smoke 闭环。

Boundary keywords: not public package proof, not post-publish proof, not package push, not release close approval.
