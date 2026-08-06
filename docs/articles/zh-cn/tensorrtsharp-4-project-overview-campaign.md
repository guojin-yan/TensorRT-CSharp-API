# TensorRtSharp 4.0 项目总览：把 TensorRT/CUDA 带到 C# 生产工作流

## 适用读者

这篇文章适合正在评估 .NET AI 推理栈的技术负责人、需要把 ONNX/TensorRT 工作流接入 C# 服务的工程师，以及关注 CUDA/TensorRT 多版本部署成本的维护者。它不是 API 逐项说明，而是面向公众号、博客和项目首页的项目介绍稿：先讲清楚为什么做 TensorRtSharp 4.0，再讲清楚当前哪些能力已经可用，哪些仍需要真实 proof 才能进入发布闭环。

## 解决问题

很多团队使用 TensorRT 时会遇到三个断点：C++ API 与 C# 业务代码之间的 ABI 边界、CUDA/TensorRT/cuDNN 版本组合带来的部署复杂度，以及“能 build”与“能在真实 consumer 中运行”之间的证据差距。TensorRtSharp 4.0 的目标是把这些断点拆开：底层用稳定 C ABI 桥接 TensorRT/CUDA，托管侧提供 SafeHandle 和高层 wrapper，样例侧提供 `applications/OnnxToEngine`、`applications/YoloVision`、`applications/TensorRtExec` 这类可学习入口，发布侧用 final-release artifact 记录真实 proof 边界。

## 项目主线

项目当前已经从“补齐 manifest/source 接口”进入“deferred 边界提升与发布候选收口”。这意味着 `100% manifest/source 匹配` 不再等于 `100% 可用`。真实完成度取决于非 deferred native 实现、高层 C# wrapper、smoke/package-consumer 验证，以及跨版本 guard 是否一致。这个转向很重要：它避免把占位入口误当作稳定 API，也避免让用户直接面对无语义的 `IntPtr` 或不清晰的对象生命周期。

从用户视角看，项目包含四层：

- Native bridge：`native/src/tensorrt` 和 `native/manifests/tensorrt` 负责把不同 TensorRT 版本的能力落到可审计 C ABI。
- Managed wrapper：`src/JYPPX.TensorRtSharp`、`src/JYPPX.CudaSharp`、`src/JYPPX.TensorRtSharp.Tools` 负责把底层能力变成 C# 可读语义。
- Samples and apps：`applications/OnnxToEngine`、`applications/YoloVision`、`applications/TensorRtExec` 负责给真实使用路径。
- Release proof：`artifacts/final-release` 下的 checklist、handoff、dashboard 和 validation 负责说明哪些可以宣传、哪些仍必须等待 owner proof。

## 当前用户入口

入门建议从 `README.zh-CN.md`、`docs/index.md` 和 `docs/articles/zh-cn/getting-started.md` 开始。模型转换用户可以阅读 `applications/OnnxToEngine/README.md` 和 `applications/TensorRtExec/README.md`；视觉模型用户从 `applications/YoloVision/README.md` 进入，它统一覆盖 YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom，以及 det、cls、seg、obb、pose、sem 等任务。发布和 proof 相关内容则以 `artifacts/final-release/clean-consumer-proof-owner-execution-pack.md`、`artifacts/final-release/final-release-close-blocker-dashboard.md` 为准。

## 示例命令

```powershell
dotnet build .\TensorRtSharp.sln -c Debug --no-restore
dotnet run --project .\applications\TensorRtExec -- --help
dotnet run --project .\applications\OnnxToEngine -- --help
dotnet run --project .\applications\YoloVision -- --help
```

这些命令能帮助读者理解项目入口，但它们本身不是 release runtime proof。真正的 package-consumer-runtime proof 需要仓库外 clean consumer、public package source、真实 smoke 日志、hash、host metadata、package metadata 和 strict validator。

## 边界说明

本文是项目介绍和采用路径说明，不是 proof。`build-only`、`dry-run`、`template`、local feed、ProjectReference、direct `.nupkg`、GUI screenshot、TensorRtExec report、YoloVision matrix、OnnxToEngine report、sidecar、readonly diagnostics 都不是 runtime proof，也不能替代 package-consumer-runtime proof、post-publish proof 或 release close approval。

## 下一步

下一篇文章建议阅读 `tensorrtsharp-4-architecture-abi-wrapper.md`，它会继续解释为什么项目坚持 C ABI、SafeHandle、version guard 和 deferred 风险分层。等 owner 提供真实 public package source 与 clean consumer logs 后，项目才能把 release proof 从 guidance 推进到可提升状态。
