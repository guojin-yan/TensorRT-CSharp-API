# TensorRtSharp 4.0.0 RC 发布说明草案

TensorRtSharp 4.0.0 RC 的目标是把 TensorRT 与 CUDA 的 C ABI bridge、C# 高层 wrapper、runtime package、smoke、文档和质量门禁收敛到可审计的发布候选状态。

## 本轮重点

- TensorRT/CUDA manifest 与 generated binding 已保持零 missing 基线，但 deferred 仍按真实可用性审计。
- Plugin registry inventory、runtime diagnostics、allocator/debug-listener safe controls、ONNX/parser/refitter 诊断、CUDA graph 与 memory range 等高价值接口已逐批提升。
- DebugListener callback 相关接口已经推进到 proof gate：默认不启用 non-null attach，不安装真实 vtable，不伪造 invocation。
- Windows `win-x64-trt11.0-cuda13.2-cudnn9.22` 已完成 native build、runtime package、bridge consumer、package consumer 和 local feed consumer。
- 当前机器 full runtime smoke 被 CUDA error 35 阻塞，记录为环境 warning；真实 callback runtime proof 仍为 false。

## 当前包

- managed package：`JYPPX.TensorRT.CSharp.API 4.0.0`
- runtime package：`JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda13.2.cudnn9.22 4.0.0`
- split bridge package：`JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda13.2.cudnn9.22.Bridge 4.0.0`

## 已知限制

- CUDA 13 runtime smoke 需要兼容驱动；当前环境触发 CUDA error 35。
- `IDebugListener::processDebugTensor` 仍未取得 full package consumer `InvocationCount>0` 真实回调证明。
- callback、裸指针、外部资源 ownership 和跨 ABI trampoline 仍按 deferred 边界逐批推进。
- 本地开发签名/信任状态用于 WDAC 排障，不等同于正式发布签名策略。

## 验证入口

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LocalNuGetFeedConsumer.ps1 `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 `
  -AllowSmokeFailure

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCandidateReadiness.ps1 `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 `
  -AllowRuntimeSmokeBlocked
```
