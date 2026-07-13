# TensorRtSharp 4.0 架构拆解：Native ABI、Manifest、Generator 与 C# Wrapper

## 适用读者

这篇文章面向库维护者、需要审查 ABI 稳定性的 C#/.NET 工程师，以及想理解 TensorRtSharp 4.0 为什么不直接暴露 TensorRT C++ 对象的读者。它适合作为架构深度文章发布，也适合作为贡献者上手前的背景材料。

## 解决问题

TensorRT 原生 API 以 C++ 对象、版本差异、插件接口和跨语言生命周期为核心。如果 C# 层直接 P/Invoke 到不稳定对象或裸指针，短期可能看起来接口更多，长期会造成 ABI 破裂、生命周期悬空、异常跨边界传播和版本 guard 混乱。TensorRtSharp 4.0 的架构目标是：让 native bridge 承担版本差异，让 manifest/generator 固化入口形状，让 C# wrapper 暴露有语义的对象和结果。

## 架构拆解

项目的核心分层可以理解为“C++ 复杂性向 native 内聚，C# 只面对稳定语义”。

- `native/manifests/tensorrt/v8|v10|v11` 描述跨版本入口与 guard。
- `native/src/tensorrt/common` 放置跨版本共享实现、字符串 copy、数组 copy 和 no-throw 边界。
- `native/src/tensorrt/v10`、`native/src/tensorrt/v11` 处理版本特有能力。
- `src/JYPPX.TensorRtSharp` 暴露高层 wrapper，不让用户直接管理 borrowed pointer。
- `eng/Generate-Bindings.ps1`、`eng/Test-BindingGeneratorOutputs.ps1`、`eng/Export-InterfaceCoverageMatrix.ps1` 保持生成物、manifest 与 coverage 的一致性。

这样的分层让 deferred 接口不再只是“有没有入口”的问题，而是被拆成：native 是否真实实现、参数是否安全、返回值是否 copy 出来、C# wrapper 是否有语义、测试是否能覆盖、跨版本 guard 是否正确。

## Wrapper 设计原则

C# 层优先提供明确生命周期对象、只读 snapshot 和可诊断结果。对 plugin creator、engine inspector、error recorder、builder config 等只读 API，优先采用 caller buffer 或 count/copy 模式；对 callback、allocator、plugin instance、external resource、enqueue 等 ownership 复杂的 API，则保留 deferred 或设计 gate，直到 native no-throw、GCHandle/delegate pinning、dispose 顺序和 package-consumer proof 都可审计。

一个典型可接受的提升路径是：先把 `registry exists`、`creator count`、`creator name/version/namespace` 这类只读数据 copy 到托管结构，再提供 `IReadOnlyList<T>` 或 record 类型。不可接受的路径是把 plugin creator 的裸地址直接暴露给 public API。

## 示例检查命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
dotnet build .\TensorRtSharp.sln -c Debug --no-restore
```

这些命令验证的是生成和构建链路。它们可以证明架构一致性的一部分，但不能证明 public package consumer runtime 已经通过。

## 边界说明

本文是架构说明，不是 runtime proof。`build-only`、`dry-run`、`template`、local feed、ProjectReference、direct `.nupkg`、GUI screenshot、TensorRtExec report、YoloVision matrix、OnnxToEngine report、sidecar、readonly diagnostics 都不能替代 package-consumer-runtime proof。只有真实仓库外 clean consumer smoke、public package source、日志、hash、host/package metadata 和 strict validator 通过后，才能提升 release proof。

## 下一步

下一步适合阅读 `cuda-tensorrt-cudnn-version-matrix-guide.md`，理解为什么 TRT8/TRT10/TRT11、CUDA 11/12/13、cuDNN 8/9 需要用 runtime package 和 version guard 管理，而不是把所有 DLL 都混在一个安装说明里。
