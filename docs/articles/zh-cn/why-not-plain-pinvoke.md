# 为什么不是简单 P/Invoke

把 TensorRT/CUDA 头文件直接翻译成 C# `DllImport` 看起来最快，但它会把 ABI、生命周期、异常、版本差异和 borrowed pointer 语义全部转嫁给最终用户。TensorRtSharp4.0 选择更长的路径：native C ABI bridge + generated interop + 高层 C# wrapper + smoke/package readiness。

这不是为了“包装而包装”，而是为了让接口能被长期维护和诊断。

## C++ ABI 不适合直接暴露

TensorRT 是 C++ API，存在虚函数、对象 ownership、builder/runtime/context 生命周期、版本间 API 变动等问题。直接从 C# 调 C++ ABI 会遇到：

- 编译器 ABI 差异。
- vtable 布局和异常传播风险。
- 对象释放顺序不清晰。
- 不同 TensorRT line 的 API 存在/移除差异。

项目因此在 `native/` 里使用 C ABI bridge，托管层只调用稳定导出入口。跨 ABI 不抛 C++ 异常，而是转换为 status/diagnostic，再由托管层决定如何表达错误。

## 生命周期必须由 wrapper 管理

TensorRT 对象通常不是孤立指针。builder、config、network、runtime、engine、execution context、host memory、optimization profile 等对象有严格的创建和释放关系。CUDA memory、stream、event、pinned memory 也有设备和同步语义。

高层 wrapper 的价值在于：

- 用 `IDisposable` 表达释放边界。
- 避免普通用户直接接触裸 `IntPtr`。
- 把 copied metadata 和 borrowed pointer 分开。
- 把无法安全表达的接口先保留为 deferred。
- 在 smoke 中验证典型生命周期路径。

例如 `TensorRtInferenceBindings` 把 input shape、device buffer、tensor address binding、enqueue 和 readback 组织成用户能理解的流程，而不是让用户拼接一组无语义 native pointer。

## 跨版本 guard 是核心功能

TensorRT 8、TensorRT 10、TensorRT 11 的 API 面并不完全一致。简单 P/Invoke 很容易出现“编译能过，但运行时符号不存在”或“某个版本参数语义不同”的问题。

当前项目要求每一批 API 都保留跨版本证据：

- manifest 中标明版本线。
- native 实现保留 version guard。
- C# interop 和 wrapper 路由到对应 API line。
- smoke 或质量测试覆盖代表性路径。

这也是为什么项目宁愿让一些 callback、plugin resource、ownership 不清晰的接口继续 deferred，也不把不稳定的裸指针 API 推给用户。

## package consumer 比本地 build 更重要

本地 solution build 只能说明源码能编译。真正用户会遇到的是 NuGet restore、runtime package restore、native DLL copy、PATH/loader、driver/runtime compatibility、应用控制策略等问题。

因此项目使用：

- `eng/Test-PackageConsumer.ps1`
- `eng/Test-RuntimePackageReadiness.ps1`
- split/full runtime package artifacts
- bridge consumer wrapper surface evidence

来验证“离开源码目录后是否还能被真实消费”。当前 `win-x64-trt11.0-cuda13.2-cudnn9.22` 的 package readiness 已 clean，但 runtime smoke 被 CUDA error 35 阻塞，这种诊断是简单 P/Invoke 很难系统化表达的。

## callback 为什么更谨慎

allocator、output allocator、debug listener 这类 callback 涉及 TensorRT 调回托管代码。这里不仅是函数指针问题，还包括：

- delegate pinning。
- `GCHandle` 生命周期。
- in-flight callback 计数。
- dispose 和 release hook 顺序。
- device pointer ledger。
- stream/async 语义。
- no-throw native trampoline。
- full package consumer real runtime evidence。

因此当前安全门禁、design gate、precheck 都明确不是 proof。只有 full package consumer smoke 真实触发 TensorRT callback，并输出完整 `real-callback-runtime` markers 后，才能提升 proof。

## 结论

TensorRtSharp4.0 的工程选择可以概括为一句话：宁愿慢一点，把 ABI、生命周期、版本和诊断边界做实，也不把 C++ API 的复杂性直接丢给 C# 用户。

这条路线的代价是需要 manifest、native、interop、wrapper、smoke、package consumer 和 docs 一起维护；收益是每个可用 API 都更接近“能被用户部署、诊断和升级”的真实产品能力。
