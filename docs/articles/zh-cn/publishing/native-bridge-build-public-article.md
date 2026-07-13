# Native Bridge 构建：让 ABI 边界可复核

TensorRtSharp4.0 的 native bridge 是 C# wrapper 与 CUDA/TensorRT ABI 之间的隔离层。它的价值不只是把函数导出来，更重要的是把不同 TensorRT 版本的入口、version guard、ownership 边界和 readonly diagnostics 固定在可测试的工程结构里。

## 适合

- 想理解 C# 到 TensorRT native ABI 如何分层的人。
- 需要审查 manifest、native source、generated interop 是否一致的维护者。
- 准备继续提升 deferred 边界，但不想破坏 TRT8/TRT10/TRT11 guard 的开发者。

## 构建链路

核心文件从 manifest 开始：

```text
native/manifests/tensorrt/v8
native/manifests/tensorrt/v10
native/manifests/tensorrt/v11
```

生成后的 native 目录会落到：

```text
native/generated/bridge_api_catalog.g.h
native/generated/bridge_entrypoints.g.h
src/JYPPX.TensorRtSharp/Internal/Interop/Generated
src/JYPPX.Shared/Generated
```

维护时先跑：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
```

如果 native 实现变化，再跑：

```powershell
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
```

## 设计重点

native bridge 应优先保守处理 borrowed pointer、字符串、数组和跨 ABI 错误。字符串与数组输出采用 count/copy 或 caller buffer 模式；跨 ABI 不抛异常；版本能力通过 TRT8/TRT10/TRT11 manifest 和 source guard 明确表达。

这也是为什么 readonly diagnostics 可以先推进，而 plugin instance create、callback trampoline、allocator ownership 等能力必须继续留在安全桥接阶段。

## proof 边界

Native bridge build 与 readonly diagnostics 可以证明实现边界更清楚，但仍不是 package-consumer-runtime proof。readonly diagnostics 能帮助判断能力是否可读、是否安全、是否跨版本一致；真实发布 proof 必须来自公开包、干净外部 consumer、真实 runtime smoke、日志和 SHA256。

## 配图建议

- 一张架构图：C# wrapper -> generated interop -> native bridge -> TensorRT/CUDA SDK。
- 一张 manifest/source/generated 对照截图。
- 一张 plugin inventory readonly API 的返回结构截图，标注不是裸 `IntPtr`。

## 下一步

下一步应继续选择安全 readonly API 批量提升，尤其是 registry、creator metadata、engine inspector 和 builder config readback。每批都要保留 deferred 记录，不用删除记录制造完成度，并把 wrapper、smoke、quality test 一起补齐。
