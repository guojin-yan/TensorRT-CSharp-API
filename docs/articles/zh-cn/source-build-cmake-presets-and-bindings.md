# CMake Presets 与 Bindings 生成教程

本文解释 TensorRtSharp4.0 的 native preset、manifest、generated entrypoint 与 C# interop 的联动关系。它适合准备修改 C++ bridge、增加 deferred API 真实实现、或维护 TRT8/TRT10/TRT11 多版本路线的贡献者。

## 为什么不能跳过 bindings 生成

TensorRtSharp4.0 的可用性不等于 manifest/source 数量匹配。一个接口真正可用至少要同时满足：

- manifest 中存在真实签名，而不是 no-arg deferred 占位。
- native source 提供 no-throw C ABI 实现。
- generated bridge entrypoint 与 native export 一致。
- C# interop 有类型化声明。
- 高层 wrapper 不向用户暴露无语义 `IntPtr` 或 borrowed pointer。
- smoke/quality gate 能覆盖至少一条可执行路径或明确保持 blocked。

## 推荐命令顺序

```powershell
Set-Location .

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
```

然后再 build：

```powershell
dotnet build .\TensorRtSharp.sln -c Debug --no-restore /p:UseSharedCompilation=false
```

如果 native 改动涉及具体 TensorRT line，再运行对应 preset：

```powershell
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
```

## Preset 维护原则

| 原则 | 要求 |
| --- | --- |
| 跨版本清晰 | TRT8/TRT10/TRT11 不共享不兼容头文件或 lib |
| ABI 稳定 | C ABI entrypoint 不抛异常，错误通过状态和诊断返回 |
| 生命周期明确 | 不把 borrowed pointer 暴露给 public C# API |
| 证据可追踪 | manifest、native、generated、wrapper、test 同步更新 |
| 可发布边界 | build-only 不写成 runtime proof |

## Deferred 提升检查表

每次把 deferred API 提升为真实实现时，按这个顺序检查：

1. 用 `rg` 和 `artifacts/interface-coverage/tensorrt-interface-comparison.csv` 找到候选。
2. 优先选择只读、查询型、部署关键型 API。
3. 每批控制在 5-15 个安全 API，避免 callback、allocator、plugin instance、borrowed pointer ownership 混在同一批。
4. 更新 `native/manifests/tensorrt/v8|v10|v11`。
5. 更新 `native/src/tensorrt/common` 或对应 version source。
6. 运行 bindings 生成。
7. 更新 `src/JYPPX.TensorRtSharp/Internal/Interop` 和高层 wrapper。
8. 补 XML 注释、smoke 或 ProjectQuality 测试。
9. 运行 coverage matrix、build、targeted tests。

## 不能做的捷径

- 不删除 deferred 记录来制造完成度。
- 不只改 manifest。
- 不只改 generated 文件。
- 不把 raw `IntPtr` 作为用户可见 API。
- 不跨 ABI 抛异常。
- 模板、runbook、dashboard、dry-run 和 failedBlockerCount=0 只能作为 non-proof 辅助信号；它们不是发布 proof，也不能替代 Owner 真实证据。

## 与发布证据的关系

CMake preset、bindings 生成、build 和本地 smoke 是工程质量证据，不是公开发布 proof。公开发布仍需要真实 Owner 输入：CleanConsumer 外部日志、PostPublish 公开包日志、公开包 URL/SHA256、push transcript 或 GitHub-only reason、GitHub Release asset、rollback review 和 final close approval。
