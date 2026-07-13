# 常见问题排查总表

本文汇总 TensorRtSharp4.0 当前最常见的构建、打包、消费端和 runtime smoke 问题。排查时先确认问题发生在哪一层：源码构建、native bridge、runtime asset collection、NuGet consumer、CUDA/TensorRT runtime，还是 callback proof。

## 快速定位

| 现象 | 最可能层级 | 首选证据 |
| --- | --- | --- |
| solution build 失败 | 托管源码或项目引用 | `dotnet build` 输出。 |
| project quality tests 失败 | API 文档、生成文件、质量规则 | `dotnet test` 输出。 |
| DocFX warning/error | 文档链接、metadata、Markdown | `dotnet docfx .\docs\docfx.json` 输出。 |
| CMake configure/build 失败 | native bridge 或本机 SDK | CMake preset 输出。 |
| native assets missing | runtime package copy | package consumer report。 |
| smoke 找不到 DLL | 输出目录或 PATH | consumer output 和 missing native asset list。 |
| CUDA error 35 | driver/runtime 兼容性 | smoke diagnostic。 |
| `0x800711C7` | Windows 应用控制 | package consumer exit code 和系统策略。 |
| callback proof false | callback evidence 未满足 schema | `RealCallbackRuntimeEvidence`。 |

## 构建与测试

常用本地门禁：

```powershell
dotnet build .\TensorRtSharp.sln -c Debug --no-restore /p:UseSharedCompilation=false
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build /p:UseSharedCompilation=false
dotnet docfx .\docs\docfx.json
```

如果修改了 manifest、native 或 generated 文件，还需要：

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
powershell -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
powershell -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
```

修改 native bridge 后还要跑对应 CMake preset。

## DLL missing 或 native-copy 不完整

先看 package consumer report 中：

- `NativeAssetsExpected`
- `NativeAssetsFound`
- `MissingNativeAssets`
- runtime package key

如果 native-copy 不完整，优先检查：

1. runtime package key 是否与本机 vendor root 匹配。
2. `pack/runtime/runtime-packages.manifest.json` 是否列出正确 DLL。
3. `eng/Collect-RuntimeAssets.ps1` 是否从正确 build preset 输出目录收集 bridge。
4. split package 的 `Bridge`、`CudaCudnn`、`TensorRt` 三个组件是否都已还原。

不要通过手工复制 DLL 掩盖 package `.targets` 或 manifest 问题。

## CUDA error 35

`cudaRuntimeGetVersion failed with CUDA error 35` 通常表示当前 NVIDIA driver 不支持正在加载的 CUDA runtime。

当前 `win-x64-trt11.0-cuda13.2-cudnn9.22` full package consumer smoke 的真实状态是：

```text
SmokeResult=blocked-by-cuda-driver
RealCallbackRuntimeEvidence.Status=blocked-by-cuda-driver
IsRealCallbackRuntimeProof=False
```

这说明 smoke 已经到达 packaged runtime 和 CUDA runtime 边界，但被当前机器 driver/runtime 兼容性阻塞。应升级驱动或换到 CUDA 13-capable 环境复测，而不是删除 deferred rows 或修改 API wrapper 来“修复”。

## Windows 应用控制

如果消费端 smoke 被 Windows Defender Application Control 或类似策略阻止，可能出现 `0x800711C7`。

可尝试：

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 `
  -RuntimePackageKey win-x64-trt11.0-cuda12.9-cudnn9.22 `
  -RunSmoke `
  -SmokeRuntimePackageKey win-x64-trt11.0-cuda12.9-cudnn9.22 `
  -SignConsumerOutput
```

如果仍失败，应把它记录为 `blocked-by-application-control`，不要写成 package layout failure。

## Readiness clean 的误读

`readiness blockers: 0` 只表示 readiness 脚本没有发现当前定义的阻塞项。它不等于：

- 所有 deferred API 都已完成。
- full package consumer runtime smoke 一定通过。
- 真实 callback runtime proof 已经完成。
- CUDA 13.2 当前机器可运行。

当前 readiness 的正确读法是：package/split/full consumer 层证据已经 clean，但 CUDA 13.2 runtime smoke 在本机被 `blocked-by-cuda-driver` 阻塞，真实 callback runtime proof 仍为 `false`。

## Callback proof false

下面这些都不是 `real-callback-runtime` proof：

- bridge-only dependency probe
- compile-only consumer
- wrapper surface compiled
- dry-run
- copied-state
- internal-runtime-prototype
- safety-gate
- design-gate
- runtime proof precheck
- 普通 `SmokeResult=passed`

真实 callback proof 必须满足 [Real Callback Runtime Evidence Schema](real-callback-runtime-evidence-schema.md)。在 proof 完成前，以下 deferred rows 必须保留：

- `IGpuAllocator::allocate`
- `IGpuAllocator::free`
- `IGpuAllocator::deallocate`
- `IGpuAllocator::reallocate`
- `IGpuAsyncAllocator::allocateAsync`
- `IGpuAsyncAllocator::deallocateAsync`
- `IOutputAllocator::notifyShape`
- `IOutputAllocator::reallocateOutput`
- `IDebugListener::processDebugTensor`

## 文档和宣传材料排查

写文档、README 或 release note 时，建议搜索这些关键词：

```powershell
rg -n "blocked-by-cuda-driver|proof=False|real callback runtime proof|readiness blockers" .\docs .\samples
```

如果文章提到 package readiness、runtime smoke 或 callback proof，必须保留当前边界，不要把 `blocked-by-cuda-driver` 改写成成功。

## 下一步阅读

- [Windows 本地开发环境准备](windows-local-dev-environment.md)
- [NuGet 消费端验证全流程](nuget-package-consumer-validation-flow.md)
- [CUDA error 35 与驱动兼容排查](cuda-error-35-troubleshooting.md)
