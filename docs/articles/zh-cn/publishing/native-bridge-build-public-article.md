# Native Bridge 构建：让 ABI 边界可复核

TensorRtSharp4.0 的 native bridge 是 C# wrapper 与 CUDA/TensorRT ABI 之间的隔离层。它的价值不只是把函数导出来，更重要的是把不同 TensorRT 版本的入口、version guard、ownership 边界和 readonly diagnostics 固定在可测试的工程结构里。

公开文章里要把这件事讲透：用户自己编译 native bridge 时，应该知道 manifest 从哪里来、generated interop 落到哪里、CMake preset 对应哪条 TensorRT/CUDA 线、如何验证生成器确定性、如何排查 DLL 依赖，以及为什么 native build 不是 package-consumer-runtime proof。

## 适合

- 想理解 C# 到 TensorRT native ABI 如何分层的人。
- 需要审查 manifest、native source、generated interop 是否一致的维护者。
- 准备继续提升 deferred 边界，但不想破坏 TRT8/TRT10/TRT11 guard 的开发者。
- 想自己编译 C++ bridge DLL，再配合 NuGet 小包使用本机 CUDA/TensorRT/cuDNN 的用户。

## 构建前提

Windows 推荐环境：

```text
Visual Studio 2022
CMake >= 3.27
.NET SDK 8
PowerShell 7
CUDA Toolkit 11.8 / 12.1 / 12.9 / 13.2
TensorRT 8.6 / 10.11 / 11.0 headers and libs
cuDNN 8 or cuDNN 9
```

关键环境变量和路径要能被 CMake 找到：

```text
CUDA_PATH
TensorRT include/lib/bin
cuDNN include/lib/bin
PATH includes CUDA/TensorRT/cuDNN runtime DLL directories
```

不要把 TensorRT、CUDA、cuDNN、ONNX、engine、runtime package 或 NuGet 临时包下载到 C 盘临时目录。大型资产建议放在 E 盘固定目录，例如 `..\downloads` 或外层仓库的 `downloads` / `runtime-packages` 工作区。

## 构建链路

核心文件从 manifest 开始：

```text
native/manifests/tensorrt/v8
native/manifests/tensorrt/v10
native/manifests/tensorrt/v11
native/manifests/cuda
```

每个 manifest 记录 `module`、`versionLine`、`apis`、`id`、`entryPoint`、`returnType`、`ownership`、`manualOverride` 和 `parameters`。`eng/Test-BindingGeneratorOutputs.ps1` 会检查这些字段、API id 唯一性、entry point 唯一性和参数形状。

生成后的文件会落到：

```text
native/generated/bridge_api_catalog.g.h
native/generated/bridge_entrypoints.g.h
src/JYPPX.Shared/Generated/GeneratedApiCatalog.g.cs
src/JYPPX.Shared/Generated/GeneratedEntryPointNames.g.cs
src/JYPPX.Shared/Generated/GeneratedNativeMethods.g.cs
src/JYPPX.TensorRtSharp/Internal/Interop/Generated/GeneratedTensorRtManifestNativeMethods.g.cs
src/JYPPX.CudaSharp/Internal/Interop/Generated/GeneratedCudaManifestNativeMethods.g.cs
src/JYPPX.TensorRtSharp/Internal/Interop/Generated/NativeMethodsTensorRt.Generated.g.cs
src/JYPPX.CudaSharp/Internal/Interop/Generated/NativeMethodsCuda.Generated.g.cs
src/JYPPX.TensorRtSharp/Internal/Interop/Generated/NativeBridgeApi.TensorRtBindings.Generated.g.cs
src/JYPPX.TensorRtSharp/Internal/Interop/Generated/NativeBridgeApi.TensorRtHelpers.Generated.g.cs
src/JYPPX.CudaSharp/Internal/Interop/Generated/NativeCudaApi.Generated.g.cs
```

维护时先跑生成和一致性检查：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
```

`Test-BindingGeneratorOutputs.ps1` 会连续执行两次 `Generate-Bindings.ps1` 并比较 SHA256，确保 binding generator 输出确定性。它还会调用 `Export-NativeMethodsComparison.ps1`、`Export-WrapperLiftCandidates.ps1` 和 `Export-GeneratedApiCoverage.ps1`，把 manifest/native/generated/wrapper 的差异暴露出来。

## CMake preset

Windows native bridge preset 覆盖当前主要组合：

```text
win-x64-dev
win-x64-trt8-cuda11-release
win-x64-trt8-cuda12-release
win-x64-trt10-cuda11-release
win-x64-trt10-cuda12-release
win-x64-trt11-cuda12-release
win-x64-trt11-cuda13-release
```

Linux preset 也保留同样的版本线：

```text
linux-x64-trt8-cuda11-release
linux-x64-trt8-cuda12-release
linux-x64-trt10-cuda11-release
linux-x64-trt10-cuda12-release
linux-x64-trt11-cuda12-release
linux-x64-trt11-cuda13-release
```

如果 native 实现变化，可选择目标 preset 构建：

```powershell
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
```

其中 `CMakePresets.json` 会设置 `JYPPX_ENABLE_TENSORRT_BINDINGS`、`JYPPX_ENABLE_CUDA_BINDINGS`、`JYPPX_TENSORRT_LINE`、`JYPPX_CUDA_LINE`、`JYPPX_CUDA_VERSION`、`JYPPX_TENSORRT_CUDA_VERSION` 和 `JYPPX_CUDNN_MAJOR`。文章里应鼓励用户选择和本机 TensorRT/CUDA/cuDNN 实际安装一致的 preset，而不是随手复制最高版本命令。

## ABI 设计重点

native bridge 应优先保守处理 borrowed pointer、字符串、数组和跨 ABI 错误。基本规则是：

- C ABI entrypoint 名称稳定，不随 C# wrapper 重命名。
- 跨 ABI 不抛 C++ exception。
- native failure 转换为 `JYPPX_StatusCode` 和 last error。
- 字符串与数组输出采用 count/copy、caller buffer 或 pointer-free snapshot。
- 对象句柄使用 `SafeTensorRtObjectHandle` / `SafeCudaObjectHandle` 进入托管层。
- 版本能力通过 TRT8/TRT10/TRT11 manifest 和 source guard 表达。
- generated entrypoint 和 handwritten helper 必须保持一一对应。
- `NativeBridgePathResolver` 和 `NativeBridgeLibraryLoader` 负责运行时 bridge DLL 查找，不让用户直接处理无语义 `IntPtr`。

这也是为什么 readonly diagnostics 可以先推进，而 plugin instance create、callback trampoline、allocator ownership、borrowed pointer、external resource、runtime deserialization ownership 等能力必须继续留在设计门和 deferred 记录中。

## Bridge 实现验收表

每次把一个 deferred entrypoint 提升为真实 native 实现时，都应沿着同一张表检查，而不是只看“函数能不能导出”：

| 层级 | 必查文件 | 需要证明 |
| --- | --- | --- |
| manifest | `native/manifests/tensorrt/v8`、`native/manifests/tensorrt/v10`、`native/manifests/tensorrt/v11`、`native/manifests/cuda` | API id、entryPoint、ownership、manualOverride、versionLine 和参数方向一致。 |
| native source | `native/src/tensorrt/v8/api.cpp`、`native/src/tensorrt/v10/api.cpp`、`native/src/tensorrt/v11/api.cpp`、`native/src/cuda/api.cpp` | 每条 version guard 有对应实现或明确 unsupported status。 |
| generated headers | `native/generated/bridge_api_catalog.g.h`、`native/generated/bridge_entrypoints.g.h` | C ABI catalog 与 entrypoint 列表稳定、确定性可复现。 |
| generated C# interop | `NativeMethodsTensorRt.Generated.g.cs`、`NativeMethodsCuda.Generated.g.cs`、`NativeBridgeApi.TensorRtBindings.Generated.g.cs`、`NativeCudaApi.Generated.g.cs` | 托管 P/Invoke 签名与 native ABI 参数宽度、返回码和 string/buffer 约定一致。 |
| high-level wrapper | `src/JYPPX.TensorRtSharp`、`src/JYPPX.CudaSharp` | 用户看到的是对象、record、snapshot、enum、array、string 或 SafeHandle，不是裸 `IntPtr`。 |
| tests/evidence | `TensorRtNativeAbiSurfaceParityTests`、`PublicApiHandleExposureAuditTests`、`NativeVendorBoundaryGuardTests` | ABI surface、public API handle 暴露和 vendor boundary 没有回退。 |

如果某项只在 TRT10/11 可用，TRT8 manifest 应明确保持 unsupported 或 parse-only 语义；如果 TRT11 删除了 setter，文章和 wrapper 都应写成 readback、diagnostic 或 rejected request，而不是把旧 raw enum 继续传入新 ABI。

## Native 到 Wrapper 的提升规则

底层 interop 出现并不等于高层 API 完成。一个面向用户的能力至少要完成这几个动作：

```text
manifest entry -> native implementation -> generated interop -> NativeBridgeApi helper -> public wrapper -> smoke/quality gate -> public article
```

只读能力可以优先走 copied snapshot 路线，例如 plugin registry inventory、engine inspector layer information、parser diagnostics、builder config readback、runtime dependency diagnostics 和 CUDA device/memory/stream 状态。这类 API 的 wrapper 应返回类似 `TensorRtPluginRegistryInventory`、`TensorRtEngineInspectorReport`、`TensorRtOnnxParserDiagnosticSnapshot`、`TensorRtBuilderConfigReadback`、`CudaDeviceInfo`、`CudaMemoryInfo` 这样的值对象。

高风险能力必须继续拆开：

- callback trampoline：先证明 no-throw、exception-to-status、keep-alive、in-flight drain 和 detach-before-release。
- allocator ownership：先证明 owner ledger、attach/detach、temporary/output allocator 生命周期和 dispose 顺序。
- plugin lifecycle：先做 registry inventory、creator metadata、field metadata copied snapshot，再考虑 create/register/deregister。
- borrowed pointer：必须复制成 stable value，不把 TensorRT 指针地址作为 public API。
- external resource：必须明确谁创建、谁释放、跨线程和异常时如何回收。
- runtime deserialization ownership：必须证明 serialized buffer copied-before-interop、Engine handle owned by wrapper、plugin library dependency diagnostics 和 loadRuntime ownership 模型。

这些规则的目的不是拖慢实现，而是让每个 uplift 都可维护。没有 public wrapper 的 native entrypoint 仍然只是低层能力；没有 smoke/quality gate 的 wrapper 只能算实验入口；没有 release proof 的 runtime path 不能推动发布。

## Loader 与运行时解析

源码编译成功后，用户真正踩坑的地方通常是 DLL 搜索。native bridge 相关代码要把搜索路径表达清楚：

```text
NativeBridgePathResolver
NativeBridgeLibraryLoader
NativeBridgePathResolver.EnumerateCandidatePaths
NativeBridgeLoadException
```

推荐解析顺序是：应用输出目录、runtime package `runtimes/<rid>/native`、显式配置目录、PATH 中的 CUDA/TensorRT/cuDNN 目录。文章里不要建议用户把 DLL 复制到系统目录，也不要把一个本机 PATH 修好写成“包已经可发布”。

常见要核对的文件包括：

```text
jyppxtrtbridge.dll
jyppxcudabridge.dll
nvinfer.dll
nvinfer_plugin.dll
nvonnxparser.dll
cudart64_*.dll
cudnn*.dll
```

`NativeVendorBoundaryGuardTests` 应继续保护 vendor DLL 不被错误塞进 managed-only 包；`NativeBridgePathResolverTests` 应继续保护 resolver 能枚举 package/native/app local 路径而不要求用户写裸路径。

## 质量门

Native bridge 相关质量门包括：

```text
TensorRtNativeAbiSurfaceParityTests
PublicApiHandleExposureAuditTests
NativeBridgePathResolverTests
NativeVendorBoundaryGuardTests
SourceBuildCmakeWindowsGuideTests
PublishingRoadmapDocsTests
UpdatedObjectiveRoadmapTests
TechnicalArticleRoadmapTests
```

其中 `TensorRtNativeAbiSurfaceParityTests` 和 `PublicApiHandleExposureAuditTests` 很关键：前者保护 ABI surface 和 generated interop 对齐，后者避免 public API 泄露裸 handle / IntPtr。源码编译文章、package strategy 文章和 native bridge 文章都应该继续引用这些门禁，保证公开材料不把 ABI 风险说轻。

如果一批改动触及 native bridge，推荐最小验证组合是：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-TensorRtNativeAbiSurface.ps1
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~TensorRtNativeAbiSurfaceParityTests|FullyQualifiedName~PublicApiHandleExposureAuditTests|FullyQualifiedName~NativeBridgePathResolverTests|FullyQualifiedName~NativeVendorBoundaryGuardTests"
```

如果只改公开文章，可以只跑 `PublishingPublicArticleTests`；如果改了 manifest/native/generated/wrapper，则不能用文章测试替代 ABI 和 wrapper 门禁。

## 与两条 package 路线的关系

Native bridge 是两种公开获取通道共同的 native 交付物。GitHub Release 与 NuGet-compatible source 都只应发布：

```text
JYPPX.TensorRT.CSharp.API
JYPPX.TensorRT.CSharp.API.NativeBridge
JYPPX.CudaSharp / JYPPX.TensorRtSharp managed assemblies
```

用户自行安装 TensorRT、CUDA 和 cuDNN，并让 `NativeBridgePathResolver` 找到本机 runtime DLL。这个路线适合宣传和轻量安装，但它仍需要 clean external consumer restore/build/smoke 和 owner input validator 才能成为 package-consumer-runtime proof。

Bridge 组件成功只能说明项目自有 bridge package 可被消费；public package source、downloaded nupkg SHA256、同提交 provenance、native asset copy、主机 NVIDIA 依赖 metadata 和 runtime smoke 仍要各自有证据。Bridge 包通过不能替代 TensorRT runtime 真实执行，也不能替代 YoloVision、OnnxToEngine 或 TensorRtExec 的真实模型 proof。

## proof 边界

Native bridge build 与 readonly diagnostics 可以证明实现边界更清楚，但仍不是 package-consumer-runtime proof。以下证据都不能单独晋级：

- `Generate-Bindings.ps1` 成功。
- `Test-BindingGeneratorOutputs.ps1` 成功。
- `Export-InterfaceCoverageMatrix.ps1` 成功。
- CMake configure/build 成功。
- `dumpbin /dependents` 输出。
- native bridge DLL 存在。
- generated interop 编译通过。
- readonly diagnostics report。
- local feed package consumer。
- ProjectReference consumer。
- direct `.nupkg` install。
- GitHub Actions dry-run。
- queued workflow。
- GUI screenshot。
- package inventory ready。
- runtime readiness ready。
- public package download template。
- Bridge package local consumer passed。

真实发布 proof 必须来自公开包或候选包、干净外部 consumer、真实 runtime smoke、日志、SHA256、host metadata、owner review、post-publish verification 和 release close 审批。

## 常见排障

如果 CMake 找不到 CUDA，先检查 `CUDA_PATH` 和 Visual Studio x64 工具链。CUDA 版本要和 preset 的 `JYPPX_CUDA_VERSION` 匹配。

如果找不到 TensorRT 头文件，检查 `NvInfer.h`、`NvOnnxParser.h`、lib 和 bin 路径是否属于同一 TensorRT 版本线。不要混用 TRT10 header 和 TRT11 runtime DLL。

如果 DLL 加载失败，优先检查 `NativeBridgePathResolver.EnumerateCandidatePaths`、`PATH`、runtime package key、CUDA/TensorRT/cuDNN DLL 是否在同一位宽和同一版本族。

如果 `Test-BindingGeneratorOutputs.ps1` 报 deterministic hash 不一致，先不要手改 generated 文件；应修正 manifest、binding generator 或模板，再重新生成。

如果 public API 泄露裸 `IntPtr`，优先修 wrapper 类型设计，而不是在文章里解释为“高级用户能力”。这类泄露会破坏对象生命周期和跨版本可维护性。

## 配图建议

- 一张架构图：C# wrapper -> generated interop -> native bridge -> TensorRT/CUDA SDK。
- 一张 manifest/source/generated 对照截图。
- 一张 CMake preset 到 TensorRT/CUDA/cuDNN 版本线的矩阵图。
- 一张 proof ladder 图，标出 native build 是 build evidence，不是 package-consumer-runtime proof。

## 下一步

下一步应继续选择安全 readonly API 批量提升，尤其是 registry、creator metadata、engine inspector、parser diagnostics 和 builder config readback。每批都要保留 deferred 记录，不用删除记录制造完成度，并把 wrapper、smoke、quality test 和公开文章一起补齐。
