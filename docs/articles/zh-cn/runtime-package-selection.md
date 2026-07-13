# Runtime Package 和 Split Package 怎么选

TensorRtSharp4.0 的托管主包和 native runtime 资产是分开的。托管主包提供 C# API，runtime package 提供某个 TensorRT/CUDA/cuDNN 组合所需的 native bridge 和 NVIDIA 依赖。

选择包时，核心原则是：runtime key 必须和目标机器的 TensorRT/CUDA/cuDNN major.minor 组合一致。

## 包类型

常见包可以分成四类：

| 包 | 作用 | 何时变化 |
| --- | --- | --- |
| `JYPPX.TensorRT.CSharp.API` | 托管主包，包含 C# wrapper | C# API 或 managed assembly 变化 |
| `Bridge` split package | 本地 C ABI bridge | native bridge 代码变化 |
| `CudaCudnn` split package | CUDA runtime、cuDNN 和相关依赖 | NVIDIA CUDA/cuDNN 依赖集合变化 |
| `TensorRt` split package | TensorRT runtime、parser、plugin、builder resources | NVIDIA TensorRT 依赖集合变化 |

collection package 使用原始 runtime package ID，用来固定一组已验证的 split component 版本。full runtime package 则把 bridge、CUDA/cuDNN、TensorRT 资产放在一个大包里，更适合本地完整验证，不一定适合公开 NuGet 分发。

## 运行时 key 怎么读

例如：

```text
win-x64-trt11.0-cuda13.2-cudnn9.22
```

含义是：

- Windows x64。
- TensorRT 11.0。
- CUDA 13.2。
- cuDNN 9.22。

对应 package ID：

```text
JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda13.2.cudnn9.22
```

完整 vendor patch 版本不放进包名，而是保留在 manifest 和文档里，例如 TensorRT `11.0.0.114`。

## 什么时候选 split package

split package 适合大体积运行时依赖，特别是 CUDA/cuDNN/TensorRT 稳定依赖不希望每次 managed 或 bridge 改动都重新发布的场景。

典型使用方式：

1. managed 包跟随 C# API 更新。
2. `Bridge` 包跟随 native bridge 更新。
3. `CudaCudnn` 和 `TensorRt` 包只有 NVIDIA 依赖集合变化时才更新。
4. collection 包固定当前组合中各组件版本。

这样既能让用户恢复出完整 runtime，又避免反复重发几百 MB 到数 GB 的稳定依赖包。

## 什么时候选 full runtime package

full runtime package 更适合：

- 本地完整验证。
- 私有源内部分发。
- package consumer smoke。
- readiness 审计。

它的缺点是体积大，可能不适合 nuget.org。正式公开发布前还需要复核 NVIDIA 二进制再分发许可和目标 feed 的包大小限制。

## 当前 TRT11/CUDA13.2 状态

当前 `win-x64-trt11.0-cuda13.2-cudnn9.22` 的 readiness 证据为：

- managed package：ready。
- bridge package：ready。
- split components：ready 3/3。
- split collection package：ready。
- full runtime package：ready。
- vendor blockers：none。
- readiness blockers：0。

但 full package consumer runtime smoke 在当前机器被 CUDA driver/runtime compatibility 阻塞为 `blocked-by-cuda-driver`。这表示包布局和 native asset copy 已经通过，TensorRT bridge 也能启动，但当前机器驱动无法完成 CUDA 13.2 runtime smoke。

## 选择建议

如果你只是本地开发和验证：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Invoke-LocalRuntimePackage.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22
```

如果你要验证 split delivery：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 -SourceRuntimeKey win-x64-trt11.0-cuda13.2-cudnn9.22 -SplitPackageRole all
```

如果你要验证消费端：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22
```

如果要请求 runtime smoke：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -RunSmoke -AllowSmokeFailure
```

若得到 `blocked-by-cuda-driver`，应先检查驱动/runtime 兼容性，而不是调整 API wrapper 或删除 deferred rows。
