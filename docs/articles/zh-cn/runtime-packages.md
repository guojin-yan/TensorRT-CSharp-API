# Runtime 包说明

runtime 包用于承载某一个明确 TensorRT / CUDA / cuDNN 组合的原生部署资产：

- JYPPX 原生桥接库
- 匹配的 CUDA runtime 动态库
- 匹配的 TensorRT 动态库
- 匹配的 cuDNN 动态库
- TensorRT 8 parser/plugin 等场景需要的 cuBLAS 等可选部署依赖

## 命名规则

runtime package key 和 NuGet package ID 必须包含依赖的 `major.minor` 版本：

- runtime key：`win-x64-trt10.11-cuda11.8-cudnn8.9`
- package ID：`JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda11.8.cudnn8.9`

公开 manifest 中仍然保留完整厂商版本，例如 TensorRT `10.11.0.33`、cuDNN `8.9.7.29`。这样包名不会过长，同时仍能追踪精确二进制来源。

不要继续使用 `win-x64-trt10-cuda11` 或 `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.cuda11` 这类有歧义的包名。

## Windows 组合

当前 Windows runtime 目标组合：

- `win-x64-trt8.6-cuda11.8-cudnn8.9`：TensorRT `8.6.1.6`，CUDA `11.8`，cuDNN `8.9.7.29`
- `win-x64-trt8.6-cuda12.1-cudnn8.9`：TensorRT `8.6.1.6`，CUDA `12.1`，cuDNN `8.9.7.29`
- `win-x64-trt10.11-cuda11.8-cudnn8.9`：TensorRT `10.11.0.33`，CUDA `11.8`，cuDNN `8.9.7.29`
- `win-x64-trt10.11-cuda12.9-cudnn9.22`：TensorRT `10.11.0.33`，CUDA `12.9`，cuDNN `9.22.0`
- `win-x64-trt11.0-cuda12.9-cudnn9.22`：TensorRT `11.0.0.114`，CUDA `12.9`，cuDNN `9.22.0`
- `win-x64-trt11.0-cuda13.2-cudnn9.22`：TensorRT `11.0.0.114`，CUDA `13.2`，cuDNN `9.22.0`

稳定的 TensorRT 10 / CUDA 11.8 路径已有当前消费端证据。2026-06-12，`win-x64-trt10.11-cuda11.8-cudnn8.9` 已从本地包源还原、构建消费端项目、复制 `16/16` 个 native assets，并通过 smoke；探针输出 TensorRT `10.11.0`、CUDA `11.8`、CUDA 设备数 `1`。

当前维护环境已安装 CUDA `12.9`。目标为 CUDA `12.9` 的包现在使用 CUDA `12.9` 作为本地编译工具链；`win-x64-trt10.11-cuda12.9-cudnn9.22` 与 `win-x64-trt11.0-cuda12.9-cudnn9.22` 已完成本地 runtime 资产收集、runtime pack、消费端验证和消费端 smoke。

TensorRT 11 已纳入矩阵并开始真实适配。Windows `trt11.0-cuda12.9-cudnn9.22` 已完成最小原生 smoke 和消费端 smoke：logger、runtime、builder、config、network、serialized engine、deserialize 和 execution context。2026-06-14，`trt11.0-cuda13.2-cudnn9.22` 已完成 native bridge 构建、完整 split 组件包与 collection 包打包，并通过 package consumer restore/build/native-copy；但当前驱动报告 CUDA `12.9`，不是 CUDA 13-capable runtime stack，因此 runtime smoke 保持 pending。

当前包消费端验证：

- `win-x64-trt10.11-cuda11.8-cudnn8.9`：`16/16` 个 native assets 成功复制，消费端 smoke 通过。
- `win-x64-trt10.11-cuda12.9-cudnn9.22`：`19/19` 个 native asset patterns 成功复制，消费端 smoke 通过。
- `win-x64-trt11.0-cuda12.9-cudnn9.22`：`19/19` 个 native asset patterns 成功复制，消费端 smoke 通过。
- `win-x64-trt11.0-cuda13.2-cudnn9.22`：2026-06-14 完整 split 包打包通过，`19/19` 个 native asset patterns 成功复制，restore/build 通过；当前驱动仅报告 CUDA `12.9`，因此未请求 CUDA 13 消费端 smoke。

## 本机 root

Windows 本机真实 root 不写入公开 manifest。请用 `pack/runtime/runtime-packages.local.json` 保存本机覆盖配置；该文件已被 Git 忽略。可从 `pack/runtime/runtime-packages.local.example.json` 复制后修改。

Windows runtime 打包前应先验证显式输入：

```powershell
$roots = powershell -ExecutionPolicy Bypass -File .\eng\Resolve-RuntimeRoots.ps1 `
  -RuntimePackageKey win-x64-trt8.6-cuda11.8-cudnn8.9 | ConvertFrom-Json

powershell -ExecutionPolicy Bypass -File .\eng\Validate-WindowsRuntimeInputs.ps1 `
  -RuntimePackageKey win-x64-trt8.6-cuda11.8-cudnn8.9 `
  -TensorRtRoot $roots.tensorRtRoot `
  -CudaRoot $roots.cudaRoot `
  -CudnnRoot $roots.cudnnRoot
```

## 资产收集

资产收集脚本和 manifest：

- `eng/Collect-RuntimeAssets.ps1`
- `pack/runtime/runtime-packages.manifest.json`

当前原生桥接库输出目录：

- `build-out/<preset>/bin/<Configuration>/`
- `build-out/<preset>/lib/<Configuration>/`

桥接库按 CMake preset 隔离输出，runtime 打包必须按 manifest 中对应的 `buildPreset` 收集。

TensorRT 8 Windows runtime 包会额外收集 parser/plugin 依赖：

- `win-x64-trt8.6-cuda11.8-cudnn8.9`：`cublas64_11.dll`、`cublasLt64_11.dll`、完整 `cudnn*_8.dll` 拆分运行时集合
- `win-x64-trt8.6-cuda12.1-cudnn8.9`：`cublas64_12.dll`、`cublasLt64_12.dll`、完整 `cudnn*_8.dll` 拆分运行时集合

TensorRT 11 Windows 包布局与旧版本不同：DLL 位于 `bin`，导入库位于 `lib`。runtime manifest 收集的是 DLL。

## Runtime 组件拆分

Windows runtime 包在 `pack/runtime-split` 下拆成组件包。

组件角色：

- `Bridge`：只承载本地 C ABI bridge。
- `CudaCudnn`：承载某一组 NVIDIA CUDA/cuDNN 依赖对应的 CUDA runtime、cuDNN 和相关共享资产。
- `TensorRt`：承载某一组 NVIDIA TensorRT 依赖对应的 TensorRT runtime、parser、plugin 和 builder resource 资产。
- 原始 runtime package ID 保留为轻量 collection 包，用来固定一组已验证的组件版本组合。

`CudaCudnn` 和 `TensorRt` 包版本不需要和 managed 包版本一致。只有对应的 NVIDIA 依赖集合变化时才重发；本地 native bridge 变化时重发 `Bridge` 和 collection 包，同时固定已有 `CudaCudnn` 和 `TensorRt` 包版本。

## Linux 状态

Linux runtime 包名必须包含发行版版本和 CPU 架构，因为 NVIDIA 针对不同系统和架构发布不同 apt 仓库和二进制集合。不要再发布泛化的 `linux-x64-trt...` 包；应使用类似 `linux-x64-ubuntu22.04-trt11.0-cuda12.9-cudnn9.22` 的明确 key。

当前 Linux 矩阵：

- Ubuntu 22.04 x64：默认 hosted 矩阵，覆盖全部 6 个 TensorRT / CUDA / cuDNN 组合。
- Ubuntu 24.04 x64：hosted 只覆盖 `trt10.11-cuda12.9-cudnn9.22`、`trt11.0-cuda12.9-cudnn9.22`、`trt11.0-cuda13.2-cudnn9.22`；NVIDIA 官方 Ubuntu 24.04 仓库不提供旧的 TensorRT 8.6 / CUDA 11.8 组合。
- Ubuntu 20.04 x64：只作为 self-hosted/manual-root 路线，覆盖 NVIDIA 仓库里仍存在的 TensorRT 8.6 与 TensorRT 10.11 旧组合；不使用 GitHub-hosted Ubuntu 20.04。
- Linux arm64/SBSA 和 Jetson/L4T 后续要单独建包线。SBSA 服务器 ARM 和 Jetson 不是同一个运行时目标，不能复用 x64 Ubuntu 包名。
- RHEL/Rocky 等其它发行版只有在明确建模对应 NVIDIA 仓库和 runner 镜像后才能加入。

当前 Linux workflow 模块：

- `runtime-linux.yml`
- `release-bundle.yml`

Linux 包保持 `dry-run-only`，直到匹配的 runner 完成 build、资产收集、package restore、`.so` 复制和可选 GPU smoke。

## 发布风险

runtime 包可能非常大，因为会包含 TensorRT builder resources、plugin、parser、CUDA runtime、cuBLAS 和 cuDNN。

当前发布策略：

- `JYPPX.TensorRT.CSharp.API` 发布到 nuget.org 和 GitHub Packages。
- 大体积 CUDA/cuDNN/TensorRT 组件包优先发布到 GitHub Packages；如果不适合 NuGet feed，则作为 GitHub Release asset 发布。
- GitHub Release assets 只是可下载的 `.nupkg` 文件，不是 NuGet feed。稳定依赖包只保留在 Release 时，发布 workflow 会先下载这些文件到临时本地包源，再验证 `bridge,collection`。
- runtime 包版本和 managed 包版本独立维护。
- 只有对应的 NVIDIA 依赖集合变化时，才重发 `CudaCudnn` 或 `TensorRt` 包。
- 本地 C ABI bridge 变化时，重发 `bridge,collection` split 包，并显式传入已有 `CudaCudnn` 和 `TensorRt` 包版本，避免重复发布稳定依赖包。

公开发布前仍需针对实际发布的 NVIDIA TensorRT / CUDA / cuDNN 二进制文件复核再分发许可。

