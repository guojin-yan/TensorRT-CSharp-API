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

当前本机已安装 CUDA `12.9`。目标为 CUDA `12.9` 的包现在使用 CUDA `12.9` 作为本地编译工具链；`win-x64-trt10.11-cuda12.9-cudnn9.22` 与 `win-x64-trt11.0-cuda12.9-cudnn9.22` 已完成本地 runtime 资产收集、runtime pack、消费端验证和消费端 smoke。

TensorRT 11 已纳入矩阵并开始真实适配。Windows `trt11.0-cuda12.9-cudnn9.22` 已完成最小原生 smoke 和消费端 smoke：logger、runtime、builder、config、network、serialized engine、deserialize 和 execution context。`trt11.0-cuda13.2-cudnn9.22` 当前可编译、可收集资产、可打包，并通过 package consumer restore/build/native-copy；但受当前驱动 CUDA 能力限制，runtime smoke 保持 pending。

当前包消费端验证：

- `win-x64-trt10.11-cuda11.8-cudnn8.9`：`16/16` 个 native assets 成功复制，消费端 smoke 通过。
- `win-x64-trt10.11-cuda12.9-cudnn9.22`：`19/19` 个 native asset patterns 成功复制，消费端 smoke 通过。
- `win-x64-trt11.0-cuda12.9-cudnn9.22`：`19/19` 个 native asset patterns 成功复制，消费端 smoke 通过。
- `win-x64-trt11.0-cuda13.2-cudnn9.22`：`19/19` 个 native asset patterns 成功复制，restore/build 通过；CUDA 13 runtime 验证仍 pending，因此未请求消费端 smoke。

## 本机 root

Windows 本机真实 root 不写入公开 manifest。请用 `pack/runtime/runtime-packages.local.json` 保存本机覆盖配置；该文件已被 Git 忽略。可从 `pack/runtime/runtime-packages.local.example.json` 复制后修改。

Windows runtime 打包前应先验证显式输入：

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Validate-WindowsRuntimeInputs.ps1 `
  -RuntimePackageKey win-x64-trt8.6-cuda11.8-cudnn8.9 `
  -TensorRtRoot "E:\TensorRtSharp\TensorRtSharp4.0\third_party\nvidia\TensorRT-8.6.1.6-cuda 11.8" `
  -CudaRoot "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8" `
  -CudnnRoot "E:\TensorRtSharp\TensorRtSharp4.0\third_party\nvidia\cudnn-windows-x86_64-8.9.7.29_cuda11-archive"
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

## 拆分交付原型

TensorRT 10 包保留 design-only 的 split-delivery 原型，位于 `pack/runtime-split`。

原型 package ID：

- `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda11.8.cudnn8.9.Core`
- `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda11.8.cudnn8.9.Extensions`
- `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Core`
- `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Extensions`

`Core` 角色承载 bridge、CUDA runtime 和核心 TensorRT runtime 库。`Extensions` 角色承载 builder resources、plugin libraries、parser libraries 等可选资产。

split 包在完成 split consumer validation、包体积策略和 NVIDIA 再分发许可复核前，不应公开发布。

## Linux 状态

Linux runtime 包条目镜像相同的 TensorRT / CUDA / cuDNN major.minor 矩阵，并保留 `.so` 通配资产规则。当前 Linux 仅做结构准备，尚未在真实 Linux runner 上验证。

当前 Linux workflow：

- `manual-build-native-linux.yml`
- `manual-pack-runtime-linux.yml`

Linux 包必须保持 `dry-run-only`，直到真实 Linux runner 完成 build、资产收集、package restore、`.so` 复制和可选 GPU smoke。

## 发布风险

runtime 包可能非常大，因为会包含 TensorRT builder resources、plugin、parser、CUDA runtime、cuBLAS 和 cuDNN。

公开发布前必须确认：

- NVIDIA TensorRT / CUDA / cuDNN 再分发许可
- NuGet.org 包体积限制
- GitHub artifact / release 托管策略
- TensorRT 10 / TensorRT 11 大包是否需要私有源或拆分交付策略
