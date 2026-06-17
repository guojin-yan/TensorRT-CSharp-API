# Runtime 分发策略说明

## 当前规则

runtime 包必须显式绑定 TensorRT / CUDA / cuDNN 的 major.minor 组合：

- TensorRT：例如 `trt10.11`
- CUDA：例如 `cuda12.9`
- cuDNN：例如 `cudnn9.22`

示例 runtime key：

- `win-x64-trt10.11-cuda12.9-cudnn9.22`

示例 package id：

- `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22`

完整 vendor patch/build 版本保留在 manifest 与文档中，不进入 NuGet 包名。

## Windows runtime 矩阵

当前 Windows runtime 包目标：

- `win-x64-trt8.6-cuda11.8-cudnn8.9`
- `win-x64-trt8.6-cuda12.1-cudnn8.9`
- `win-x64-trt10.11-cuda11.8-cudnn8.9`
- `win-x64-trt10.11-cuda12.9-cudnn9.22`
- `win-x64-trt11.0-cuda12.9-cudnn9.22`
- `win-x64-trt11.0-cuda13.2-cudnn9.22`

CUDA `12.9` 当前已安装。所有目标为 `cuda12.9` 的 runtime 包现在必须使用 CUDA `12.9` 构建和验证；之前的 CUDA `12.3` 临时 fallback 已废弃，不能再用于 `local-validated` 结论。

当前 Windows package-consumer readiness：

- `win-x64-trt10.11-cuda11.8-cudnn8.9`：package consumer smoke 通过。
- `win-x64-trt10.11-cuda12.9-cudnn9.22`：package consumer smoke 通过。
- `win-x64-trt11.0-cuda12.9-cudnn9.22`：package consumer smoke 通过。
- `win-x64-trt11.0-cuda13.2-cudnn9.22`：2026-06-14 已完成完整 split 组件包与 collection 包本地打包，restore/build/native-copy 通过，native asset patterns 为 `19/19`；但 CUDA 13 runtime/builder smoke 可用前 readiness 保持 blocked。

## Linux runtime 矩阵

Linux package key 必须包含发行版版本和架构。默认 hosted Linux 发布线为 Ubuntu 22.04 x64：

- `linux-x64-ubuntu22.04-trt8.6-cuda11.8-cudnn8.9`
- `linux-x64-ubuntu22.04-trt8.6-cuda12.1-cudnn8.9`
- `linux-x64-ubuntu22.04-trt10.11-cuda11.8-cudnn8.9`
- `linux-x64-ubuntu22.04-trt10.11-cuda12.9-cudnn9.22`
- `linux-x64-ubuntu22.04-trt11.0-cuda12.9-cudnn9.22`
- `linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22`

Ubuntu 24.04 x64 只建模 NVIDIA Ubuntu 24.04 仓库中存在的现代组合。Ubuntu 20.04 x64 只走 self-hosted。arm64/SBSA、Jetson/L4T、非 Ubuntu 发行版都必须作为独立包线加入，不能混用 x64 Ubuntu 包名。

`runtime-linux` workflow 支持用 `runtime_key_set` 选择发行线：

- `ubuntu22-hosted`：默认 hosted 发布线，包含 Ubuntu 22.04 x64 的 6 个组合。
- `hosted-all`：所有 hosted Linux 线，当前为 Ubuntu 22.04 x64 的 6 个组合加 Ubuntu 24.04 x64 的 3 个现代组合。
- `ubuntu24-hosted`：只发布 Ubuntu 24.04 x64 的现代组合。
- `self-hosted-ubuntu20`：只发布 Ubuntu 20.04 x64 的 self-hosted 组合，必须配合 `runner_mode=self-hosted`。
- `custom`：必须显式填写 `runtime_keys`。

如果 `runtime_keys` 非空，则以显式 key 为准；如果为空，则使用 `runtime_key_set`。这样日常发布可以保持 Ubuntu 22.04 hosted 主线，完整 hosted 发布可以切到 `hosted-all`，Ubuntu 20.04 则单独走 self-hosted。

`release-bundle` workflow 现在有两条 Linux 编排线：

- `run_linux_runtime_packaging`：hosted Linux 发布线，默认使用 `hosted-all`，会一起触发 Ubuntu 22.04 x64 与已经建模的 Ubuntu 24.04 x64 组合。
- `run_linux_self_hosted_ubuntu20_runtime_packaging`：Ubuntu 20.04 x64 self-hosted 发布线，默认使用 `self-hosted-ubuntu20`，并固定以 `runner_mode=self-hosted` 触发。

Linux split 包角色、稳定依赖版本也有单独输入。日常只改 bridge 或 managed 代码时，可以发布 Linux `bridge,collection` 并固定已发布的 `CudaCudnn` 与 `TensorRt` 版本；只有 NVIDIA 依赖集合变化时才使用 `cuda-cudnn`、`tensorrt` 或 `all` 重发稳定依赖。如果一次 dispatch 覆盖多个稳定依赖发布版本，例如 `hosted-all`，不要用一个全局版本覆盖所有 runtime key，而应使用 runtime-key 版本映射。当前 hosted Linux bridge/collection 刷新应把 `linux-x64-ubuntu22.04-*` 映射到 `4.0.6167`，把 `linux-x64-ubuntu24.04-*` 映射到 `4.0.6169`；默认 release tag 会按解析出的版本使用 `v<version>`，除非另外提供 release-tag map。较少使用的 delivery mode、稳定依赖 release tag、bridge/meta 包版本、跳过验证开关等通过 `release_config_json` 传入，避免超过 GitHub Actions `workflow_dispatch` 顶层输入数量限制。

建议优先使用 `eng/Invoke-RemoteReleaseBundle.ps1` 从工作站触发远程发布。这个脚本会把支持的顶层参数继续作为 `-f key=value` 传给 workflow，同时把高级参数自动序列化进 `release_config_json`，避免误传未声明的 workflow input。

Linux 组合保持 `dry-run-only`，等待匹配的真实 Linux runner 验证。

## 分发策略

当前建议：

- `TRT8` Windows 包可作为公开预览候选，但正式公开前仍需复核 NVIDIA 再分发许可和 NuGet.org 包体积限制。
- `TRT10` Windows 包更适合私有源或 split-delivery，因为 builder resource、plugin、parser 等资产体积较大。两条 Windows TRT10 package-consumer smoke 路径均已有 2026-06-12 本地证据。
- `TRT11` Windows CUDA `12.9` 当前作为私有源候选，已有 package-consumer smoke 证据；Windows CUDA `13.2` 在 driver/runtime-compatible smoke 可用前保持 blocked。
- Linux 包保持 dry-run 候选，等待真实 runner 验证。

## nuget.org 大小边界

nuget.org 单个包大小限制约为 `250 MB`。Windows split runtime 包需要每次发布前重新审计大小，因为 CUDA/cuDNN 和 TensorRT 组件包仍可能超过该限制。因此：

- `JYPPX.TensorRT.CSharp.API` managed 包可以发布到 nuget.org。
- 体积较小的 `Bridge` 和 collection 包可以在需要时发布到 nuget.org 或 GitHub Packages。
- CUDA/cuDNN 和 TensorRT 稳定依赖组件包多数不适合 nuget.org；如果需要 NuGet feed 自动 restore，应优先放 GitHub Packages；如果可以直接下载 `.nupkg` 文件，则可以保留为 GitHub Release assets。
- GitHub Release assets 不会被 NuGet restore 自动查询。稳定依赖包只放 Release 时，验证和用户消费前都需要先把匹配 `.nupkg` 下载到本地 package source。
- 后续如果只修改本地 C ABI bridge 或 C# wrapper，重发 `Bridge`、collection 和 managed 包即可，不需要重发 `CudaCudnn` 或 `TensorRt` 包，除非对应 NVIDIA 依赖集合变化。
- 当 collection 包要引用不同 runtime key 下不同版本的稳定依赖包时，使用 `cuda_cudnn_package_version_map` 和 `tensorrt_package_version_map`，不要使用单个全局版本。

## 工程规则

`pack/runtime/runtime-packages.manifest.json` 中每个 runtime 包必须记录：

- `tensorRtVersion`
- `cudaVersion`
- `cudnnVersion`
- `cudnnMajor`
- `distributionTier`
- `validationState`
- `distributionNotes`

校验规则：

- runtime key 和 package id 必须包含 TensorRT / CUDA / cuDNN 的 major.minor 片段。
- 完整 vendor patch 版本只写入 manifest 和文档，不进入 package id。
- CUDA `12.9` 目标包只有在确实使用 CUDA `12.9` 工具链和匹配 TensorRT/cuDNN 资产验证后，才允许标记为 `local-validated`。
- private-feed 和 split-delivery readiness 必须要求 `local-validated`；`pending-local-validation` 和 `dry-run-only` 必须保持 blocked。
- NVIDIA 二进制依赖不能提交到 Git。

相关脚本：

- `eng/Validate-RuntimeManifest.ps1`
- `eng/Validate-WindowsRuntimeInputs.ps1`
- `eng/Collect-RuntimeAssets.ps1`
- `eng/Export-RuntimeDistributionReport.ps1`
- `eng/Export-RuntimeDeliveryStrategy.ps1`
- `eng/Test-RuntimePublishReadiness.ps1`
- `eng/Export-ReleaseCandidateChecklist.ps1`

## Runtime 组件拆分

split runtime 模型适用于体积较大、或者不应跟随 managed 代码频繁重发的 Windows runtime 组合：

- `Bridge`：本地 C ABI bridge。只有 native wrapper 代码变化时重发。
- `CudaCudnn`：CUDA runtime、cuDNN 和相关共享资产。只有 CUDA/cuDNN 依赖集合变化时重发。
- `TensorRt`：TensorRT runtime、parser、plugin 和 builder-resource 资产。只有 TensorRT 依赖集合变化时重发。
- collection 包：保留原始 runtime package ID，用来声明一组已验证的组件版本组合。

managed 包可以和这些 runtime 组件包独立发版。日常 C# 或 bridge 改动只需要发布 managed 包、`Bridge` 和 collection 包，并固定已有 `CudaCudnn` 和 `TensorRt` 包版本。
