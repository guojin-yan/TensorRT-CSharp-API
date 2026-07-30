# NuGet 与 GitHub 双通道 Bridge-only 策略

TensorRtSharp4.0 保留两种公开获取通道，但两种通道交付相同的包边界：managed C# API 与项目自有 C++ bridge。CUDA、TensorRT、cuDNN、NVRTC 及其 builtins 由用户自行安装，不进入 NuGet 包、GitHub Packages 或 GitHub Release 资产。

## 两种通道

| 通道 | 内容 | 适合用户 | 证据边界 |
| --- | --- | --- | --- |
| GitHub Release assets | managed `.nupkg`、匹配的 `.Bridge` `.nupkg`、源码归档、不可变 URL 与 GitHub SHA256 digest | 需要按 Release tag 下载固定资产的用户 | 下载资产先验证 URL、digest、包身份、nuspec 仓库提交和 bridge-only 内容，再进入隔离 restore staging |
| NuGet-compatible source | managed 包与匹配的 `.Bridge` 包 | 希望使用标准 `PackageReference` 的用户 | 记录公开 source、解析版本、下载 hash 与 restore 日志；restore/build 不是 runtime proof |

通道不同不代表包内容不同。历史 `CudaCudnn`、`TensorRt`、`CudaRtc`、collection、meta 和 full-runtime identity 只用于清理与审计，不能重新 pack、push 或上传到 Release。

## 用户安装模型

consumer 同时引用 managed 与匹配的 bridge 包，并让 probing 找到机器上的 NVIDIA runtime：

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"
$env:JYPPX_TENSORRT_ROOT = "C:\nvidia\TensorRT-10.x"
$env:JYPPX_CUDA_ROOT = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"
```

文档和包 metadata 必须明确用户负责 NVIDIA 依赖的安装、许可与版本匹配。bridge 包只能包含 `jyppxtrtbridge.dll` 或 `libjyppxtrtbridge.so`。

## GitHub Release 资产验证

GitHub Release 不是 NuGet feed。仓库执行器会下载公开 managed/bridge 资产，验证 GitHub digest 与 nupkg SHA256，把验证后的文件放入隔离目录，并仍通过 `PackageReference` restore：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Invoke-PublicReleaseBridgePackageConsumer.ps1 `
  -ManagedReleaseTag <managed-tag> `
  -BridgeReleaseTag <bridge-tag> `
  -SourceRuntimeKey win-x64-trt10.11-cuda12.9-cudnn9.22
```

managed 与 bridge 的 nuspec `repository commit` 必须一致。`-AllowCrossCommitPair` 仅用于诊断历史资产，并且永远不能晋级 public asset consumer evidence、package-consumer-runtime proof 或 post-publish proof。

## 证明边界

| 材料 | 是否可作为公开消费 proof |
| --- | --- |
| local feed 或 ProjectReference consumer | 否 |
| direct `.nupkg` / DLL 引用 | 否 |
| build-only、dependency-probe-only 或 pre-publish smoke | 否 |
| 单独的 URL、dashboard、截图或 `failedBlockerCount=0` | 否 |
| 公开 managed/bridge URL + digest + 下载 hash + 同提交 provenance + clean consumer runtime 日志 | 候选，仍需独立 validator 与 Owner 审核 |
| post-publish clean consumer、真实 host metadata、日志 hash 与 Owner 决策 | 可进入最终发布闭环 |

对外可以说明项目提供 GitHub Release 与 NuGet-compatible source 两种 managed + bridge-only 通道；在真实公开包、clean consumer、post-publish 与 Owner validator 全部通过前，不能声称发布闭环已经完成。
