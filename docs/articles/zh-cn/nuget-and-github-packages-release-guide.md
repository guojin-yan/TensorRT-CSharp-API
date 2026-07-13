# NuGet 与 GitHub Packages 发布指南

本文面向 release owner，说明 TensorRtSharp4.0 如何选择 package source、如何验证 local feed、以及正式发布前哪些证据必须齐全。

## 发布前顺序

建议先完成本地 dry run：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateChecklist.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCandidateReadiness.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -AllowRuntimeSmokeBlocked
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-UserAcceptanceSampleCatalog.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalReleaseDryRun.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -AllowRuntimeSmokeBlocked -WarnOnly
```

如果 final release dry run 不是 `ready`，应先处理 blocker 或 manual approval。

正式发布脚本不会自动执行 `dotnet nuget push`、GitHub Packages 上传、GitHub Release 上传、delete、delist 或 withdraw。相关命令只应作为 release owner 审核后的人工 checklist item 出现。

## Local Feed

local feed 是正式发布前最重要的用户模拟路径：

- 只使用 `.nupkg`。
- 禁止 `ProjectReference`。
- 验证 managed package。
- 验证 runtime package。
- 验证 native assets copy。

命令：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LocalNuGetFeedConsumer.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -AllowSmokeFailure
```

当前已知状态是 `dependency-probe-passed`，这不是 full runtime smoke passed。

## nuget.org

nuget.org 适合发布 managed package 和体积较小、再分发许可清晰的包。注意：

- nuget.org 有包体积限制。
- NVIDIA 组件再分发条款必须先复核。
- `NUGET_API_KEY` 必须是有效、未过期、有目标 package push 权限的 key。
- 如果 nuget.org 返回 `403`，不要重试发布；应替换或修复 secret 权限。

## GitHub Packages

GitHub Packages 适合：

- 私有或受控 feed。
- runtime split packages。
- 较大的 Bridge / collection / dependency component packages。
- 内部验证和 RC 分发。

用户需要在 `NuGet.config` 中配置 GitHub Packages 源和凭据。

## GitHub Release Assets

GitHub Release assets 适合直接下载 `.nupkg`，但不适合作为 NuGet restore 自动源。用户必须先下载对应 `.nupkg` 并加入本地 package source。

## Runtime Package 选择

不要只看最新 release tag。runtime package key 必须匹配：

- RID / platform。
- TensorRT major.minor。
- CUDA major.minor。
- cuDNN major.minor。

示例：

```text
win-x64-trt11.0-cuda13.2-cudnn9.22
```

当前该 key 的 package/native-copy evidence 可用，但 full runtime smoke 在当前机器上是 `blocked-by-cuda-driver`。

## 发布后 Proof

发布到 nuget.org、GitHub Packages、GitHub Release assets 或 private feed 后，必须重新建立 clean consumer，并填写 `post-publish-verification-record`。真实 proof 至少要包含：

- managed/runtime package id、version、channel URL 和下载后 nupkg SHA256。
- `consumerProjectName` 和指向 clean consumer `.csproj` 的 `consumerProjectPath`。
- `restoreCommand`、`buildCommand`、`smokeCommand`，其中 `smokeCommand` 必须包含 `--runtime-package-key <runtime package key>`。
- smoke host 的 OS、GPU、driver、CUDA driver/runtime、TensorRT runtime/line 和 cuDNN version。
- restore/native asset/dependency probe/smoke log 的路径与 SHA256。
- `stdoutSummary` 与 `stderrSummary`。

template、draft、local bin output、`ProjectReference`、dependency-probe-only 或 `blocked-by-cuda-driver` 都不能关闭 release issue。

## 回滚

发布出错时应先判断渠道：

- nuget.org：已发布版本通常不能覆盖，只能 unlist 或发布修复版本。
- GitHub Packages：可删除/替换策略取决于组织权限和 retention 设置。
- GitHub Release assets：可删除资产或发布修正 release notes。
- 私有 feed：按组织 feed 策略回滚。

回滚说明也应进入 release notes 或维护日志。任何 delist/delete/withdraw 都必须是 release owner 的显式决策，脚本只记录候选操作和证据，不自动执行。
