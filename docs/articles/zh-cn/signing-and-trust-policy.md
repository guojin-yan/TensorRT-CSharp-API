# 签名与信任策略

本文说明 TensorRtSharp4.0 在本地 RC、私有 feed、GitHub Packages、nuget.org 和 GitHub Release assets 中的签名与信任边界。

## 当前状态

当前 final release dry run 报告：

- signing/trust：`unsigned-or-not-requested`
- package consumer smoke：`blocked-by-cuda-driver`
- release status：`ready-needs-manual-approval`

这表示当前本地 RC 可以用于工程验证，但正式公开发布前仍需要 release owner 对签名、发布渠道和信任链做决策。

## 本地 RC

本地 RC 允许 unsigned output，因为它用于验证：

- managed package 是否生成。
- runtime package 是否存在。
- local feed consumer 是否能 restore/build/copy native assets。
- release readiness 是否能正确汇总 blocker/warning/manual approval。

本地 RC 的 unsigned 状态必须保留在 final release dry run 中，不能静默当作正式发布状态。

## Authenticode

Windows native bridge 或 managed executable 如果遇到 WDAC / application control 阻止，可以使用本地开发证书进行验证性签名。这类签名只适合本机或受控环境：

- 可以证明本地信任链配置可用。
- 可以绕过某些本机执行策略用于 smoke。
- 不能替代正式发布证书。
- 不能说明用户机器会信任该证书。

正式发布若需要 Authenticode，应记录：

- certificate subject。
- thumbprint。
- timestamp provider。
- root / intermediate trust。
- release owner 审批记录。

## NuGet 包签名

NuGet package signing 与 Windows Authenticode 不是同一件事。正式发布前需要决定：

- managed package 是否签名。
- runtime package 是否签名。
- split runtime component packages 是否签名。
- 使用 nuget.org 还是 GitHub Packages。
- 是否需要 repository signing 或 author signing。

当前项目不能因为本地 `.nupkg` restore 成功就宣称 NuGet 签名完成。

## 发布渠道

建议发布渠道：

- `JYPPX.TensorRT.CSharp.API` managed package：可作为 nuget.org 候选，但仍需 release owner 确认 API 文档和 package metadata。
- Bridge / collection package：可进入 GitHub Packages 或私有 feed；nuget.org 需要先审计包体积和依赖策略。
- CUDA/cuDNN/TensorRT 稳定依赖组件：优先 GitHub Packages、私有 feed 或 GitHub Release assets；正式公开前必须完成 NVIDIA 再分发条款复核。
- GitHub Release assets：适合直接下载 `.nupkg`，但 NuGet restore 不会自动发现 release assets，用户需要先加入本地 package source。

## 必须保留的边界

- `unsigned-or-not-requested` 是 manual approval，不是发布通过。
- `blocked-by-cuda-driver` 是环境阻塞，不是签名问题。
- package consumer restore/build/native-copy 通过，不代表签名策略完成。
- final release dry run 不发布包，也不修改远端 feed。

## 发布前确认清单

- [ ] release owner 确认当前版本是否允许 unsigned RC。
- [ ] 正式发布渠道确认：nuget.org、GitHub Packages、私有 feed 或 GitHub Release assets。
- [ ] NVIDIA CUDA/cuDNN/TensorRT 再分发条款完成复核。
- [ ] 包体积符合目标渠道限制。
- [ ] NuGet API key / GitHub token 权限已验证。
- [ ] 如启用签名，证书、timestamp、root trust 已记录。
- [ ] final release dry run 中 signing/trust manual approval 已处理或明确保留。
