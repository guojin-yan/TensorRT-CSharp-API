# Package Consumer Runtime Proof：为什么必须用仓库外 Clean Consumer

## 适用读者

这篇文章适合 release owner、项目维护者、企业用户和任何需要判断“这个包是否真的可以发布”的读者。它是 proof 说明文章，不是普通安装教程。

## 解决问题

项目内部 build、local feed、ProjectReference、direct `.nupkg`、sample run、TensorRtExec report 都可能证明某个局部路径工作正常，但它们不能证明外部用户从 public package source 安装后可以运行。package-consumer-runtime proof 要回答的是：一个完全仓库外的 consumer，是否能从公开源 restore managed/runtime 包，build，并在兼容 CUDA/TensorRT host 上完成 runtime smoke。

## Clean Consumer 必须满足什么

真实 clean consumer 至少需要：

- 仓库外项目路径和项目 hash。
- public package source URL。
- managed package id/version。
- runtime package id/version/runtime key。
- restore/build/runtime smoke 命令与日志。
- stdout/stderr log SHA256。
- managed/runtime nupkg SHA256。
- OS、architecture、GPU、driver、CUDA、TensorRT、cuDNN。
- owner name、machine name、review timestamp。
- strict validator 通过。

这些字段已经被 `artifacts/final-release/clean-consumer-proof-owner-execution-pack.md` 压缩成 owner 执行包。

## 推荐执行顺序

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CleanConsumerProofOwnerExecutionPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-CleanConsumerProofOwnerExecutionPack.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-PackageConsumerRuntimeProofOwnerInput.ps1 -InputPath <owner-input.json> -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof
```

前两条命令只是准备和验证执行包；后两条只有在 owner 提供真实输入时才可能提升 proof。

## 为什么不能用替代证据

local feed 不能证明公开源可用；ProjectReference 绕过了 NuGet 包；direct `.nupkg` 绕过了 public package source；build-only 没有运行；GUI screenshot 只是界面；TensorRtExec report 是工具诊断；YoloVision matrix 和 OnnxToEngine report 是样例/转换证据。它们都可能有价值，但都不能替代 package-consumer-runtime proof。

## 边界说明

本文解释 proof 流程，本身不是 runtime proof。`build-only`、`dry-run`、`template`、local feed、ProjectReference、direct `.nupkg`、GUI screenshot、TensorRtExec report、YoloVision matrix、OnnxToEngine report、sidecar、readonly diagnostics 都不能替代 clean consumer runtime proof。

## 下一步

owner 提供真实 logs、hash 和 host metadata 后，运行 strict validator。如果 validator 仍 blocked，就按 validation finding 修复 owner input，而不是修改 dashboard 或删除 forbidden 记录。
