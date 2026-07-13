# 发布候选包消费闭环图

`artifacts/final-release/release-candidate-package-consumer-closure-map.json` 是发布候选阶段的 proof 合同图。它把 real-model-runtime、package-consumer-runtime、post-publish verification 和 release issue close 四条线放在一起，防止模板、报告、文章和矩阵被误当成真实 proof。

这份闭环图不会发布包，也不会批准发布，更不会关闭 release issue。它只说明当前还缺哪些 owner 输入，以及每条 proof lane 必须由哪个 validator 兜底。

## 四条 Proof Track

| Track | 当前状态 | 需要什么 |
| --- | --- | --- |
| real-model-runtime | blocked-owner-action-required | YoloVision det/cls/seg/obb/pose/sem 的真实模型、输入、输出、日志、hash、host metadata 和 owner review |
| package-consumer-runtime | template-only | 仓库外部 clean consumer、public package source、无 ProjectReference、无 local feed、无 direct `.nupkg`、smoke passed 和 hash 匹配 |
| post-publish-verification | template-only | 真实公开发布之后，从 public channel 重新 restore/build/smoke 的 clean consumer 证据 |
| release-issue-close | blocked-template-only | owner 最终 close 决策，并且 real-model、package-consumer、post-publish 三条 proof 都已经通过 |

## Package Consumer 不能省略的字段

package-consumer-runtime proof 不是“本地能 build”这么简单。owner 输入至少要覆盖：

- `cleanExternalConsumerRoot`
- `consumerProjectPath`
- `publicPackageSource`
- `managedPackageId` / `managedPackageVersion` / `managedNupkgSha256`
- `runtimePackageId` / `runtimePackageVersion` / `runtimePackageKey` / `runtimeNupkgSha256`
- `ownerName` / `machineName` / `hostOs` / `hostArchitecture`
- `gpuName` / `cudaDriverVersion` / `cudaDriverSupportedRuntime` / `cudaRuntimeVersion`
- `cudnnVersion` / `tensorRtVersion` / `tensorRtLine`
- `restoreCommand` / `buildCommand` / `smokeCommand`
- `exitCode=0`
- `dependencyProbeStatus=passed`
- `smokeStatus=passed`
- `nativeAssetsCopied=true`
- `smokeLogSha256`
- stdout/stderr summary

只要这些字段还是 template、placeholder、hash 不匹配或日志不存在，就不能晋级。

## 不能替代 Proof 的材料

下面这些材料仍然是 non-proof：

- YoloVision matrix
- TensorRtExec report
- OnnxToEngine report
- local feed
- ProjectReference
- direct `.nupkg`
- dependency-probe-only
- Skipped=True
- screenshot-only
- preflight-only
- blocked-by-cuda-driver
- template
- template-only
- draft
- runbook
- article
- roadmap

这些材料可以帮助 owner 执行和排查，但不能写成 `canPublishPublicly=true` 或 `canCloseReleaseIssue=true`。

## 最短发布闭环

1. 用真实模型填 `real-case-evidence-record.json`，跑 `eng/Test-RealCaseEvidenceRecord.ps1`。
2. 在兼容主机上用外部 clean consumer 填 `package-consumer-runtime-proof-record.json`，跑 `eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof`。
3. 公开发布后填 `post-publish-verification-record.json`，跑 `eng/Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof`。
4. 最后跑 `eng/Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`。

没有这四步，就继续保持 blocked / owner-action-required。
