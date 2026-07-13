# 发布前最终质量门 Dashboard

`artifacts/final-release/final-prepublish-quality-gate-dashboard.json` 是发布前最后一层质量门。它把真实模型运行、包消费运行、公开发布后验证和 release issue close 四条线集中到同一个 dashboard，避免把模板、报告、矩阵、文章或 dry-run 误写成可发布 proof。

这份 dashboard 不会执行模型，不会发布包，不会批准发布，也不会关闭 release issue。它只回答一个问题：离公开发布还缺哪些真实 owner 输入。

## 当前结论

当前仍是 `blocked-owner-action-required`：

- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isPackageConsumerRuntimeProof=false`
- `isPostPublishVerificationProof=false`

原因很直接：真实 YoloVision 运行证据、仓库外部 clean consumer 包消费证据、公开发布后的 post-publish 证据、owner 最终 close 决策都还没有同时通过严格 validator。

## 四条质量门

| 质量门 | 当前状态 | Validator | 晋级前必须满足 |
| --- | --- | --- | --- |
| real-model-runtime | blocked-owner-action-required | `eng/Test-RealCaseEvidenceRecord.ps1` | YoloVision det/cls/seg/obb/pose/sem 真实模型、输入、输出、日志、截图、hash、host metadata |
| package-consumer-runtime | template-only | `eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof` | 仓库外部 clean consumer、public package source、无 ProjectReference、无 local feed、无 direct `.nupkg`、smoke passed |
| post-publish-verification | template-only | `eng/Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof` | 真实公开发布后，从 public channel 重新 restore/build/smoke 的 clean consumer 证据 |
| release-issue-close | blocked-template-only | `eng/Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady` | owner 明确最终 close 决策，并且前三条真实 proof 都已经通过 |

## Owner 下一步要补什么

### 真实模型运行

YoloVision 需要覆盖：

- `det`
- `cls`
- `seg`
- `obb`
- `pose`
- `sem`

每个任务都要提供真实模型来源、license、ONNX/engine/input/output/log/screenshot 的 SHA256、运行命令、stdout/stderr 摘要、主机 CUDA/TensorRT/cuDNN/GPU 信息和 owner review。

### 包消费运行

package-consumer proof 不能在仓库内部做，也不能用本地 feed 混过去。它必须至少证明：

- clean consumer root 在仓库外部。
- restore 来自 public package source。
- consumer project 没有 ProjectReference。
- 没有 local feed。
- 没有 direct `.nupkg` 引用。
- `exitCode=0`。
- `dependencyProbeStatus=passed`。
- `smokeStatus=passed`。
- `nativeAssetsCopied=true`。
- managed/runtime package SHA256 和 smoke log SHA256 都匹配。

### 公开发布后验证

post-publish verification 只能发生在真实公开发布之后。预发布包消费 proof、local `.nupkg`、local feed、template、draft 和 runbook 都不能替代它。

### 最终关闭

release issue close 必须等 real-model-runtime、package-consumer-runtime、post-publish-verification 都通过后，再由 owner 填写最终 close decision、release issue ID、证据 bundle SHA256 和 rollback/deprecation plan。

## 不能当 Proof 的材料

以下材料只能辅助执行和排查，不能让项目进入可发布状态：

- YoloVision matrix
- TensorRtExec report
- OnnxToEngine report
- article / roadmap
- template / template-only
- draft / runbook
- dry-run / preflight-only
- sidecar-only / build-only
- dependency-probe-only
- blocked-by-cuda-driver
- `Skipped=True`
- screenshot-only
- local feed
- ProjectReference
- direct `.nupkg`

## 最短收口路径

1. 用真实模型填 `real-case-evidence-record.json`，跑 `eng/Test-RealCaseEvidenceRecord.ps1`。
2. 用仓库外部 clean consumer 填 `package-consumer-runtime-proof-record.json`，跑 `eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof`。
3. 公开发布后填 `post-publish-verification-record.json`，跑 `eng/Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof`。
4. 最后填 `release-issue-close-record.json`，跑 `eng/Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`。

只要这四步没有全部通过，项目就应继续保持 blocked，而不是为了“看起来完成”放宽边界。
