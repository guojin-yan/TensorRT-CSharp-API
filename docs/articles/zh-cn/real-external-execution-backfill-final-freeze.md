# 真实外部执行回填与发布候选最终冻结

`artifacts/final-release/real-external-execution-backfill-final-freeze.json` 是发布候选最终冻结前的状态总表。它把 Owner proof 导入总控包、发布前最终质量门、release close blocker dashboard、release close proof lane worklist 和 release evidence closure index 串到一个入口里。

这份总表不是 proof，不会执行模型，不会发布包，也不会关闭 release issue。它只说明：当前发布候选为什么还不能冻结为可发布版本，以及 Owner 需要回填哪些真实外部执行证据。

## 当前结论

当前仍然是 `blocked-owner-action-required`：

- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isPackageConsumerRuntimeProof=false`
- `isPostPublishVerificationProof=false`

## 四条最终冻结 Lane

| Lane | 当前状态 | 需要的记录 | 阻塞原因 |
| --- | --- | --- | --- |
| real-model-runtime | blocked-owner-action-required | `real-case-evidence-record.json` | 缺少 YoloVision det/cls/seg/obb/pose/sem 真实运行证据 |
| package-consumer-runtime | template-only | `package-consumer-runtime-proof-record.json` | 缺少仓库外部 clean consumer 包消费 proof |
| post-publish-verification | template-only | `post-publish-verification-record.json` | 尚未真实公开发布 |
| release-issue-close | blocked-template-only | `release-issue-close-record.json` | 缺少 owner final close decision 和全部真实 proof |

## YoloVision Owner Delta

YoloVision 每个任务都还需要 Owner 回填：

- model source
- model license
- labels
- input tensor/image
- engine
- output artifact
- stdout/stderr log
- screenshot
- SHA256
- host OS/GPU/CUDA/TensorRT/cuDNN
- owner review

覆盖任务必须包含：

- `det`
- `cls`
- `seg`
- `obb`
- `pose`
- `sem`

现有 `YoloVisionRealAssetOwnerProofInput` 已覆盖 YOLOv8n 的 `det/seg/pose/obb/cls/sem` 六任务辅助链路，但它仍然不能替代完整 real-case proof。

这条链路只证明六任务 Owner 输入模板、候选 evidence 和 contract 没有漂移；发布冻结仍然要求真实外部执行日志、SHA256、host/package metadata、Owner review，以及 package-consumer-runtime/post-publish/release close 证据。

## Package Consumer Final Preflight Delta

package-consumer runtime proof 还必须证明：

- clean root outside repository
- consumer project exists
- public package source
- no ProjectReference
- no local feed
- no direct `.nupkg`
- managed/runtime package hash match
- smoke log exists
- smoke log SHA256 match
- `exitCode=0`
- `dependencyProbeStatus=passed`
- `smokeStatus=passed`
- `nativeAssetsCopied=true`

## 不能替代 Proof 的材料

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

只要这些真实外部执行记录没有全部通过严格 validator，发布候选就不能最终冻结为可公开发布版本。
