# 真实 Owner Proof 导入总控包

`artifacts/final-release/owner-real-proof-import-master-pack.json` 是给 Owner 使用的真实 proof 导入总控包。它把发布前必须补齐的四类真实记录放到一个入口里：真实模型运行、仓库外部包消费、公开发布后验证和最终 release issue close。

这份总控包不是 proof，不会执行模型，不会发布包，也不会关闭 release issue。它的作用是减少来回翻文件，让 Owner 一次性知道该填什么、从哪个模板开始、运行哪个导入命令和 validator。

## 四个必须补齐的记录

| 记录 | 质量门 | 当前状态 | 校验命令 |
| --- | --- | --- | --- |
| `real-case-evidence-record.json` | real-model-runtime | blocked-owner-action-required | `eng/Test-RealCaseEvidenceRecord.ps1 -FailOnNotProof` |
| `package-consumer-runtime-proof-record.json` | package-consumer-runtime | template-only | `eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof` |
| `post-publish-verification-record.json` | post-publish-verification | template-only | `eng/Test-PostPublishVerificationRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof` |
| `release-issue-close-record.json` | release-issue-close | blocked-template-only | `eng/Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady` |

## YoloVision 辅助链路边界

当前仓库里已有 `Import-YoloVisionRealAssetOwnerProofInput.ps1` 和 `Test-YoloVisionRealAssetOwnerProofInput.ps1`，这条辅助链路覆盖 `det/seg/pose/obb/cls/sem` 六任务 Owner input candidate。与这条可选候选链路分开，`samples/assets` 已提交官方 YOLOv8n det/cls/seg/pose/obb 和 torchvision LRASPP sem 的六份 source-tree real-model-runtime evidence；`Export-YoloVisionSixTaskRealProofChainDashboard.ps1` 会 fail-closed 校验这些记录，当前结果为 `6/6` source-tree runtime ready。

六任务源码树记录仍然不能替代完整 release proof：它们不等于 `real-case-evidence-record.json` 已被 Owner 最终准入链正式导入和接受，也不证明 clean package-consumer-runtime、public package、post-publish-verification 或 release issue close。最终发布门中的 `real-model-runtime-owner-proof-required` 表示 Owner 准入导入仍待执行，不表示六个源码树真实模型案例缺失。

它可以帮助收集完整六任务 owner 输入，但不能替代完整的 YoloVision `det/cls/seg/obb/pose/sem` 真实运行 proof。完整发布 proof 仍必须落到 `real-case-evidence-record.json` 并通过 `Test-RealCaseEvidenceRecord.ps1 -FailOnNotProof`。

## Package Consumer 导入边界

包消费 proof 必须来自仓库外部 clean consumer：

- clean root 在仓库外部。
- restore 使用 public package source。
- consumer project 不包含 ProjectReference。
- 不允许 local feed。
- 不允许 direct `.nupkg`。
- smoke log 必须存在并且 SHA256 匹配。
- managed/runtime package SHA256 必须匹配。
- `exitCode=0`。
- `dependencyProbeStatus=passed`。
- `smokeStatus=passed`。
- `nativeAssetsCopied=true`。

## 公开发布后验证

post-publish verification 只能在真实公开发布之后执行。预发布本地包、local feed、模板、候选记录和 runbook 都不能替代 post-publish proof。

## 最终关闭

release issue close 必须等 real-model-runtime、package-consumer-runtime、post-publish-verification 三条 proof 全部通过，然后由 Owner 明确填写 final close decision、release issue ID、证据 bundle SHA256 和 rollback/deprecation plan。

## 不能当 Proof 的材料

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

只要这些真实记录没有全部通过，项目就必须继续保持 `canPublishPublicly=false` 和 `canCloseReleaseIssue=false`。
