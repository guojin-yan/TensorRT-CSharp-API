# 真实 Owner Proof 回填后发布判定

本页对应 `real-owner-proof-postback-release-decision`。它只记录 Owner 真实 proof 是否已经回填，以及当前是否可以重新评估公开发布。

## 当前判定

- `decisionState=blocked-owner-proof-not-posted-back`
- `publishBlockedReason=owner-real-proof-missing`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

## 判定依据

当前四条 proof lane 的 Owner evidence 目录仍未回填真实材料：

- `artifacts/final-release/owner-evidence/real-model-runtime`
- `artifacts/final-release/owner-evidence/package-consumer-runtime`
- `artifacts/final-release/owner-evidence/post-publish-verification`
- `artifacts/final-release/owner-evidence/release-issue-close`

因此本阶段不运行发布、不提升 lane、不关闭 release issue。

## 下一步

Owner 必须先回填真实运行日志、record、SHA256、host metadata 和 strict validator output。只有证据出现后，才允许运行对应 strict validators 并重新判定。
