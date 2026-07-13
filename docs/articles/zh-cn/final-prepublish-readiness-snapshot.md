# 发布最终复验与打包发布准备快照

本页对应 `final-prepublish-readiness-snapshot`。它用于复验发布准备一致性，但不是公开发布批准，也不是 release proof。

## 当前判定

- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `publishBlockedReason=owner-real-proof-missing`
- `snapshotState=blocked-owner-real-proof-missing`

## 已复验范围

- final gates：`release-proof-owner-backfill-summary-validation`、`release-final-blocker-convergence`、`owner-real-proof-final-action-worklist`
- package readiness：managed PackageId、Description、RepositoryUrl、Authors、VersionPrefix、runtime split scripts
- docs readiness：`docs/index.md`、`docs/toc.yml` 已链接最终 gate
- sample rename readiness：`samples/YoloVision` 已取代旧检测样例名，公开材料不再使用旧 live path

## 仍然阻塞

Owner evidence 目录仍缺真实执行输入：

- `artifacts/final-release/owner-evidence/real-model-runtime`
- `artifacts/final-release/owner-evidence/package-consumer-runtime`
- `artifacts/final-release/owner-evidence/post-publish-verification`
- `artifacts/final-release/owner-evidence/release-issue-close`

因此本快照只能证明发布准备资料更完整，不能证明可以公开发布。

## 边界

template、report、matrix、article、command pack、summary pack、dry-run、build-only、sidecar-only、screenshot-only、`Skipped=True`、`blocked-by-cuda-driver`、local feed、ProjectReference、direct `.nupkg` 都不能作为 release proof。
