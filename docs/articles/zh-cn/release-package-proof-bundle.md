# Release Package Proof Bundle

`Export-ReleasePackageProofBundle.ps1` 用来把发布包相关证据收束到一个可审查的包证明视图。它读取 runtime matrix、package consumer、本地 feed consumer、runtime package readiness、runtime manifest、split runtime manifest、本地 split runtime validation 和 native asset manifest，并输出：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleasePackageProofBundle.ps1
```

如果需要 owner 审阅最终 `.nupkg` 文件清单、大小、SHA256、package id/version 和 native asset count，请继续运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalPackageReviewBundle.ps1
```

它会生成 `artifacts/final-release/final-package-review-bundle.json` 和 `.md`。该记录是 local package review，不是 public channel proof，也不会发布包。

输出文件：

- `artifacts/final-release/release-package-proof-bundle.json`
- `artifacts/final-release/release-package-proof-bundle.md`

这个 bundle 的核心边界是：

- `canUseAsPublicPackageProof=false`
- `isRuntimeExecutionProof=false`
- `isDependencyProbeOnly=true`
- `runtimeProofStatus=blocked-by-cuda-driver`
- `postPublishVerificationState=template-only`
- `postPublishCommandsReady=false`

也就是说，本地包、split 包、本地 feed、`NativeAssetsFound=19/19`、`DependencyProbe BridgeInitialized` 都是很有价值的发布包诊断证据，但它们不能替代真实外部包源证明，也不能替代 CUDA 兼容主机上的 runtime smoke proof。

## 证据来源

脚本聚合以下来源：

| 证据 | 用途 | 边界 |
| --- | --- | --- |
| `artifacts/release-candidate/runtime-package-matrix.json` | 运行时包矩阵、TRT/CUDA/cuDNN 组合 | 矩阵存在不等于已发布 |
| `pack/runtime/runtime-packages.manifest.json` | runtime key 与用户机器依赖/编译输入目录 | manifest 不是 vendor package 发布清单，也不是 public package proof |
| `pack/runtime-split/split-runtime-packages.manifest.json` | bridge-only package identity 与资产布局 | 只有 `role=bridge` 可 pack，仍需 Owner/channel 审批 |
| `artifacts/package-consumer/package-consumer-validation-summary.json` | package consumer、native-copy、smoke 分类 | driver-blocked 不是 smoke passed |
| `artifacts/local-feed-consumer/local-nuget-feed-consumer-summary.json` | 本地 feed restore/build/native-copy/dependency probe | 本地 feed 不是 nuget.org/GitHub Packages |
| `artifacts/runtime/<key>/artifact-manifest.json` | native asset collection | asset collection 不是 runtime execution proof |
| `artifacts/final-release/post-publish-verification-validation.json` | 发布后 package identity/hash、clean consumer、host metadata、commands、stdout/stderr 摘要验证快照 | template-only 或缺字段时不能作为 public package proof |
| `artifacts/final-release/final-package-review-bundle.json` | 本地 nupkg 文件、SHA256、大小、package id/version 与 native asset count | local package review 不是 public channel proof |

## 发布判断

这个 bundle 只回答“本地包证据是否足够 owner 审查”，不回答“是否可以执行公开发布”。真正发布前仍要结合：

- `release-evidence-bundle`
- `release-publish-execution-checklist`
- `release-promotion-issue-record`
- owner approval input/decision
- external runtime proof
- post-publish verification

其中 post-publish verification 只有在真实渠道发布后，且 `postPublishConsumerProjectIdentityReady=true`、`postPublishHostReady=true`、`postPublishCommandsReady=true`、`postPublishSmokeCommandRuntimeKeyReady=true`、`postPublishStdoutStderrSummaryReady=true`、managed/runtime nupkg SHA256 均 ready 时，才可能用于关闭 release issue。

在 `runtimeProofStatus=blocked-by-cuda-driver` 没有被兼容 CUDA 主机证据替换前，不允许把 package proof 写成 runtime execution proof。
