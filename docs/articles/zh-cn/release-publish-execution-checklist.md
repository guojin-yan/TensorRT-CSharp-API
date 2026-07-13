# Release Publish Execution Checklist

`release-publish-execution-checklist` 是 release owner 真正执行发布前的清单层。它只生成 preflight、channel placeholder、rollback 和 post-publish verification 项，不会推送 NuGet、不会上传 GitHub Packages，也不会创建或修改 GitHub Release。

Owner 最短执行面以 `owner-release-execution-package` 的 `oneScreenReleaseHoldChecklist` 为准；本 checklist 镜像该一屏 Release Hold 清单的 blocker 和 validator，但仍是人工执行 guidance，不是 public approval、runtime proof 或 post-publish proof。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleasePublishExecutionChecklist.ps1
```

输出：

- `artifacts/final-release/release-publish-execution-checklist.json`
- `artifacts/final-release/release-publish-execution-checklist.md`

默认状态必须保持：

- `recordKind=release-publish-execution-checklist`
- `executionState=blocked-owner-input-required`
- `canExecutePublicPublish=false`
- `performsPublish=false`
- `requiresHumanOwner=true`

## 它检查什么

清单会读取 owner approval input validation、final release dry run、promotion issue、stale claims、package consumer smoke 和 Linux runner validation，并把这些状态汇总到发布执行视角。

它还会读取：

- `release-evidence-bundle.json`
- `final-package-review-bundle.json`
- `release-package-proof-bundle.json`
- `docs-publish-readiness-bundle.json`
- `external-runtime-proof-record-template.json`
- `external-runtime-proof-validation.json`
- `external-runtime-proof-owner-handoff.json`
- `compatible-host-runtime-proof-runbook.json`
- `owner-release-execution-package.json`
- `post-publish-verification-record-template.json`
- `post-publish-verification-validation.json`

这两个模板及其 validation artifact 用于外部 runtime proof 和发布后 clean consumer 回填；默认都是 `template-only`，不能当作 proof。

`release-evidence-bundle` 是聚合层，不能覆盖更保守的源状态；它生成成功也不会让 `canExecutePublicPublish` 变成 `true`。

`compatible-host-runtime-proof-runbook` 是兼容 GPU 主机上的执行手册。它可以进入 checklist 作为 owner-action 材料，但只要 `runbookState=owner-action-required`，就仍然不是 runtime proof，也不能让 publish checklist 变成可执行。

`release-package-proof-bundle` 负责把 runtime matrix、full/split package、本地 feed、native asset copy 和 package consumer 证据放到同一个包审阅视图；默认 `canUseAsPublicPackageProof=false` 且 `isRuntimeExecutionProof=false`。

`final-package-review-bundle` 只枚举本地 `.nupkg`、SHA256、大小、package id/version 和 native asset count，不是 public channel proof，也不会发布包。

`docs-publish-readiness-bundle` 负责统计 30+ 中文文章、sample-backed 文档、roadmap 和 DocFX 本地输出；默认 `canPublishDocsExternally=false`，只表示可进入 owner review。

关键边界：

- `blocked-by-cuda-driver` 不是 smoke passed。
- dependency probe 不是 runtime execution proof。
- `externalRuntimeProofState=template-only` 不是 runtime execution proof。
- `externalRuntimeProofRuntimePackageKeyMatches=true` 只说明 key 对齐，不代表 smoke 通过。
- `externalRuntimeProofConsumerProjectIdentityReady=false`、`externalRuntimeProofHostReady=false` 或 `externalRuntimeProofSmokeCommandRuntimeKeyReady=false` 仍然必须保持 owner action。
- `externalRuntimeProofLogSha256FormatReady=false` 或 `externalRuntimeProofLogSha256Matches=false` 必须继续显示 owner action。
- `externalRuntimeProofStdoutSummaryReady=false` 或 `externalRuntimeProofStderrSummaryReady=false` 也必须继续显示 owner action；stderr 为空时必须有 `no-stderr-emitted` 复核说明。
- release evidence bundle 不是发布批准。
- final package review bundle 不是 public package proof。
- release package proof bundle 不是 public package proof。
- docs publish readiness bundle 不是 external docs publication proof。
- `template-only` / handoff-only 不是 Linux runner proof。
- external runtime proof template 不是 runtime execution proof。
- compatible host runtime proof runbook 是命令指南，不是 runtime proof、public approval 或 package push。
- post-publish verification template 不是 clean consumer proof。
- post-publish verification 必须包含 clean consumer project identity、host CUDA/TensorRT/cuDNN metadata、带 `--runtime-package-key` 的 smoke command、stdout/stderr summary 和 SHA256-backed logs。
- `IsRealCallbackRuntimeProof=false` 不是 callback proof complete。
- publish placeholder 只是人工审核命令，不会被脚本执行。

## 渠道

清单覆盖以下渠道：

- local feed
- nuget.org
- GitHub Packages
- GitHub Release assets
- private feed

每个渠道都会生成：

- preflight
- publish placeholder
- rollback
- post-publish verification
- boundary

## 发布后验证

真实发布后必须回填 clean consumer 验证，而不是复用当前仓库 build：

1. fresh directory，无 `ProjectReference`。
2. 记录 `consumerProjectName` 与指向 `.csproj` 的 `consumerProjectPath`。
3. managed package restore 来源来自目标 channel。
4. runtime package restore 来源来自目标 channel。
5. 记录 smoke host 的 OS、GPU、driver、CUDA runtime、TensorRT line/runtime 和 cuDNN version。
6. native bridge 和 vendor runtime assets 已复制到输出目录。
7. 记录 `DependencyProbe BridgeInitialized`。
8. `smokeCommand` 必须包含 `--runtime-package-key <runtime package key>`。
9. 记录 stdout/stderr summary，并保留 restore/build/dependency/smoke log SHA256；stderr 为空时也必须写明 `no-stderr-emitted`。
10. 只有在兼容 CUDA driver/GPU host 上跑通，才写 runtime smoke passed。

## 与 promotion issue 的关系

`Export-ReleasePromotionIssueRecord.ps1` 会读取 `release-publish-execution-checklist.json`，并在 promotion item 中显示：

- `publish-execution-checklist`
- `publishExecutionChecklistState`
- `canExecutePublicPublish`
- `externalRuntimeProofRuntimePackageKeyMatches`
- `externalRuntimeProofConsumerProjectIdentityReady`
- `externalRuntimeProofSmokeCommandRuntimeKeyReady`
- `externalRuntimeProofHostReady`
- `externalRuntimeProofLogSha256FormatReady`
- `externalRuntimeProofLogSha256Matches`
- `externalRuntimeProofStdoutSummaryReady`
- `externalRuntimeProofStderrSummaryReady`
- `postPublishConsumerProjectIdentityReady`
- `postPublishSmokeCommandRuntimeKeyReady`
- `postPublishHostReady`
- `postPublishStdoutStderrSummaryReady`

默认情况下这些字段仍会阻止公开发布。只有 release owner 提供真实 approval input record、相关 proof/disposition 完整，并人工执行发布流程后，才能把发布状态从计划推进到真实发布证据。

## 与 release candidate freeze 的关系

`Export-ReleaseCandidateFreezeSummary.ps1` 和 `Export-ReleaseCandidateFreezeChecklist.ps1` 会在发布前把当前 checklist 状态冻结到：

- `artifacts/release/release-candidate-freeze-summary.json`
- `artifacts/release/release-candidate-freeze-checklist.json`
- `artifacts/release/release-candidate-freeze-validation.json`

freeze 层会复核 `canExecutePublicPublish=false`、`performsPublish=false`、`canCloseReleaseIssue=false`，并确认 publish checklist、promotion issue、release evidence、final dry run 与 post-publish validation 对 close readiness 的判断一致。

如果仍缺真实 `external-runtime-proof-record.json` 或真实 `post-publish-verification-record.json`，freeze summary 必须保持 `freezeState=blocked-freeze-owner-action-required`。这不是脚本失败，而是正确的发布前 owner-action 状态。

## 与 owner authorized publish command plan 的关系

`Export-OwnerAuthorizedPublishCommandPlan.ps1` 会在 freeze 之后生成 owner-facing 命令计划：

- `artifacts/final-release/owner-authorized-publish-command-plan.json`
- `artifacts/final-release/owner-authorized-publish-command-plan.md`
- `artifacts/final-release/owner-authorized-publish-command-plan-validation.json`
- `artifacts/final-release/owner-authorized-publish-command-plan-validation.md`

默认必须保持：

- `planState=blocked-owner-authorization-required`
- `performsPublish=false`
- `requiresExplicitOwnerAuthorization=true`
- `canMaterializeExecutableCommands=false`

命令计划会列出 `dotnet nuget push`、GitHub Packages 和 `gh release upload` 的 placeholder，但每条 `publishCommands` 都必须保持 `authorized=false`、`executable=false`、`performsPublish=false`。它只帮助 owner 审阅命令、package identity、runtime package key、SHA256、channel source 和 post-publish 回填顺序，不执行真实发布。
