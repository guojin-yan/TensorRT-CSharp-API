# 发布候选冻结与 Owner 授权边界

`release-candidate-freeze` 是发布前最后一层 owner-facing 汇总。它把 release evidence、final dry run、package proof、owner approval、owner decision、publish checklist、promotion issue、external runtime proof 与 post-publish verification 的当前状态冻结成同一份证据视图。

该阶段只做审阅和门禁，不执行发布。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateFreezeSummary.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateFreezeChecklist.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCandidateFreezeSummary.ps1
```

输出：

- `artifacts/release/release-candidate-freeze-summary.json`
- `artifacts/release/release-candidate-freeze-summary.md`
- `artifacts/release/release-candidate-freeze-checklist.json`
- `artifacts/release/release-candidate-freeze-checklist.md`
- `artifacts/release/release-candidate-freeze-validation.json`
- `artifacts/release/release-candidate-freeze-validation.md`

## Owner 一屏清单入口

当前 owner 最短执行面以 `artifacts/final-release/owner-release-execution-package.json` 的 `oneScreenReleaseHoldChecklist` 为准；可读说明见 `docs/articles/zh-cn/owner-release-execution-package.md` 与 `artifacts/final-release/owner-release-execution-package.md`。`release-candidate-freeze-summary` 会镜像 owner authorization、`package-consumer-runtime`、Linux runner proof、`real-model-runtime` 和 post-publish verification 这 5 项 Release Hold 清单，但它仍是 guidance，不是 proof，也不会让 `canCloseReleaseIssue=false` 变成可关闭状态。

`artifacts/final-release/release-proof-readiness-snapshot.json` 是同一组 5 个 blocker 的紧凑状态视图，面向 owner 最终判断。它只回答当前是否具备真实 proof，不执行发布、不上传包、不关闭 release issue。

## 默认状态

在缺少真实 compatible-host package-consumer runtime proof、owner approval、owner decision 与真实 post-publish proof 时，默认状态必须保持：

- `recordKind=release-candidate-freeze-summary`
- `freezeState=blocked-freeze-owner-action-required`
- `canPublish=false`
- `canPromote=false`
- `canCloseReleaseIssue=false`
- `performsPublish=false`
- `requiresHumanOwner=true`
- `realExternalRuntimeProofReady=false`
- `realPostPublishVerificationReady=false`
- `closeReadinessConsistent=true`
- `externalRuntimeProofBackfillPlanState=blocked-compatible-host-proof-required`
- `externalRuntimeProofBackfillCanPromoteRuntimeProof=false`
- `postPublishVerificationBackfillPlanState=blocked-real-post-publish-proof-required`
- `postPublishVerificationBackfillCanCloseReleaseIssue=false`

`closeReadinessConsistent=true` 只表示 release evidence、final dry run、publish checklist、promotion issue 与 post-publish validation 对关闭状态的判断一致。它不是 release issue 可关闭证明；在真实 post-publish proof 缺失时，`canCloseReleaseIssue` 必须仍为 `false`。

## Freeze Summary 聚合内容

`Export-ReleaseCandidateFreezeSummary.ps1` 读取并聚合：

- `artifacts/final-release/release-evidence-bundle.json`
- `artifacts/final-release/owner-release-execution-package.json`
- `artifacts/final-release/release-proof-readiness-snapshot.json`
- `artifacts/final-release/final-release-dry-run-summary.json`
- `artifacts/final-release/release-package-proof-bundle.json`
- `artifacts/final-release/release-owner-approval-input-validation.json`
- `artifacts/final-release/release-owner-decision-record.json`
- `artifacts/final-release/release-publish-execution-checklist.json`
- `artifacts/final-release/release-promotion-issue-record.json`
- `artifacts/final-release/external-runtime-proof-validation.json`
- `artifacts/final-release/external-runtime-proof-backfill-plan.json`
- `artifacts/final-release/external-runtime-proof-backfill-plan.md`
- `artifacts/final-release/post-publish-verification-validation.json`
- `artifacts/final-release/post-publish-verification-backfill-plan.json`
- `artifacts/final-release/post-publish-verification-backfill-plan.md`
- `artifacts/final-release/stale-release-claims-audit.json`
- `artifacts/package-consumer/package-consumer-validation-summary.json`
- `artifacts/linux-dry-run/<runtime-key>/linux-runner-evidence-validation.json`

summary 会输出 `blockingItems` 与 `artifactRefs`。其中必须长期保留以下阻断项，直到真实 owner proof 出现：

- `real-external-runtime-proof`
- `external-runtime-proof-backfill-required`
- `owner-approval`
- `owner-decision`
- `release-evidence-complete`
- `publish-checklist-authorized`
- `post-publish-verification`
- `post-publish-verification-backfill-required`
- `close-readiness-consistency`
- `stale-release-claims`
- `blocked-cuda-driver-visible`

## Freeze Checklist

`Export-ReleaseCandidateFreezeChecklist.ps1` 生成 owner-facing 清单，面向人工决策，而不是自动执行。

清单包含：

- `ownerDecisionItems`：owner 需要逐项审阅和处理的发布前事项。
- `publishPreflight`：发布前必须确认的事项。
- `publishPlaceholders`：NuGet、GitHub Packages、GitHub Release assets 的文本占位命令。
- `postPublishActions`：真实发布后必须执行的 clean consumer 回填动作。

`publishPlaceholders` 可以包含：

```powershell
dotnet nuget push <package>.nupkg --api-key <NUGET_API_KEY> --source https://api.nuget.org/v3/index.json
dotnet nuget push <package>.nupkg --api-key <GITHUB_TOKEN> --source <github-packages-source>
gh release upload <tag> <package>.nupkg <package>.sha256
```

这些都是文本占位符。脚本不会执行 `dotnet nuget push`，不会上传 GitHub Packages，不会上传 GitHub Release assets，也不会执行 delete、delist 或 withdraw。

## Validator 规则

`Test-ReleaseCandidateFreezeSummary.ps1` 校验：

- summary 必须是 `recordKind=release-candidate-freeze-summary`。
- runtime package key 必须匹配当前发布目标。
- `performsPublish=false`。
- close readiness 必须在 release evidence、final dry run、publish checklist、promotion issue 与 post-publish validation 之间保持一致。
- `canCloseReleaseIssue` 只有在真实 post-publish verification proof 已存在并通过 validator 后才允许进入可关闭状态。
- `canPublish=true` 或 `canPromote=true` 时必须已经有真实 compatible-host external runtime proof。
- `blocked-by-cuda-driver` 不能和 `realExternalRuntimeProofReady=true` 同时成立。
- external runtime proof 缺 `stdoutSummary` 或 `stderrSummary` 时必须保持 `realExternalRuntimeProofReady=false`；无 stderr 时也必须写明 `no-stderr-emitted`。
- `real-external-runtime-proof` blocking item 必须存在。
- `external-runtime-proof-backfill-required` blocking item 必须存在，并保持 backfill plan 只是 guidance。
- `post-publish-verification` blocking item 必须存在。
- `post-publish-verification-backfill-required` blocking item 必须存在，并保持 backfill plan 不是 post-publish proof。
- 如果 freeze checklist 已生成，它也必须保持 `performsPublish=false`，并与 summary 的 close readiness 一致。

默认缺真实 proof 时，validator 的期望状态是：

- `ValidationState=blocked-freeze-owner-action-required`
- `FailedValidationItemCount=0`
- `CanCloseReleaseIssue=False`

这表示冻结门禁本身是健康的，但项目仍需要 owner action；不是发布已完成。

## 关键边界

- release candidate freeze 不是 publish。
- owner approval / owner decision 不是 proof 本身。
- blocked-by-cuda-driver 不是 smoke passed；它只能表示当前主机 CUDA driver/runtime compatibility 阻塞，不能晋级为真实 runtime proof。
- `allowRuntimeSmokeBlocked=true` 只记录 dry-run 意图，不提升 runtime proof。
- `template-only`、`draft-only`、example、runbook、collection bundle、dependency-probe-only 都不是真实 proof。
- external runtime proof backfill plan 和 post-publish verification backfill plan 只是 guidance-only；它们不是 runtime proof、post-publish proof、publication approval、release close approval 或 package push。
- external runtime proof 必须来自兼容 CUDA/TensorRT/cuDNN/GPU/driver host 上的 clean package-consumer runtime smoke。
- post-publish verification 必须来自真实渠道发布后的 clean consumer restore/build/smoke。
- post-publish record 缺 clean consumer identity、host metadata、带 `--runtime-package-key` 的 smoke command、stdout/stderr summary 或 SHA256-backed logs 时，release issue 不能关闭。

## 下一步

当 freeze summary 显示 `realExternalRuntimeProofReady=false` 时，owner 应先在兼容主机执行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -SmokeRuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -RunSmoke -KeepConsumerOutput
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -RequireExistingLog -FailOnNotProof
```

当真实发布完成后，owner 才能回填 `post-publish-verification-record.json`，并重新运行 release evidence、final dry run、publish checklist、promotion issue、freeze summary/checklist 与 freeze validation。

若需要把 owner 授权后的发布命令整理成一份安全审阅包，先运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerAuthorizedPublishCommandPlan.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerAuthorizedPublishCommandPlan.ps1
```

输出 `owner-authorized-publish-command-plan.json/.md` 与 `owner-authorized-publish-command-plan-validation.json/.md`。该计划默认 `canMaterializeExecutableCommands=false`，所有 publish commands 都是 placeholder，不执行 `dotnet nuget push`、GitHub Packages upload 或 GitHub Release upload。
