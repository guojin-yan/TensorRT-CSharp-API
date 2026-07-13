# Owner Authorized Publish Command Plan

`owner-authorized-publish-command-plan` 是 release owner 授权前的安全命令计划层。它把 NuGet、GitHub Packages、GitHub Release assets 和 post-publish clean consumer 回填命令放到同一个审阅视图中，但默认不执行任何发布动作。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerAuthorizedPublishCommandPlan.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerAuthorizedPublishCommandPlan.ps1
```

输出：

- `artifacts/final-release/owner-authorized-publish-command-plan.json`
- `artifacts/final-release/owner-authorized-publish-command-plan.md`
- `artifacts/final-release/owner-authorized-publish-command-plan-validation.json`
- `artifacts/final-release/owner-authorized-publish-command-plan-validation.md`
- `artifacts/final-release/external-runtime-proof-backfill-plan.json`
- `artifacts/final-release/post-publish-verification-backfill-plan.json`

默认状态必须保持：

- `recordKind=owner-authorized-publish-command-plan`
- `planState=blocked-owner-authorization-required`
- `performsPublish=false`
- `requiresHumanOwner=true`
- `requiresExplicitOwnerAuthorization=true`
- `canMaterializeExecutableCommands=false`
- `placeholderOnly=true`
- `materializedExecutableCommand=""`
- `modelExecutionForbidden=true`

## 它读取什么

命令计划读取并串联：

- `artifacts/release/release-candidate-freeze-summary.json`
- `artifacts/release/release-candidate-freeze-checklist.json`
- `artifacts/release/release-candidate-freeze-validation.json`
- `artifacts/final-release/release-publish-execution-checklist.json`
- `artifacts/final-release/final-package-review-bundle.json`
- `artifacts/final-release/external-runtime-proof-backfill-plan.json`
- `artifacts/final-release/post-publish-verification-backfill-plan.json`
- `artifacts/final-release/release-owner-approval-input-validation.json`
- `artifacts/final-release/release-owner-decision-record.json`
- `artifacts/final-release/external-runtime-proof-validation.json`
- `artifacts/final-release/post-publish-verification-validation.json`
- `artifacts/final-release/stale-release-claims-audit.json`

## 命令边界

`publishCommands` 中的命令全部是 placeholder：

- `authorized=false`
- `executable=false`
- `performsPublish=false`

即使命令文本包含 `dotnet nuget push` 或 `gh release upload`，脚本也不会执行它们。真实执行必须由 release owner 在具备 owner approval、owner decision、compatible-host runtime proof 和 release channel 访问凭证后手工完成。

## 大模型执行边界

- 大模型和自动化只能生成、刷新、校验这个 owner review artifact，不得执行真实发布。
- `publishCommands` 只允许保留 placeholder 文本；当 `canMaterializeExecutableCommands=false` 时，`materializedExecutableCommand` 必须为空。
- release owner 手工复制命令前，必须在 owner 私有环境替换 `<NUGET_API_KEY>`、`<GITHUB_TOKEN>`、`<tag>`、`<package>` 等 placeholder，并复核 package URL、SHA256、channel source 和授权记录。
- 大模型不得代替 owner 运行 NuGet push、GitHub Packages upload、GitHub Release upload、delete、delist 或 withdraw。

## Owner Authorization Proof Gate

`ownerAuthorizationProofGate` 是手工发布命令能否物化的最小安全门。默认状态必须保持：

- `gateState=blocked-owner-authorization-required`
- `requiredFieldCount=14`
- `missingOwnerInputCount=14`
- `canMaterializeExecutableCommands=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

Owner 授权记录至少要补齐这些字段，且必须与 `release-owner-approval-input-record.json`、`release-owner-decision-record.json`、`owner-authorized-publish-command-plan-validation.json` 的校验结果一致：

- `ownerName`
- `ownerDecisionId`
- `approvalTimestampUtc`
- `targetChannel`
- `selectedRuntimePackageKey`
- `approvedManagedPackageId`
- `approvedRuntimePackageId`
- `approvedManagedPackageVersion`
- `approvedRuntimePackageVersion`
- `approvedCommandPlanSha256`
- `approvedProofBundleSha256`
- `rollbackPlan`
- `credentialHandlingAcknowledged`
- `nvidiaRedistributionApproval`

`manualMaterializationPrerequisites` 还必须同时覆盖 owner authorization、owner decision、freeze summary、package-consumer runtime proof、publish checklist 和 stale release claims audit。只要任一项仍是模板、draft、placeholder、过期 claim 或缺少真实 proof，命令物化就必须保持 blocked。

## Post-publish 回填

`postPublishCommandPlan` 固定列出真实发布后的回填顺序：

1. 创建源码仓库外的 clean consumer。
2. 运行 `Test-PostPublishCleanConsumerProject.ps1` 确认没有 `ProjectReference` 且 consumer 位于仓库外。
3. 从真实发布 channel restore。
4. build clean consumer 并保存 native asset listing。
5. 运行 DependencyProbe 并保存 SHA256-backed log。
6. 在兼容 CUDA/TensorRT 主机上用 `--runtime-package-key` 运行 runtime smoke。
7. 可用 `Export-PostPublishVerificationRecordInputDraft.ps1` 生成 input draft，但 draft 仍不是 proof。
8. 运行 `Test-PostPublishVerificationRecord.ps1 -FailOnNotProof`。

只有真实 `post-publish-verification-record.json` 通过验证，才允许 release issue 从不可关闭变为关闭就绪。

Post-publish 回填证据必须与 `postPublishRequiredEvidence` 对齐，至少包括 selected channel、channel source URI、published package URL、managed/runtime package URL、managed/runtime nupkg SHA256、源码仓库外 clean consumer、consumer `.csproj` 路径、无 `ProjectReference`、restore log、native asset listing SHA256、DependencyProbe log/SHA256、`runtimeSmokeLogPath`、`runtimeSmokeLogSha256`、`runtimeSmokePassed=true`、`runtimeSmokeExitCode=0`、stdout/stderr 摘要和 host metadata。stderr 没有输出时也要写明 `no-stderr-emitted` 或等价复核说明。

`Export-ReleaseClosePreflight.ps1` 可在 owner proof、post-publish proof 和 stale claims 刷新后聚合检查 release close 缺口；它只输出 close preflight artifact，不执行发布命令。

## Backfill Plan 聚合

命令计划会读取：

- `external-runtime-proof-backfill-plan.json`
- `post-publish-verification-backfill-plan.json`

并输出：

- `externalRuntimeProofBackfillPlanState`
- `externalRuntimeProofBackfillStepCount`
- `externalRuntimeProofBackfillCanPromoteRuntimeProof`
- `postPublishVerificationBackfillPlanState`
- `postPublishVerificationBackfillStepCount`
- `postPublishVerificationBackfillCanCloseReleaseIssue`

这些字段只用于 owner 审阅剩余工作。backfill plan 是 guidance，不是 runtime proof、post-publish proof、publish approval 或 release close approval。

## 不能误读

- command plan 不是 publish。
- owner approval / owner decision 不是 runtime proof。
- backfill plan 不是 proof，也不是 release close approval。
- `blocked-by-cuda-driver` 不是 smoke passed。
- template、draft、runbook、collection bundle、dependency-probe-only 都不是真实 proof。
- external runtime proof 缺 `stdoutSummary` 或 `stderrSummary` 时不能发布；stderr 为空也必须写明 `no-stderr-emitted`。
- 没有真实 `external-runtime-proof-record.json` 时不能发布。
- 没有真实 `post-publish-verification-record.json` 时不能关闭 release issue。
