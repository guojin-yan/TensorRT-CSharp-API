# Release Evidence Bundle

`release-evidence-bundle` 是发布证据汇总层。它把 owner approval、final dry run、package consumer、runtime readiness、Linux runner、external runtime proof、owner proof backfill execution pack、real external proof backfill execution bundle、real proof runner input backfill、real proof execution record projection、owner real proof report pack、real proof record validator、owner real proof execution closure pack、owner proof execution handoff、owner external proof input preflight、owner proof input repair pack、owner proof input draft pack、owner external proof backfill orchestrator、package consumer runtime proof owner input、package consumer runtime proof record、package consumer runtime proof worklist、release close proof worklist、package consumer external smoke scaffold、package consumer runtime proof candidate、release issue close record owner input、release issue close record candidate、real external proof overlay pack、release issue close record overlay candidate、owner external execution result backfill kit、owner input cross-hash audit、release close strict record candidate、owner external proof execution bundle、owner external proof execution result import、real external proof record import validator、release close owner input bridge、owner proof real input convergence、release close final owner runbook、final package review、package proof bundle、docs publish readiness bundle、publish checklist、promotion issue、post-publish verification、stale claims 和 user acceptance catalog 聚合到同一份 JSON/Markdown 中。

它不是发布批准，不会执行 `dotnet nuget push`，不会上传 GitHub Packages，也不会把模板、example、dry run 或 DependencyProbe 变成 runtime proof。

Owner 最短执行面以 `owner-release-execution-package` 的 `oneScreenReleaseHoldChecklist` 为准；`owner-proof-backfill-execution-pack` 是聚焦到真实输入字段、first command、validator 和不可替代 proof 类型的 companion；`owner-proof-execution-handoff` 把这些 proof line 转成候选产物、缺失输入和 owner next action 看板；`owner-external-proof-input-preflight` 用于预审候选 owner 输入是否仍只是 template/guidance/candidate；`owner-proof-input-repair-pack` 继续把 blocked line 拆成 placeholder、existing file、SHA256、clean consumer evidence、owner decision、rollback plan 与 validator 的修复清单；`owner-proof-input-draft-pack` 提供 per-line draft path 和 strict validator；`owner-external-proof-backfill-orchestrator` 把 draft spec 转成 owner command plan；`package-consumer-runtime-proof-owner-input` 与 `release-issue-close-record-owner-input` 定义 candidate overlay 所需真实字段；`real-external-proof-overlay-pack` 将 package-consumer-runtime、post-publish verification、release close owner input 和 final close decision 的真实字段集中成 Owner 回填包；`release-issue-close-record-overlay-candidate` 将 close record 所需路径、SHA256 和 placeholder 输入映射为候选关闭面；`owner-external-execution-result-backfill-kit` 把 Owner 外部执行结果缺口聚合为回填包；`owner-input-cross-hash-audit` 只审计本地 artifact/path/hash 一致性；`release-close-strict-record-candidate` 将 close record 关键 hash、Owner 输入缺口和 real proof 缺口收束到更严格候选面；`owner-proof-real-backfill-execution-pack` 将 strict candidate 拆成 owner input、real proof 和 hash check 三类 Owner 回填任务；`release-issue-close-record-real-input-map` 将 Owner 输入任务映射到最终 close record 字段和目标 artifact；`owner-proof-real-input-convergence` 将剩余 Owner 输入、validator 和 proof blocker 收敛成 Owner 可读矩阵；`release-close-final-owner-runbook` 将收敛矩阵整理成最终 Owner 执行手册，串联 public package proof、clean consumer runtime smoke、rollback review、final close decision 和 strict close validator；`package-consumer-runtime-proof-record` 把 package consumer owner input 投影到 strict proof record 并可桥接 `external-runtime-proof-record`；`package-consumer-external-smoke-scaffold` 只生成仓库外 clean consumer 项目骨架，不能替代运行证据；`package-consumer-runtime-proof-candidate` 和 `release-issue-close-record-candidate` 进一步聚焦 clean external consumer smoke 与 release close owner input，但仍保持 blocked/non-proof。本 bundle 可以引用或镜像这些材料，但它们仍是 guidance/candidate/input/scaffold/audit/runbook，不是 proof。真实 release close 仍需要 owner authorization、`package-consumer-runtime`、Linux runner proof、`real-model-runtime` 和 `post-publish verification` 的记录与 validator；这些全部通过后，还必须通过 `release-issue-close-record-validation` 和 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`。缺失时 `canCloseReleaseIssue=false` 必须保持不变。

## Non-Substitute Proof Kinds

## Owner External Proof Execution Chain

`owner-real-input-landing-pack`、`final-owner-execution-checklist` 和 `real-proof-import-boundary-audit` 将真实 Owner 输入落地、最终人工执行清单和 forbidden substitute claim 边界审计接入 release evidence bundle。三者默认 `Passed=false`，只暴露 blocked owner input、manual execution guidance 或 boundary audit；它们不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

`owner-external-proof-execution-bundle` 将 6 条 proof lane 转成 Owner 外部执行命令、hash 命令、host metadata 和导入字段映射。它仍是 blocked owner execution guidance，不是 runtime proof、publish approval、post-publish proof 或 release close approval。

`owner-external-proof-execution-result-import` 将 Owner 执行结果槽位投影为导入记录，逐 lane 标出仍缺失的真实文件、hash、metadata、validator output 和 reviewer 字段。默认状态为 `blocked-owner-external-proof-execution-result-required`，不能晋级 proof、发布或关闭 release issue。

`real-external-proof-record-import-validator` 将 result import 转成 6 个严格真实 proof 导入合同，列出 forbidden substitutes、strict validator command 和 blocked reasons。合同校验本身不是 proof，也不能替代真实 runtime proof records、post-publish proof 或 release close approval。

`release-close-owner-input-bridge` 将真实外部执行、结果导入、proof record 导入、post-publish、rollback/final owner decision 和 strict close validator 聚合成 release close owner gate。即使 classification audit gate ready，其他真实 proof gate 未完成时仍必须保持 blocked，不能把 bridge 当作 close approval。

`public-package-proof-owner-input`、`post-publish-proof-owner-confirmation` 和 `release-close-public-proof-bridge` 进一步把真实公开包发布后的 owner input、post-publish proof gate 和 final close public proof gate 接入 bundle。公开包 owner input 现在要求 NuGet package source、GitHub Release asset path/hash、公开包 URL/hash、仓库外 clean consumer restore/build/smoke logs、stdout/stderr SHA256、host metadata 和 Owner review；三者默认均为 blocked/non-proof，不能替代 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

`release-issue-close-final-owner-decision-audit` 和 `final-post-publish-audit-pack` 将最终 Owner close decision gate 与发布后审计 lane 汇总进 release evidence bundle。两者默认 `Passed=false`，只暴露 remaining owner action，不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

`release-candidate-final-freeze-manifest`、`public-publish-owner-manual-command-handoff` 和 `final-release-close-blocker-dashboard` 进一步补齐公开发布 Owner 手动执行前的最终交接层。三者分别记录本地冻结 hash、发布占位命令和 close blocker 看板，但默认仍是 non-proof / handoff / dashboard，不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

`public-publish-result-owner-input`、`public-publish-result-import`、`post-publish-clean-consumer-result-convergence` 和 `strict-close-ready-convergence-dashboard` 继续把 Owner 真实公开发布结果回填、hash/URL/timestamp/transcript 校验、clean consumer proof 缺口和 strict close ready lane 收敛进 bundle。四者默认 `Passed=false`，不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

`public-publish-final-owner-execution-pack`、`public-publish-command-cross-check`、`release-issue-close-owner-decision-input` 和 `final-evidence-freeze-non-proof-audit` 将最终 Owner 人工执行包、发布命令核对、release issue close 决策输入和 non-proof 边界复查接入 bundle。四者默认 `Passed=false`，只暴露 Owner action required 和 boundary audit，不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。`post-publish-clean-consumer-owner-proof-input` 只是说明真实 clean consumer proof 应如何回填，不能替代真实公开渠道运行记录。

`public-publish-real-result-owner-input-contract`、`post-publish-clean-consumer-proof-record-contract`、`release-issue-close-strict-owner-decision-import` 和 `final-close-gate-convergence` 将真实发布后的 Owner 回填合同、仓库外 clean consumer proof record 合同、严格关闭决策导入和最终 close gate 收敛接入 bundle。四者默认 `Passed=false`，只暴露缺失 Owner 输入和最终 gate blocker，不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

`public-publish-real-result-record-draft`、`post-publish-clean-consumer-proof-record-draft`、`public-publish-forbidden-substitute-scan`、`release-close-real-proof-import-bridge` 和 `final-owner-close-readiness-checkpoint` 将真实 proof 回填执行面继续前移到 Owner 草稿、替代物扫描、真实 proof import bridge 和最终 close readiness checkpoint。五者默认 `Passed=false`，只暴露 blocked 字段、blocked check 或 blocked lane；它们不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

`final-release-close-record-real-validator`、`final-owner-release-close-record-projection`、`final-release-close-hash-consistency-gate` 和 `final-close-owner-approval-boundary-audit` 将最终 ReleaseCloseRecord 真实验证面继续前移到字段合同、关闭记录投影、hash 一致性和 Owner approval 边界审计。四者默认 `Passed=false`，只暴露 blocked 字段、blocked lane 或当前本地 hash 结果；它们不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

`release-candidate-final-publishability-audit`、`release-candidate-owner-action-roadmap`、`release-candidate-non-substitute-final-scan` 和 `release-candidate-final-owner-checklist` 将最终可发布性总检继续前移到发布 gate、Owner 行动路线、禁止替代物扫描和一页式 checklist。四者默认 `Passed=false`，只暴露 blocked gate、blocked action、blocked substitute check 或 blocked checklist item；它们不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

Bundle 必须把以下材料统一列为非替代 proof，避免 release-facing 入口出现口径漂移：

- template
- draft
- runbook
- collection package
- input package
- local feed
- ProjectReference
- build-only
- parse-only
- sidecar-only
- dependency-probe-only
- Skipped=True
- blocked-by-cuda-driver
- bridge-only package consumer log
- bridge-only wrapper surface
- WrapperSurfaceEvidenceKind=compile-surface-proof
- IsRuntimeExecutionProof=False
- mismatched log SHA256
- Parser/ParserRefitter diagnostic snapshots
- copied managed diagnostic snapshot
- Windows handoff for Linux proof
- schema-only release issue close record
- template-only release issue close record
- preflight-only release issue close record
- release-issue-close-record-template.json

这些条目只能作为 blocker、diagnostic、owner guidance 或待回填上下文出现；不能关闭 release issue、不能替代 post-publish verification，也不能替代 package-consumer runtime proof。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CompatibleHostRuntimeProofRunbook.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CompatibleHostRuntimeProofCollectionBundle.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofBackfillPlan.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationBackfillPlan.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerProofBackfillExecutionPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerProofExecutionHandoff.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerExternalProofInputPreflight.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofWorklist.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCloseProofWorklist.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealExternalProofBackfillExecutionBundle.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealProofRunnerInputBackfill.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealProofRunnerInputBackfill.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealProofExecutionRecordProjection.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealProofExecutionRecordProjection.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerRealProofReportPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerRealProofReportPack.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealProofInputCandidateStrictRecord.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealProofInputCandidateStrictRecord.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerRealProofFieldDeltaPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerRealProofFieldDeltaPack.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealProofCandidatePromotionGuard.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealProofCandidatePromotionGuard.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealProofRecordValidator.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealProofRecordValidator.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerRealProofExecutionClosurePack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerRealProofExecutionClosurePack.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RuntimeProofExecutionInputRecord.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RuntimeProofExecutionInputRecord.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerRuntimeProofExecutionRunbook.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerRuntimeProofExecutionRunbook.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCloseStrictValidationBridge.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCloseStrictValidationBridge.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealExternalProofOverlayPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealExternalProofOverlayPack.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseRecordOverlayCandidate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecordOverlayCandidate.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerExternalExecutionResultBackfillKit.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerExternalExecutionResultBackfillKit.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerInputCrossHashAudit.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerInputCrossHashAudit.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCloseStrictRecordCandidate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCloseStrictRecordCandidate.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseEvidenceClassificationAudit.ps1 -Strict
```

输出：

- `artifacts/final-release/release-evidence-bundle.json`
- `artifacts/final-release/release-evidence-bundle.md`
- `artifacts/final-release/release-evidence-classification-audit.json`
- `artifacts/final-release/release-evidence-classification-audit.md`

默认状态必须保持：

- `bundleState=blocked-evidence-incomplete`
- `canPublishPublicly=false`
- `canExecutePublicPublish=false`
- `canPromoteRuntimeProof=false`
- `canCloseReleaseIssue=false`
- `owner-action-required.md` 被纳入 evidence bundle 的 source/evidence item，但它仍只是 owner handoff checklist，不是 owner authorization、runtime proof、post-publish proof 或 release close approval。
- `isReleaseEvidenceComplete=false`
- `postPublishConsumerProjectIdentityReady=false`
- `postPublishSmokeCommandRuntimeKeyReady=false`
- `postPublishHostReady=false`
- `postPublishCommandsReady=false`
- `postPublishStdoutStderrSummaryReady=false`
- `externalRuntimeProofBackfillPlanState=blocked-compatible-host-proof-required`
- `externalRuntimeProofBackfillCanPromoteRuntimeProof=false`
- `postPublishVerificationBackfillPlanState=blocked-real-post-publish-proof-required`
- `postPublishVerificationBackfillCanCloseReleaseIssue=false`
- `ownerProofBackfillExecutionPackState=blocked-real-proof-required`
- `ownerProofBackfillExecutionPackReadyBackfillItemCount=0`
- `ownerProofExecutionHandoffState=blocked-real-proof-required`
- `ownerProofExecutionHandoffReadyLineCount=0`
- `ownerExternalProofInputPreflightState=blocked-real-proof-required`
- `ownerExternalProofInputPreflightReadyLineCount=0`
- `ownerProofInputRepairPackState=blocked-real-input-repair-required`
- `ownerProofInputRepairPackBlockedItemCount=6`
- `ownerProofInputDraftPackState=blocked-draft-non-proof`
- `ownerProofInputDraftPackBlockedItemCount=6`
- `ownerExternalProofBackfillOrchestratorState=blocked-owner-external-proof-required`
- `ownerExternalProofBackfillOrchestratorBlockedLineCount=6`
- `releaseIssueCloseRecordValidationState=blocked-template-only`
- `releaseIssueCloseRecordProofClassification=template-only`
- `releaseIssueCloseRecordFailedValidationItemCount=10`
- `releaseIssueCloseRecordCanPromoteReleaseIssueCloseRecord=false`
- `realExternalProofOverlayPackState=blocked-real-owner-input-required`
- `realExternalProofOverlayPackValidationState=blocked-real-owner-input-required`
- `releaseIssueCloseRecordOverlayCandidateState=blocked-release-close-real-proof-required`
- `releaseIssueCloseRecordOverlayCandidateValidationState=blocked-release-close-overlay-owner-input-required`
- `ownerExternalExecutionResultBackfillKitState=blocked-owner-external-execution-results-required`
- `ownerExternalExecutionResultBackfillKitValidationState=blocked-owner-external-execution-results-required`
- `ownerInputCrossHashAuditState=blocked-owner-input-cross-hash-audit-owner-proof-required`
- `ownerInputCrossHashAuditValidationState=blocked-owner-input-cross-hash-audit-owner-proof-required`
- `ownerInputCrossHashAuditMismatchedHashCount=0`
- `releaseCloseStrictRecordCandidateState=blocked-release-close-strict-record-owner-input-required`
- `releaseCloseStrictRecordCandidateValidationState=blocked-release-close-strict-record-owner-input-required`
- `releaseCloseStrictRecordCandidateMismatchedHashCount=0`
- `releaseCloseStrictRecordCandidateMissingOwnerInputCount>=1`
- `releaseCloseStrictRecordCandidateMissingRealProofCount>=1`
- `packageConsumerRuntimeProofWorklistState=blocked-real-package-consumer-runtime-proof-required`
- `packageConsumerRuntimeProofWorklistIsRuntimeExecutionProof=false`
- `releaseCloseProofWorklistState=blocked-release-close-real-proof-required`
- `releaseCloseProofWorklistIsReleaseCloseProof=false`
- `releaseCloseProofWorklistIsRuntimeExecutionProof=false`
- `realExternalProofBackfillExecutionBundleState=blocked-real-external-proof-execution-required`
- `realExternalProofBackfillExecutionBundleIsRuntimeExecutionProof=false`
- `realExternalProofBackfillExecutionBundleIsReleaseCloseProof=false`
- `realProofRunnerInputBackfillState=blocked-owner-runner-input-required`
- `realProofRunnerInputBackfillValidationState=blocked-owner-runner-input-required`
- `realProofRunnerInputBackfillFailedBlockerCount=0`
- `realProofRunnerInputBackfillFailedActionRequiredCount>=1`
- runner input validation 必须保留 `failedActionRequiredCount>=1`，表示模板仍等待 Owner 真实 proof runner 输入。
- `realProofRunnerInputBackfillIsRuntimeExecutionProof=false`
- `realProofRunnerInputBackfillIsReleaseCloseProof=false`
- `realProofExecutionRecordProjectionState=blocked-real-proof-execution-record-input-required`
- `realProofExecutionRecordProjectionValidationState=blocked-real-proof-execution-record-input-required`
- `realProofExecutionRecordProjectionRecordCount=6`
- `realProofExecutionRecordProjectionReadyRecordCount=0`
- `realProofExecutionRecordProjectionFailedBlockerCount=0`
- `realProofExecutionRecordProjectionFailedActionRequiredCount>=1`
- projection validation 必须保留 `failedActionRequiredCount>=1`，表示 execution record 仍等待 Owner 真实执行结果、log、hash 和 validator output。
- `realProofExecutionRecordProjectionIsRuntimeExecutionProof=false`
- `realProofExecutionRecordProjectionIsReleaseCloseProof=false`
- `ownerRealProofReportPackState=blocked-owner-real-proof-report-input-required`
- `ownerRealProofReportPackValidationState=blocked-owner-real-proof-report-input-required`
- `ownerRealProofReportPackReportItemCount=6`
- `ownerRealProofReportPackReadyForOwnerReviewCount=0`
- `ownerRealProofReportPackReadyForPromotionCount=0`
- `ownerRealProofReportPackFailedBlockerCount=0`
- `ownerRealProofReportPackFailedActionRequiredCount>=1`
- owner real proof report pack validation 必须保留 `failedActionRequiredCount>=1`，表示 Owner 真实 proof 填报包仍等待真实证据、hash、validator output 和人工复查。
- `ownerRealProofReportPackIsRuntimeExecutionProof=false`
- `ownerRealProofReportPackIsReleaseCloseProof=false`
- `realProofInputCandidateStrictRecordState=blocked-real-proof-input-candidate-required`
- `realProofInputCandidateStrictRecordValidationState=blocked-real-proof-input-candidate-required`
- `realProofInputCandidateStrictRecordCandidateCount=6`
- `realProofInputCandidateStrictRecordBlockedCandidateCount=6`
- `realProofInputCandidateStrictRecordReadyCandidateCount=0`
- `realProofInputCandidateStrictRecordFailedBlockerCount=0`
- `realProofInputCandidateStrictRecordFailedActionRequiredCount>=1`
- real proof input candidate strict record validation 必须保留 `failedActionRequiredCount>=1`，表示候选合同仍等待真实证据、log、hash、validator output、forbidden substitute 复查和 Owner review。
- `realProofInputCandidateStrictRecordIsRuntimeExecutionProof=false`
- `realProofInputCandidateStrictRecordIsReleaseCloseProof=false`
- `ownerRealProofFieldDeltaPackState=blocked-owner-real-proof-field-delta-required`
- `ownerRealProofFieldDeltaPackValidationState=blocked-owner-real-proof-field-delta-required`
- `ownerRealProofFieldDeltaPackCandidateCount=6`
- `ownerRealProofFieldDeltaPackBlockedFieldContractCount>=6`
- `ownerRealProofFieldDeltaPackReadyFieldContractCount=0`
- `ownerRealProofFieldDeltaPackFailedBlockerCount=0`
- `ownerRealProofFieldDeltaPackFailedActionRequiredCount>=1`
- owner real proof field delta pack validation 必须保留 `failedActionRequiredCount>=1`，表示 Owner 仍需完成真实字段填报、日志、hash、validator 和 review。
- `ownerRealProofFieldDeltaPackIsRuntimeExecutionProof=false`
- `ownerRealProofFieldDeltaPackIsReleaseCloseProof=false`
- `realProofCandidatePromotionGuardState=blocked-real-proof-candidate-promotion-not-allowed`
- `realProofCandidatePromotionGuardValidationState=blocked-real-proof-candidate-promotion-not-allowed`
- `realProofCandidatePromotionGuardCandidateCount=6`
- `realProofCandidatePromotionGuardPromotionAllowedCandidateCount=0`
- `realProofCandidatePromotionGuardBlockedCandidateCount=6`
- `realProofCandidatePromotionGuardFailedBlockerCount=0`
- `realProofCandidatePromotionGuardFailedActionRequiredCount>=1`
- real proof candidate promotion guard validation 必须保留 `failedActionRequiredCount>=1`，表示候选提升仍被真实输入、non-substitute 和后续 real proof validator 阻断。
- `realProofCandidatePromotionGuardIsRuntimeExecutionProof=false`
- `realProofCandidatePromotionGuardIsReleaseCloseProof=false`
- `releaseEvidenceClassificationAudit.auditState=classification-audit-passed-non-proof-boundaries-intact`
- `releaseEvidenceClassificationAudit.isRuntimeExecutionProof=false`

## 汇总范围

bundle 会读取：

- `release-owner-approval-input-validation.json`
- `final-release-dry-run-summary.json`
- `package-consumer-validation-summary.json`
- `runtime-package-readiness-summary.json`
- `linux-runner-evidence-validation.json`
- `external-runtime-proof-validation.json`
- `external-runtime-proof-record.draft.json`
- `external-runtime-proof-owner-handoff.json`
- `compatible-host-runtime-proof-runbook.json`
- `compatible-host-runtime-proof-collection-bundle.json`
- `external-runtime-proof-backfill-plan.json`
- `owner-proof-backfill-execution-pack.json`
- `owner-release-execution-package-validation.json`
- `owner-proof-execution-handoff.json`
- `owner-external-proof-input-preflight.json`
- `owner-proof-input-repair-pack.json`
- `owner-proof-input-draft-pack.json`
- `owner-external-proof-backfill-orchestrator.json`
- `package-consumer-runtime-proof-candidate.json`
- `package-consumer-runtime-proof-owner-input.template.json`
- `package-consumer-runtime-proof-owner-input-validation.json`
- `package-consumer-runtime-proof-record.template.json`
- `package-consumer-runtime-proof-record.json`
- `package-consumer-runtime-proof-record-validation.json`
- `package-consumer-runtime-proof-worklist.json`
- `release-close-proof-worklist.json`
- `real-external-proof-backfill-execution-bundle.json`
- `package-consumer-external-smoke-scaffold.json`
- `release-issue-close-record-candidate.json`
- `release-issue-close-record-owner-input.template.json`
- `release-issue-close-record-owner-input-validation.json`
- `final-evidence-freeze.json`
- `final-evidence-freeze-validation.json`
- `release-issue-final-close-decision.template.json`
- `release-issue-final-close-decision-validation.json`
- `real-external-proof-overlay-pack.json`
- `real-external-proof-overlay-pack-validation.json`
- `release-issue-close-record-overlay-candidate.json`
- `release-issue-close-record-overlay-candidate-validation.json`
- `owner-external-execution-result-backfill-kit.json`
- `owner-external-execution-result-backfill-kit-validation.json`
- `owner-input-cross-hash-audit.json`
- `owner-input-cross-hash-audit-validation.json`
- `release-close-strict-record-candidate.json`
- `release-close-strict-record-candidate-validation.json`
- `owner-proof-real-backfill-execution-pack.json`
- `owner-proof-real-backfill-execution-pack-validation.json`
- `real-proof-record-validator.json`
- `real-proof-record-validator-validation.json`
- `owner-real-proof-execution-closure-pack.json`
- `owner-real-proof-execution-closure-pack-validation.json`
- `runtime-proof-execution-input-record.json`
- `runtime-proof-execution-input-record-validation.json`
- `owner-runtime-proof-execution-runbook.json`
- `owner-runtime-proof-execution-runbook-validation.json`
- `release-close-strict-validation-bridge.json`
- `release-close-strict-validation-bridge-validation.json`
- `owner-runtime-proof-result-input.template.json`
- `owner-runtime-proof-result-input-validation.json`
- `runtime-proof-lane-dry-run-summary.json`
- `runtime-proof-lane-dry-run-summary-validation.json`
- `release-close-strict-dry-run-summary.json`
- `release-close-strict-dry-run-summary-validation.json`
- `owner-external-proof-execution-bundle.json`
- `owner-external-proof-execution-bundle-validation.json`
- `owner-external-proof-execution-result-import.json`
- `owner-external-proof-execution-result-import-validation.json`
- `real-external-proof-record-import-validator.json`
- `real-external-proof-record-import-validator-validation.json`
- `release-close-owner-input-bridge.json`
- `release-close-owner-input-bridge-validation.json`
- `public-package-proof-owner-input.template.json`
- `public-package-proof-owner-input-validation.json`
- `post-publish-proof-owner-confirmation.json`
- `post-publish-proof-owner-confirmation-validation.json`
- `release-close-public-proof-bridge.json`
- `release-close-public-proof-bridge-validation.json`
- `release-issue-close-record-real-input-map.json`
- `release-issue-close-record-real-input-map-validation.json`
- `owner-proof-real-input-convergence.json`
- `owner-proof-real-input-convergence-validation.json`
- `release-close-final-owner-runbook.json`
- `release-close-final-owner-runbook-validation.json`
- `release-issue-close-final-owner-decision-audit.json`
- `release-issue-close-final-owner-decision-audit-validation.json`
- `final-post-publish-audit-pack.json`
- `final-post-publish-audit-pack-validation.json`
- `release-candidate-final-freeze-manifest.json`
- `release-candidate-final-freeze-manifest-validation.json`
- `public-publish-owner-manual-command-handoff.json`
- `public-publish-owner-manual-command-handoff-validation.json`
- `final-release-close-blocker-dashboard.json`
- `final-release-close-blocker-dashboard-validation.json`
- `public-publish-result-owner-input.template.json`
- `public-publish-result-owner-input-validation.json`
- `public-publish-result-import.json`
- `public-publish-result-import-validation.json`
- `post-publish-clean-consumer-result-convergence.json`
- `post-publish-clean-consumer-result-convergence-validation.json`
- `strict-close-ready-convergence-dashboard.json`
- `strict-close-ready-convergence-dashboard-validation.json`
- `public-publish-final-owner-execution-pack.json`
- `public-publish-final-owner-execution-pack-validation.json`
- `public-publish-command-cross-check.json`
- `public-publish-command-cross-check-validation.json`
- `release-issue-close-owner-decision-input.template.json`
- `release-issue-close-owner-decision-input-validation.json`
- `final-evidence-freeze-non-proof-audit.json`
- `final-evidence-freeze-non-proof-audit-validation.json`
- `public-publish-real-result-owner-input-contract.json`
- `public-publish-real-result-owner-input-contract-validation.json`
- `post-publish-clean-consumer-proof-record-contract.json`
- `post-publish-clean-consumer-proof-record-contract-validation.json`
- `release-issue-close-strict-owner-decision-import.json`
- `release-issue-close-strict-owner-decision-import-validation.json`
- `final-close-gate-convergence.json`
- `final-close-gate-convergence-validation.json`
- `public-publish-real-result-record-draft.json`
- `public-publish-real-result-record-draft-validation.json`
- `post-publish-clean-consumer-proof-record-draft.json`
- `post-publish-clean-consumer-proof-record-draft-validation.json`
- `public-publish-forbidden-substitute-scan.json`
- `public-publish-forbidden-substitute-scan-validation.json`
- `release-close-real-proof-import-bridge.json`
- `release-close-real-proof-import-bridge-validation.json`
- `final-owner-close-readiness-checkpoint.json`
- `final-owner-close-readiness-checkpoint-validation.json`
- `final-release-close-record-real-validator.json`
- `final-release-close-record-real-validator-validation.json`
- `final-owner-release-close-record-projection.json`
- `final-owner-release-close-record-projection-validation.json`
- `final-release-close-hash-consistency-gate.json`
- `final-release-close-hash-consistency-gate-validation.json`
- `final-close-owner-approval-boundary-audit.json`
- `final-close-owner-approval-boundary-audit-validation.json`
- `release-candidate-final-publishability-audit.json`
- `release-candidate-final-publishability-audit-validation.json`
- `release-candidate-owner-action-roadmap.json`
- `release-candidate-owner-action-roadmap-validation.json`
- `release-candidate-non-substitute-final-scan.json`
- `release-candidate-non-substitute-final-scan-validation.json`
- `release-candidate-final-owner-checklist.json`
- `release-candidate-final-owner-checklist-validation.json`
- `final-package-review-bundle.json`
- `release-package-proof-bundle.json`
- `docs-publish-readiness-bundle.json`
- `release-publish-execution-checklist.json`
- `release-promotion-issue-record.json`
- `post-publish-verification-validation.json`
- `post-publish-verification-owner-input.template.json`
- `post-publish-verification-owner-input-validation.json`
- `post-publish-verification-record.json`
- `post-publish-verification-backfill-plan.json`
- `release-issue-close-record-template.json`
- `release-issue-close-record-validation.json`
- `stale-release-claims-audit.json`
- `sample-smoke-catalog.json`

这些输入中任何关键项仍为 template-only、blocked、false 或缺失时，bundle 都必须继续保持阻断。

## 不能误读

- bundle 是证据聚合，不是 release owner approval。
- template/example 不是 proof。
- DependencyProbe 不是 runtime execution proof。
- external runtime proof owner handoff 是兼容主机交接材料，不是 runtime proof。
- compatible host runtime proof runbook 是外部兼容主机执行手册，不会发布包、不会批准公开发布，也不会替代 `external-runtime-proof-record.json`。
- compatible host runtime proof collection bundle 是外部兼容主机收集包，只串联 runbook、input template、smoke、hash 和 record validation，不会发布包、不会批准公开发布，也不会替代真实 runtime proof。
- external runtime proof backfill plan 是 owner 执行指引，不是 runtime proof；`canPromoteRuntimeProof=false` 必须保持，直到真实 `external-runtime-proof-record.json` 和 `-FailOnNotProof` validation 出现。
- owner proof backfill execution pack 是真实 proof 回填执行指引，不是 owner authorization、runtime proof、post-publish proof 或 release close proof；`performsPublish=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false` 必须保持。
- owner release execution package validation 只验证 owner 执行包是否包含一屏 hold 清单、validator、manual publish placeholder 和 non-substitute 边界；它不是 owner authorization、runtime proof、post-publish proof 或 close approval。
- owner proof execution handoff 是 owner 执行交接看板，不是 owner authorization、runtime proof、post-publish proof 或 release close proof；它只能展示候选产物、缺失输入和 validator，不能把 release gate 改成 ready。
- owner external proof input preflight 是候选输入预审，不是 owner authorization、runtime proof、post-publish proof 或 release close proof；即使 candidate artifacts 存在，只要没有 validator-passing real proof，release gate 仍必须 blocked。
- package consumer / release close owner input template 只是 overlay 输入面；即使字段形状通过，也不能替代真实 package-consumer smoke、post-publish proof、rollback approval 或 owner final decision。
- package consumer runtime proof record 在 `canPromoteRuntimeProof=true` 之前仍不是 release proof；template-only、local feed、ProjectReference 和 direct `.nupkg` 均保持 non-substitute。
- owner proof input repair pack 是真实输入修复清单，不是 owner authorization、runtime proof、post-publish proof 或 release close proof；repair markdown 和 input draft 不能替代 validator-passing real proof。
- owner proof input draft pack 是非 proof 填写面，不是 owner authorization、runtime proof、post-publish proof 或 release close proof；strict validator command 只是检查入口，draft 本身不能晋级。
- owner external proof backfill orchestrator 是 owner 命令计划，不是 owner authorization、runtime proof、post-publish proof 或 release close proof；它不能自行采集 proof 或把 release gate 改成 ready。
- final package review bundle 是本地 `.nupkg` 文件、SHA256、大小和 native asset count 的 owner review 清单，不是 public channel proof。
- package proof bundle 是本地包、split 包、native-copy 和 consumer 证据汇总，不是 public package proof。
- docs publish readiness bundle 是文档本地材料审阅证据，不是外部文档发布证明。
- `blocked-by-cuda-driver` 不是 smoke passed。
- post-publish verification 只能在真实渠道发布后，用 package identity/hash、clean consumer identity、host metadata、commands、`--runtime-package-key` smoke command 和 stdout/stderr summary 回填。
- post-publish verification backfill plan 是发布后回填指引，不是 post-publish proof；`canCloseReleaseIssue=false` 必须保持，直到真实渠道 proof 和 clean consumer smoke 出现。
- `postPublishCommandsReady=false` 或 `postPublishStdoutStderrSummaryReady=false` 时，release issue 不能被标记为 post-publish verified。
- release issue close record validator 会核对真实 post-publish proof、release close preflight、stale claim audit、evidence bundle SHA256、rollback plan 和 owner final close decision；`release-issue-close-record-template.json` 或 `blocked-template-only` validation 都不能作为关闭 proof。
- final evidence freeze 只冻结关键 artifact 的 SHA256 和 blocked state，不会晋级 publish/close gate；`final-evidence-freeze-valid-blocked-owner-action-required` 仍表示等待 Owner 真实执行。
- release issue final close decision template 只是最终 Owner 输入合同；在真实 post-publish proof、clean consumer smoke、rollback review 和 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady` 全部通过前，`blocked-owner-final-close-decision-required` 不能被解释为 close ready。
- real external proof overlay pack 只是 Owner 回填包；`blocked-real-owner-input-required` 不能解释为 proof ready。
- real proof runner input backfill 只是 Owner 待填字段模板和 shape validator；即使 blocker 为 0，只要 `failedActionRequiredCount>0`，它仍不能解释为 runtime proof、post-publish proof 或 release close proof。
- real proof execution record projection 只是执行记录形状投影；hash match 只能证明文件完整性，不能替代 runtime proof、Linux runner proof、post-publish proof 或 release close approval。
- owner real proof report pack 是 Owner 填报包，不是 proof；readyForOwnerReview 不等于 readyForPromotion，真实 release close 仍需要严格 validator 和外部 proof。
- real proof record validator 是 validator contract，不是执行结果；字段完整、命令存在和 blocker shape 合法不能替代真实 runtime logs、host metadata、package identity、log SHA256 或 post-publish proof。
- owner real proof execution closure pack 是 Owner 执行闭环清单，不是 proof；first command、expected artifacts、required logs、required SHA256 和 validator commands 不能自行运行 proof、发布包或关闭 release issue。
- release issue close record overlay candidate 只是 close record 输入映射；在 placeholder 被真实 Owner 输入替换并通过 strict close validator 前，不能关闭 release issue。
- owner external execution result backfill kit 只是 Owner 外部执行结果回填指导；它不能自行采集 proof、发布包、批准公开发布或关闭 release issue。
- owner input cross-hash audit 只证明本地 artifact/path/hash 一致；即使 `mismatchedHashCount=0`，也不能替代真实外部 proof、post-publish verification、Owner approval 或 release-close approval。
- release close strict record candidate 仍然只是最终 close record 候选面；hash consistency 和 blocked-shape validity 不能替代真实 Owner approval、post-publish proof、clean consumer runtime proof、rollback approval 或 strict close validation。
- release close proof worklist 只是最终关闭 blocker map；即使 worklist shape 合法，也不能替代真实 post-publish proof、clean consumer runtime smoke、Linux runner proof、real-model runtime proof、rollback approval、final close decision 或 strict close validation。
- release close final owner runbook 仍然只是最终 Owner 执行手册；即使 runbook shape 合法，也不能替代真实 post-publish proof、clean consumer runtime smoke、rollback approval、final close decision 或 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`。
- public publish real result owner input contract 只是 Owner 回填真实公开发布结果的合同；即使字段清单完整，也不能替代真实公开渠道 package URL、下载 hash、命令 transcript 或 package push。
- post-publish clean consumer proof record contract 只是仓库外 clean consumer proof 的记录合同；local feed、ProjectReference、direct `.nupkg`、template-only、scaffold-only 或 build-only 结果不能替代真实 post-publish proof。
- release issue close strict owner decision import 只是最终 Owner 关闭决策导入面；缺少真实公开包、clean consumer proof、rollback review、final decision 或 strict close validator 时，不能解释为 release close approval。
- final close gate convergence 只是最终 blocker 收敛视图；只要任一 lane 仍为 blocked、template-only、local-only、dry-run 或 owner-input-required，就不能关闭 release issue。
- 真实 publish 必须由 release owner 人工执行，脚本只生成审阅材料。


`public-release-owner-execution-package`、`external-clean-consumer-proof-kit`、`runtime-proof-compatible-host-kit`、`post-publish-owner-verification-kit` 和 `owner-public-release-execution-readiness-pack` 将真实公开发布 Owner 执行面继续前移到命令模板、外部 clean consumer proof、兼容主机 runtime proof、发布后 verification 和 readiness 汇总。五者默认 `Passed=false`，只暴露 blocked owner input，不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

`owner-external-real-proof-input-contract`、`owner-external-real-proof-import-validator`、`post-publish-clean-consumer-real-proof-gate`、`runtime-compatible-host-real-proof-gate` 和 `release-close-real-proof-readiness-gate` 将真实外部 proof 回填继续前移到字段合同、导入校验、仓库外 clean consumer、兼容主机 runtime proof 和 ReleaseClose 准入。五者默认 `Passed=false`，只暴露 blocked owner input，不执行发布、不下载包、不关闭 issue，也不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

`release-candidate-real-proof-final-freeze`、`owner-real-input-import-preflight`、`public-package-hash-cross-check-gate`、`clean-consumer-runtime-proof-cross-check-gate`、`post-publish-rollback-owner-decision-gate` 和 `release-close-final-real-input-admission-pack` 将最终真实 Owner 输入准入继续前移到证据 hash 冻结、Owner 输入预检、公开包 hash 交叉核对、clean consumer/runtime proof metadata 交叉核对、rollback Owner 决策和最终 ReleaseClose blocker 汇总。六者默认 `Passed=false`，只暴露 blocked owner input，不执行发布、不下载包、不关闭 issue，也不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

`owner-real-input-json-contract`、`owner-real-input-json-import`、`owner-real-input-hash-and-path-validator`、`owner-real-input-forbidden-substitute-validator`、`strict-close-real-input-dry-run`、`strict-close-real-input-finding-report`、`strict-close-owner-action-pack` 和 `release-close-real-input-final-blocker-ledger` 将 StrictClose 真实输入验证继续前移到字段合同、JSON 导入、hash/path 校验、禁止替代项校验、dry-run、finding report、Owner action pack 和最终 blocker ledger。八者默认 `Passed=false`，只暴露 blocked owner input，不执行发布、不下载包、不关闭 issue，也不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。
