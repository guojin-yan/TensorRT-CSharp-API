# TensorRtSharp4.0

TensorRtSharp4.0 是面向生产部署的 TensorRT / CUDA .NET 桥接工程。

## 项目范围

- 托管程序集：`JYPPX.TensorRtSharp`、`JYPPX.CudaSharp`
- NuGet 主包：`JYPPX.TensorRT.CSharp.API`
- 原生桥接库：`jyppxtrtbridge`
- 首批发布目标：Windows x64 和 Linux x64
- TensorRT 版本线：8.x、10.x、11.x
- CUDA 版本线：11.x、12.x、13.x

## 发布候选前台入口

当前发布面状态：

- 证据冻结状态：`blocked-real-proof-required`。
- 发布自动化状态：`performsPublish=false`。
- 公开渠道授权状态：`canPublishPublicly=false`。
- release issue 关闭状态：`canCloseReleaseIssue=false`。
- release issue 关闭记录校验：`release-issue-close-record-validation=blocked-template-only`；`release-issue-close-record-template.json` 不是 proof。
- 最终质量冻结状态：`blocked-final-quality-freeze-real-proof-required`；`final-quality-freeze-dashboard` 是 non-proof 看板，不是公开发布批准。
- 公开 proof claim 边界审计：`public-proof-claim-boundary-audit-passed`；它只扫描公开材料 stale claim，不是 runtime proof、post-publish proof、package push 或 release close approval。
- 30+ 文章矩阵校验：`article-roadmap-30plus-validation-passed-non-proof-planning`；文章矩阵只是内容规划。
- Owner real input landing pack 状态：`blocked-owner-real-input-required`；它把 5 个最终 blocker 映射到真实 Owner 文件、字段、strict validator 和 forbidden substitutes。
- Final Owner execution checklist 状态：`blocked-final-owner-execution-checklist-real-owner-input-required`；它只是最短人工执行/回填路径，不执行 `dotnet nuget push`。
- Real proof import boundary audit：`real-proof-import-boundary-audit-passed`；它只扫描公开/最终发布面是否误称 forbidden substitutes 为 proof。
- 剩余 owner proof blocker：owner authorization、`package-consumer-runtime`、Linux runner proof、`real-model-runtime` 和 post-publish verification。
- owner final backfill track 固定为 `package-consumer-runtime`、`linux-runner-proof`、`real-model-runtime` 和 `post-publish verification`；`ownerProofFinalBackfillTracks` 是执行地图，不是 proof。
- local feed、ProjectReference、bridge-only 日志、`Skipped=True`、mismatched log SHA256、build-only/precheck output、sidecar-only report、runbook、collection package 和 Windows handoff for Linux proof 仍然不能晋级为 release proof。

评估项目时建议优先阅读：

- Owner 一屏 Release Hold 清单：`docs/articles/zh-cn/owner-release-execution-package.md`
- Owner release execution package artifact：`artifacts/final-release/owner-release-execution-package.json`
- Owner release execution package validation：`artifacts/final-release/owner-release-execution-package-validation.json`
- Owner proof backfill execution pack artifact：`artifacts/final-release/owner-proof-backfill-execution-pack.json`
- Owner proof execution handoff artifact：`artifacts/final-release/owner-proof-execution-handoff.json`
- Owner external proof input preflight artifact：`artifacts/final-release/owner-external-proof-input-preflight.json`
- Owner proof input repair pack artifact：`artifacts/final-release/owner-proof-input-repair-pack.json`
- Owner proof input draft pack artifact：`artifacts/final-release/owner-proof-input-draft-pack.json`
- Owner external proof backfill orchestrator artifact：`artifacts/final-release/owner-external-proof-backfill-orchestrator.json`
- Package consumer runtime proof candidate artifact：`artifacts/final-release/package-consumer-runtime-proof-candidate.json`
- Package consumer runtime proof owner input template：`artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json`
- Package consumer runtime proof record validation：`artifacts/final-release/package-consumer-runtime-proof-record-validation.json`
- Package consumer external smoke scaffold：`artifacts/final-release/package-consumer-external-smoke-scaffold.json`
- Post-publish verification owner input template：`artifacts/final-release/post-publish-verification-owner-input.template.json`
- Post-publish verification record projection：`artifacts/final-release/post-publish-verification-record.json`
- Release issue close record candidate artifact：`artifacts/final-release/release-issue-close-record-candidate.json`
- Release issue close record owner input template：`artifacts/final-release/release-issue-close-record-owner-input.template.json`
- Final evidence freeze artifact：`artifacts/final-release/final-evidence-freeze.json`
- Final evidence freeze validation：`artifacts/final-release/final-evidence-freeze-validation.json`
- Release issue final close decision template：`artifacts/final-release/release-issue-final-close-decision.template.json`
- Release issue final close decision validation：`artifacts/final-release/release-issue-final-close-decision-validation.json`
- Real external proof overlay pack artifact：`artifacts/final-release/real-external-proof-overlay-pack.json`
- Real external proof overlay pack validation：`artifacts/final-release/real-external-proof-overlay-pack-validation.json`
- Release issue close record overlay candidate artifact：`artifacts/final-release/release-issue-close-record-overlay-candidate.json`
- Release issue close record overlay candidate validation：`artifacts/final-release/release-issue-close-record-overlay-candidate-validation.json`
- Owner external execution result backfill kit artifact：`artifacts/final-release/owner-external-execution-result-backfill-kit.json`
- Owner external execution result backfill kit validation：`artifacts/final-release/owner-external-execution-result-backfill-kit-validation.json`
- Owner input cross-hash audit artifact：`artifacts/final-release/owner-input-cross-hash-audit.json`
- Owner input cross-hash audit validation：`artifacts/final-release/owner-input-cross-hash-audit-validation.json`
- Release close strict record candidate artifact：`artifacts/final-release/release-close-strict-record-candidate.json`
- Release close strict record candidate validation：`artifacts/final-release/release-close-strict-record-candidate-validation.json`
- Owner proof real backfill execution pack artifact：`artifacts/final-release/owner-proof-real-backfill-execution-pack.json`
- Owner proof real backfill execution pack validation：`artifacts/final-release/owner-proof-real-backfill-execution-pack-validation.json`
- Release issue close record real input map artifact：`artifacts/final-release/release-issue-close-record-real-input-map.json`
- Release issue close record real input map validation：`artifacts/final-release/release-issue-close-record-real-input-map-validation.json`
- Owner real proof field delta pack artifact：`artifacts/final-release/owner-real-proof-field-delta-pack.json`
- Owner real proof field delta pack validation：`artifacts/final-release/owner-real-proof-field-delta-pack-validation.json`
- Real proof candidate promotion guard artifact：`artifacts/final-release/real-proof-candidate-promotion-guard.json`
- Real proof candidate promotion guard validation：`artifacts/final-release/real-proof-candidate-promotion-guard-validation.json`
- Real proof record validator artifact：`artifacts/final-release/real-proof-record-validator.json`
- Real proof record validator validation：`artifacts/final-release/real-proof-record-validator-validation.json`
- Owner real proof execution closure pack artifact：`artifacts/final-release/owner-real-proof-execution-closure-pack.json`
- Owner real proof execution closure pack validation：`artifacts/final-release/owner-real-proof-execution-closure-pack-validation.json`
- Runtime proof execution input record artifact：`artifacts/final-release/runtime-proof-execution-input-record.json`
- Runtime proof execution input record validation：`artifacts/final-release/runtime-proof-execution-input-record-validation.json`
- Final quality freeze dashboard artifact：`artifacts/final-release/final-quality-freeze-dashboard.json`
- Final quality freeze dashboard validation：`artifacts/final-release/final-quality-freeze-dashboard-validation.json`
- Public proof claim boundary audit artifact：`artifacts/final-release/public-proof-claim-boundary-audit.json`
- Article roadmap 30+ validation：`artifacts/final-release/article-roadmap-30plus-validation.json`
- Owner real input landing pack artifact：`artifacts/final-release/owner-real-input-landing-pack.json`
- Owner real input landing pack validation：`artifacts/final-release/owner-real-input-landing-pack-validation.json`
- Final Owner execution checklist artifact：`artifacts/final-release/final-owner-execution-checklist.json`
- Final Owner execution checklist validation：`artifacts/final-release/final-owner-execution-checklist-validation.json`
- Real proof import boundary audit artifact：`artifacts/final-release/real-proof-import-boundary-audit.json`
- Owner runtime proof execution runbook artifact：`artifacts/final-release/owner-runtime-proof-execution-runbook.json`
- Owner runtime proof execution runbook validation：`artifacts/final-release/owner-runtime-proof-execution-runbook-validation.json`
- Release close strict validation bridge artifact：`artifacts/final-release/release-close-strict-validation-bridge.json`
- Release close strict validation bridge validation：`artifacts/final-release/release-close-strict-validation-bridge-validation.json`
- Owner runtime proof result input template：`artifacts/final-release/owner-runtime-proof-result-input.template.json`
- Owner runtime proof result input validation：`artifacts/final-release/owner-runtime-proof-result-input-validation.json`
- Runtime proof lane dry-run summary：`artifacts/final-release/runtime-proof-lane-dry-run-summary.json`
- Runtime proof lane dry-run summary validation：`artifacts/final-release/runtime-proof-lane-dry-run-summary-validation.json`
- Release close strict dry-run summary：`artifacts/final-release/release-close-strict-dry-run-summary.json`
- Release close strict dry-run summary validation：`artifacts/final-release/release-close-strict-dry-run-summary-validation.json`
- Owner external proof execution bundle：`artifacts/final-release/owner-external-proof-execution-bundle.json`
- Owner external proof execution bundle validation：`artifacts/final-release/owner-external-proof-execution-bundle-validation.json`
- Owner external proof execution result import：`artifacts/final-release/owner-external-proof-execution-result-import.json`
- Owner external proof execution result import validation：`artifacts/final-release/owner-external-proof-execution-result-import-validation.json`
- Real external proof record import validator：`artifacts/final-release/real-external-proof-record-import-validator.json`
- Real external proof record import validator validation：`artifacts/final-release/real-external-proof-record-import-validator-validation.json`
- Release close owner input bridge：`artifacts/final-release/release-close-owner-input-bridge.json`
- Release close owner input bridge validation：`artifacts/final-release/release-close-owner-input-bridge-validation.json`
- Public package proof owner input template：`artifacts/final-release/public-package-proof-owner-input.template.json`
- Public package proof owner input validation：`artifacts/final-release/public-package-proof-owner-input-validation.json`
- Post-publish proof owner confirmation：`artifacts/final-release/post-publish-proof-owner-confirmation.json`
- Post-publish proof owner confirmation validation：`artifacts/final-release/post-publish-proof-owner-confirmation-validation.json`
- Release close public proof bridge：`artifacts/final-release/release-close-public-proof-bridge.json`
- Release close public proof bridge validation：`artifacts/final-release/release-close-public-proof-bridge-validation.json`
- Owner proof real input convergence artifact：`artifacts/final-release/owner-proof-real-input-convergence.json`
- Owner proof real input convergence validation：`artifacts/final-release/owner-proof-real-input-convergence-validation.json`
- Release close final owner runbook artifact：`artifacts/final-release/release-close-final-owner-runbook.json`
- Release close final owner runbook validation：`artifacts/final-release/release-close-final-owner-runbook-validation.json`
- Release issue close final owner decision audit：`artifacts/final-release/release-issue-close-final-owner-decision-audit.json`
- Release issue close final owner decision audit validation：`artifacts/final-release/release-issue-close-final-owner-decision-audit-validation.json`
- Final post-publish audit pack：`artifacts/final-release/final-post-publish-audit-pack.json`
- Final post-publish audit pack validation：`artifacts/final-release/final-post-publish-audit-pack-validation.json`
- Release docs and NuGet metadata audit：`artifacts/final-release/release-docs-and-nuget-metadata-audit.json`
- Release docs and NuGet metadata audit validation：`artifacts/final-release/release-docs-and-nuget-metadata-audit-validation.json`
- Post-publish user verification pack：`artifacts/final-release/post-publish-user-verification-pack.json`
- Post-publish user verification pack validation：`artifacts/final-release/post-publish-user-verification-pack-validation.json`
- Release candidate final freeze manifest：`artifacts/final-release/release-candidate-final-freeze-manifest.json`
- Release candidate final freeze manifest validation：`artifacts/final-release/release-candidate-final-freeze-manifest-validation.json`
- Public publish owner manual command handoff：`artifacts/final-release/public-publish-owner-manual-command-handoff.json`
- Public publish owner manual command handoff validation：`artifacts/final-release/public-publish-owner-manual-command-handoff-validation.json`
- Final release close blocker dashboard：`artifacts/final-release/final-release-close-blocker-dashboard.json`
- Final release close blocker dashboard validation：`artifacts/final-release/final-release-close-blocker-dashboard-validation.json`
- Public publish result owner input：`artifacts/final-release/public-publish-result-owner-input.template.json`
- Public publish result owner input validation：`artifacts/final-release/public-publish-result-owner-input-validation.json`
- Public publish result import：`artifacts/final-release/public-publish-result-import.json`
- Public publish result import validation：`artifacts/final-release/public-publish-result-import-validation.json`
- Post-publish clean consumer result convergence：`artifacts/final-release/post-publish-clean-consumer-result-convergence.json`
- Post-publish clean consumer result convergence validation：`artifacts/final-release/post-publish-clean-consumer-result-convergence-validation.json`
- StrictCloseReady convergence dashboard：`artifacts/final-release/strict-close-ready-convergence-dashboard.json`
- StrictCloseReady convergence dashboard validation：`artifacts/final-release/strict-close-ready-convergence-dashboard-validation.json`
- Public publish final owner execution pack：`artifacts/final-release/public-publish-final-owner-execution-pack.json`
- Public publish final owner execution pack validation：`artifacts/final-release/public-publish-final-owner-execution-pack-validation.json`
- Public publish command cross-check：`artifacts/final-release/public-publish-command-cross-check.json`
- Public publish command cross-check validation：`artifacts/final-release/public-publish-command-cross-check-validation.json`
- Release issue close owner decision input：`artifacts/final-release/release-issue-close-owner-decision-input.template.json`
- Release issue close owner decision input validation：`artifacts/final-release/release-issue-close-owner-decision-input-validation.json`
- Final evidence freeze non-proof audit：`artifacts/final-release/final-evidence-freeze-non-proof-audit.json`
- Final evidence freeze non-proof audit validation：`artifacts/final-release/final-evidence-freeze-non-proof-audit-validation.json`
- Public publish real result owner input contract：`artifacts/final-release/public-publish-real-result-owner-input-contract.json`
- Public publish real result owner input contract validation：`artifacts/final-release/public-publish-real-result-owner-input-contract-validation.json`
- Post-publish clean consumer proof record contract：`artifacts/final-release/post-publish-clean-consumer-proof-record-contract.json`
- Post-publish clean consumer proof record contract validation：`artifacts/final-release/post-publish-clean-consumer-proof-record-contract-validation.json`
- Release issue close strict owner decision import：`artifacts/final-release/release-issue-close-strict-owner-decision-import.json`
- Release issue close strict owner decision import validation：`artifacts/final-release/release-issue-close-strict-owner-decision-import-validation.json`
- Final close gate convergence：`artifacts/final-release/final-close-gate-convergence.json`
- Final close gate convergence validation：`artifacts/final-release/final-close-gate-convergence-validation.json`
- Public publish real result record draft：`artifacts/final-release/public-publish-real-result-record-draft.json`
- Public publish real result record draft validation：`artifacts/final-release/public-publish-real-result-record-draft-validation.json`
- Post-publish clean consumer proof record draft：`artifacts/final-release/post-publish-clean-consumer-proof-record-draft.json`
- Post-publish clean consumer proof record draft validation：`artifacts/final-release/post-publish-clean-consumer-proof-record-draft-validation.json`
- Public publish forbidden substitute scan：`artifacts/final-release/public-publish-forbidden-substitute-scan.json`
- Public publish forbidden substitute scan validation：`artifacts/final-release/public-publish-forbidden-substitute-scan-validation.json`
- Release close real proof import bridge：`artifacts/final-release/release-close-real-proof-import-bridge.json`
- Release close real proof import bridge validation：`artifacts/final-release/release-close-real-proof-import-bridge-validation.json`
- Final owner close readiness checkpoint：`artifacts/final-release/final-owner-close-readiness-checkpoint.json`
- Final owner close readiness checkpoint validation：`artifacts/final-release/final-owner-close-readiness-checkpoint-validation.json`
- Final release close record real validator：`artifacts/final-release/final-release-close-record-real-validator.json`
- Final release close record real validator validation：`artifacts/final-release/final-release-close-record-real-validator-validation.json`
- Final owner release close record projection：`artifacts/final-release/final-owner-release-close-record-projection.json`
- Final owner release close record projection validation：`artifacts/final-release/final-owner-release-close-record-projection-validation.json`
- Final release close hash consistency gate：`artifacts/final-release/final-release-close-hash-consistency-gate.json`
- Final release close hash consistency gate validation：`artifacts/final-release/final-release-close-hash-consistency-gate-validation.json`
- Final close owner approval boundary audit：`artifacts/final-release/final-close-owner-approval-boundary-audit.json`
- Final close owner approval boundary audit validation：`artifacts/final-release/final-close-owner-approval-boundary-audit-validation.json`
- Release candidate final publishability audit：`artifacts/final-release/release-candidate-final-publishability-audit.json`
- Release candidate final publishability audit validation：`artifacts/final-release/release-candidate-final-publishability-audit-validation.json`
- Release candidate owner action roadmap：`artifacts/final-release/release-candidate-owner-action-roadmap.json`
- Release candidate owner action roadmap validation：`artifacts/final-release/release-candidate-owner-action-roadmap-validation.json`
- Release candidate non-substitute final scan：`artifacts/final-release/release-candidate-non-substitute-final-scan.json`
- Release candidate non-substitute final scan validation：`artifacts/final-release/release-candidate-non-substitute-final-scan-validation.json`
- Release candidate final owner checklist：`artifacts/final-release/release-candidate-final-owner-checklist.json`
- Release candidate final owner checklist validation：`artifacts/final-release/release-candidate-final-owner-checklist-validation.json`
- Release proof readiness snapshot artifact：`artifacts/final-release/release-proof-readiness-snapshot.json`
- 发布候选冻结 summary artifact：`artifacts/release/release-candidate-freeze-summary.json`
- 最终审计地图：`docs/articles/zh-cn/release-final-audit-map.md`
- 对外介绍素材包：`docs/articles/zh-cn/release-public-story-pack.md`
- Owner proof backlog：`docs/articles/zh-cn/release-owner-proof-backlog.md`
- 不可替代 proof 清单：`docs/articles/zh-cn/release-proof-non-substitutes.md`
- 文章索引与推荐发布顺序：`docs/articles/zh-cn/release-article-index-and-publishing-order.md`
- 技术文章全量收口台账：`docs/articles/zh-cn/publishing/technical-article-closure-ledger.md`
- 技术文章基础第一批审计（2-6、10、15-16、19）：`docs/articles/zh-cn/publishing/technical-article-foundations-first-batch-audit.md`
- 技术文章基础第二批审计（28-32、38-45、103）：`docs/articles/zh-cn/publishing/technical-article-foundations-second-batch-audit.md`
- 技术文章 Proof Backlog（42 条 owner/runtime proof）：`docs/articles/zh-cn/publishing/technical-article-proof-backlog.md`
- README 前台入口检查清单：`docs/articles/zh-cn/release-readme-frontpage-checklist.md`
- Owner 最后一公里执行顺序：`docs/articles/zh-cn/release-final-owner-action-sequence.md`
- Release issue close record 校验器：`eng/Test-ReleaseIssueCloseRecord.ps1`
- Release issue close record 校验产物：`artifacts/final-release/release-issue-close-record-validation.json`
- 前台与 proof boundary 最终审计：`docs/articles/zh-cn/release-frontpage-and-proof-boundary-final-audit.md`
- 最终证据冻结 artifact：`artifacts/final-release/release-candidate-final-evidence-freeze.json`
- stale release claims audit artifact：`artifacts/final-release/stale-release-claims-audit.json`
- 发布候选最终总检：`docs/articles/zh-cn/release-candidate-final-cross-check.md`
- 发布候选文章矩阵总结：`docs/articles/zh-cn/release-candidate-article-matrix-summary.md`
- 发布候选发布总结：`docs/articles/zh-cn/release-candidate-publication-summary.md`
- 发布候选 final hold 与等待 Owner 状态：`docs/articles/zh-cn/release-candidate-final-hold-owner-waiting.md`
- Final hold owner 执行清单：`docs/articles/zh-cn/release-owner-action-checklist-final-hold.md`
- Release hold 最终巡检：`docs/articles/zh-cn/release-hold-final-inspection.md`

Owner 最短执行面以 `owner-release-execution-package` 的 `oneScreenReleaseHoldChecklist` 为准。新增的 execution package validator 只校验 owner 执行包形状，并继续保持 `performsPublish=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`。它镜像 5 个剩余 blocker，但仍只是 guidance：owner authorization、`package-consumer-runtime`、Linux runner proof、`real-model-runtime` 和 post-publish verification 仍需要真实记录与 validator；这些通过后还必须由 owner 回填最终 release issue close record，并通过 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`。在这些记录出现并全部通过前，`canCloseReleaseIssue=false` 必须保持不变。

`final-evidence-freeze` 会冻结当前 release evidence bundle、post-publish validation、release-close candidate validation、owner execution package validation 和 final close decision validation 的 SHA256。它是审计快照，不是 proof 晋级面。`release-issue-final-close-decision` 模板是最后的 Owner 输入合同，用于 rollback review、真实公开包源确认、仓库外 clean consumer 确认、runtime smoke exit code 和日志/hash 审阅；模板状态仍必须保持 `blocked-owner-final-close-decision-required`。

`real-external-proof-overlay-pack` 与 `release-issue-close-record-overlay-candidate` 现在把剩余真实 Owner 回填字段和 close record hash 映射集中到同一条最后一公里链路。二者仍然是 blocked/non-proof surface：在真实 post-publish proof、仓库外 clean consumer smoke、rollback approval、最终 Owner 决策和 strict close validation 全部通过前，必须继续保持 `performsPublish=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`。

`owner-external-execution-result-backfill-kit` 与 `owner-input-cross-hash-audit` 继续把这条链路推进到真实 Owner 外部执行结果回填和本地 cross-hash 一致性审计。kit 不能自行采集 proof、发布包、批准公开发布或关闭 release issue；audit 只能证明本地 artifact/path/hash 一致，hash 全匹配也不能替代真实外部 proof、post-publish verification、Owner approval 或 strict release-close validation。

`release-close-strict-record-candidate` 继续把最终关闭记录候选面做强：它绑定 evidence bundle、final freeze、post-publish validation、close candidate validation、final close decision、overlay validation、owner backfill kit 和 cross-hash audit，但在真实 Owner approval、post-publish proof、clean consumer runtime proof、rollback approval 以及 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady` 全部通过前，仍必须保持 blocked/non-proof。

`owner-proof-real-backfill-execution-pack` 把 strict candidate 继续拆成 owner input tasks、real proof tasks 和 hash check tasks，方便 Owner 按任务回填真实证据。它仍只是 handoff，必须保持 `performsPublish=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`。

`release-issue-close-record-real-input-map` 把这些 Owner 输入任务映射到最终 close record 字段和目标产物。它仍保持 blocked/non-proof，直到真实 Owner 输入和 strict close validation 通过。

`owner-real-proof-field-delta-pack` 会把 blocked strict candidate field contract 转成具体 Owner 字段 delta；`real-proof-candidate-promotion-guard` 则继续阻止 candidate 在 delta、non-substitute 检查和后续 real proof validator 通过前被误提升。二者仍是 blocked/non-proof surface，必须保持 `performsPublish=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`。

`real-proof-record-validator` 定义未来真实 proof record 的严格 validator contract；`owner-real-proof-execution-closure-pack` 把这些 contract 转成 Owner 执行闭环项，包含 first command、expected artifacts、logs、SHA256、validator commands 和 release-close follow-up。二者仍是 blocked/non-proof surface，不能发布、不能晋级 runtime proof、不能验证 post-publish 状态，也不能关闭 release issue。

`runtime-proof-execution-input-record`、`owner-runtime-proof-execution-runbook` 和 `release-close-strict-validation-bridge` 会继续把这条链路推进到 Owner 可填写执行输入、逐 lane 命令序列和严格关闭前置条件聚合。三者仍是 blocked/non-proof：placeholder 字段、runbook 命令、hash 槽位、bridge readiness flag 和本地 evidence 聚合都不能替代真实 runtime log、post-publish proof、rollback approval、final close decision 或 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`。

`owner-runtime-proof-result-input`、`runtime-proof-lane-dry-run-summary` 和 `release-close-strict-dry-run-summary` 继续补上 Owner 结果回填与严格 dry-run 层。`owner-external-proof-execution-bundle`、`owner-external-proof-execution-result-import`、`real-external-proof-record-import-validator` 和 `release-close-owner-input-bridge` 继续把这条链路推进到 Owner 外部执行命令、结果导入槽、真实 proof 导入合同和 release close owner gate 聚合。它们会暴露缺失的真实文件、hash、host/package metadata、validator output、reviewer 字段、lane blocker、close blocker 和 owner gate，同时必须保持 `canPromoteRuntimeProof=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`。

`owner-proof-real-input-convergence` 把剩余 Owner 输入、validator 和 proof blocker 收敛为一个 Owner 可读的校验矩阵。它仍不是 proof，也不能关闭 release issue。

`public-package-proof-owner-input`、`post-publish-proof-owner-confirmation` 和 `release-close-public-proof-bridge` 补上公开包 proof 准备层。它们会暴露 NuGet package source、GitHub Release asset path/hash、公开包 URL/hash、仓库外 clean consumer restore/build/smoke logs、stdout/stderr SHA256、host metadata、Owner review、post-publish proof gate 和最终 public proof bridge gate，但必须保持 `performsPublish=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`。

`release-close-final-owner-runbook` 把这个收敛矩阵转成最终 Owner 执行手册：公开包源确认、包 SHA256、仓库外 clean consumer runtime smoke、smoke log/hash/host metadata、rollback review、final close decision、strict candidate 刷新，以及 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`。它仍是 blocked/non-proof，必须保持 `performsPublish=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`。

`release-issue-close-final-owner-decision-audit` 和 `final-post-publish-audit-pack` 继续补上最终 Owner close decision 与发布后审计的 blocked 聚合层。它们只聚合 gate/lane，不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

`release-candidate-final-freeze-manifest`、`public-publish-owner-manual-command-handoff` 和 `final-release-close-blocker-dashboard` 补上最终 Owner handoff 层：冻结本地 artifact hash、列出手动发布占位命令、汇总剩余 close blockers，同时保持 `performsPublish=false`、适用处 `notExecutedByAutomation=true`、`canCloseReleaseIssue=false`。

`public-publish-result-owner-input`、`public-publish-result-import`、`post-publish-clean-consumer-result-convergence` 和 `strict-close-ready-convergence-dashboard` 继续把 Owner 真实公开发布结果回填、clean consumer proof 缺口和 StrictCloseReady 关闭条件收敛到同一条 blocked/non-proof 证据链。它们不执行发布、不上传包、不批准公开发布，也不关闭 release issue。

`public-publish-final-owner-execution-pack`、`public-publish-command-cross-check`、`release-issue-close-owner-decision-input` 和 `final-evidence-freeze-non-proof-audit` 补上最终人工执行与边界审计层。它们会在真实 Owner 公开发布结果、post-publish clean consumer proof、rollback review 和 strict close validation 回填前保持 release gate blocked；它们不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

`public-publish-real-result-owner-input-contract`、`post-publish-clean-consumer-proof-record-contract`、`release-issue-close-strict-owner-decision-import` 和 `final-close-gate-convergence` 继续补上真实发布后的 Owner 回填合同层。它们要求真实公开包来源、下载包 hash、仓库外 clean consumer smoke、rollback review、最终 Owner decision 和 strict close validation 全部到位后，才允许进入关闭晋级；它们仍不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

`public-publish-real-result-record-draft`、`post-publish-clean-consumer-proof-record-draft`、`public-publish-forbidden-substitute-scan`、`release-close-real-proof-import-bridge` 和 `final-owner-close-readiness-checkpoint` 继续补上真实 proof 回填执行层。它们暴露 Owner 草稿字段、禁止替代物 blocker、真实 proof import lane 和最终 readiness check，同时保持 blocked/non-proof；它们不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

`final-owner-strict-close-execution-order` 已归档最终 Owner 执行顺序，覆盖 7 个 action worklist、7 个 execution step、clean/post-publish runbook、10 个公开发布 lane、11 个命令交叉检查、12 个 readiness check、19 个 blocker 和 Owner 输入合同收敛。它仍然只是 blocked/non-proof guidance，不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

`real-owner-evidence-strict-validator-orchestration` 把真实 Owner 输入合同与 strict validator 联调起来，覆盖 11 个 source record 和 16 个 readiness 字段，包括公开包 URL/hash、clean consumer 日志、post-publish 日志、stdout/stderr、host metadata、non-substitute confirmation、rollback review 和 final close decision。它仍然保持 blocked/non-proof，拒绝 local feed、ProjectReference、direct nupkg、dry-run、dashboard、runbook、candidate 和 build-only 替代物，也不会执行发布或提升 proof。

`release-close-real-input-candidate-promotion-readiness` 继续补上 ReleaseClose 真实输入候选晋级层。它把 public package proof 到 final publish proof gate 汇总成 11 条 blocked promotion lane，并记录每条 lane 必须由哪些 strict validator 接受真实 Owner 输入后才能晋级。它保持 non-proof，不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push；`failedBlockerCount=0` 也不是 ready。

`final-release-close-record-real-validator`、`final-owner-release-close-record-projection`、`final-release-close-hash-consistency-gate` 和 `final-close-owner-approval-boundary-audit` 继续补上最终 ReleaseCloseRecord 真实验证层。它们显式投影 Owner 必填字段、关闭记录 lane、当前 hash 一致性和 Owner approval 边界，同时保持 blocked/non-proof；它们不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

`release-candidate-final-publishability-audit`、`release-candidate-owner-action-roadmap`、`release-candidate-non-substitute-final-scan` 和 `release-candidate-final-owner-checklist` 继续补上发布候选最终可发布性总检层。它们汇总可发布性 gate、Owner 执行顺序、禁止替代物检查和一页式 Owner checklist，同时保持 blocked/non-proof；它们不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

`owner-proof-backfill-execution-pack` 是这条 owner 执行面的聚焦命令与输入 companion。它逐项列出 `owner-authorization`、`package-consumer-runtime`、`linux-runner-proof`、`real-model-runtime`、`post-publish-verification` 和 `release-issue-close-record` 所需的真实字段、first command、validator、期望产物和不可替代 proof 类型，但仍然只是 guidance。

`owner-proof-execution-handoff` 把 backfill pack 转成 owner 可执行交接看板：每条 proof line 都包含 current state、owner next action、候选产物、缺失真实输入和 validator command。它仍然只是 guidance，不能发布包，也不能关闭 release issue。

`owner-external-proof-input-preflight` 用于在 release-close review 前预审 owner 候选输入：逐条把 proof line 分类为 missing、template-only、guidance-only、candidate-needs-owner-review 或 validator-passed-real-proof，同时继续保持 `canPublishPublicly=false` 和 `canCloseReleaseIssue=false`。

`owner-proof-input-repair-pack` 把这些 blocked preflight line 拆成字段级修复清单：placeholder、必须存在的文件、SHA256、clean consumer evidence、owner decision、rollback plan、first repair command 和 validator。repair pack 与 input draft 仍然不是 proof。

`owner-proof-input-draft-pack` 是这些 repair item 的非 proof 填写面：它列出每条 proof line 的 draft path、strict validator 和 proof-substitute blocker，同时保持所有 draft 不可晋级。

`owner-external-proof-backfill-orchestrator` 把 draft spec 转成 owner 可执行的真实外部 proof record 回填命令计划。它仍然只是 guidance，不能自行采集 proof、发布或关闭 release issue。

`package-consumer-runtime-proof-owner-input` 与 `release-issue-close-record-owner-input` 模板定义下一轮 candidate overlay 所需的 owner 回填字段。`package-consumer-runtime-proof-record` 会把这些字段投影为 strict runtime proof record，并可桥接到 `external-runtime-proof-record`；`package-consumer-external-smoke-scaffold` 生成仓库外 clean consumer 项目骨架。二者都必须等待真实 clean external consumer smoke evidence 通过验证后才能晋级。candidate 与 scaffold surface 仍保持 blocked/non-proof，不能发布、不能关闭 release issue，也不能替代 validator-passing real proof。

`release-proof-readiness-snapshot` 是同一条 proof 链的 5-blocker 紧凑状态视图，也只是 guidance：它不会发布包、不会上传资产、不会关闭 release issue。

`applications/TensorRtExec` 可以生成 ONNX build/precheck report，但 build-only、parse-only、sidecar-only、local feed、ProjectReference、collection bundle、runbook 和 `blocked-by-cuda-driver` 都不是 release proof record。`samples/YoloVision` 是统一 YOLO-family 样例，覆盖 YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom 与 det、cls、seg、obb、pose、sem；`real-model-runtime` 仍需要真实资产、日志、hash 和 validator。

YOLOv8n-seg 现已同时闭合 source-tree 真实运行与仓库外 bridge-only 本地包消费：后者只引用 managed API、YoloVision 和一个 `.Bridge` 包，比较 `1,793,600` 个 raw 值、四份 source-image mask 和独立 PyTorch IoU，并验证 raw reference 与 mask 篡改均 fail closed。入口为 `eng/Test-YoloVisionSegmentationLocalPackageConsumer.ps1`，轻量记录为 `samples/assets/yolovision-yolov8n-seg-local-package-consumer-runtime-evidence.json`。它仍不是公开包、post-publish、再分发授权或 release proof。

发布文章矩阵已经补齐 `YoloVision`、`OnnxToEngine` 和 `TensorRtExec` 的直接用户路径：建议从 `docs/articles/zh-cn/yolovision-sample-overview.md`、`docs/articles/zh-cn/onnx-to-engine-quickstart.md` 和 `docs/articles/zh-cn/tensorrtexec-cli-parameter-map.md` 开始。首批可直接用于公众号/博客扩写的正文也已经落地：`docs/articles/zh-cn/tensorrtsharp-4-project-overview-campaign.md`、`docs/articles/zh-cn/tensorrtsharp-4-architecture-abi-wrapper.md`、`docs/articles/zh-cn/cuda-tensorrt-cudnn-version-matrix-guide.md`、`docs/articles/zh-cn/tensorrtsharp-nuget-runtime-package-guide.md`、`docs/articles/zh-cn/windows-installation-and-troubleshooting-guide.md`、`docs/articles/zh-cn/linux-installation-runner-boundary-guide.md`、`docs/articles/zh-cn/tensorrtexec-gui-cli-parity-design.md`、`docs/articles/zh-cn/plugin-registry-inventory-readonly-design.md`、`docs/articles/zh-cn/package-consumer-runtime-proof-clean-consumer-guide.md`、`docs/articles/zh-cn/release-proof-and-post-publish-verification-guide.md`、`docs/articles/zh-cn/tensorrtsharp-4-faq.md`、`docs/articles/zh-cn/tensorrtsharp-4-release-story.md`。第二批正文继续补强用户采用路径：`docs/articles/zh-cn/cuda-memory-wrapper.md`、`docs/articles/zh-cn/cuda-stream-event-multistream-tutorial.md`、`docs/articles/zh-cn/dynamic-shape-optimization-profile-tutorial.md`、`docs/articles/zh-cn/onnx-parser-to-serialized-engine-tutorial.md`、`docs/articles/zh-cn/network-layer-coverage-guide.md`、`docs/articles/zh-cn/blog-refit-weights-guide.md`、`docs/articles/zh-cn/tensorrtexec-cli-parameter-map.md`、`docs/articles/zh-cn/yolovision-sample-overview.md`、`docs/articles/zh-cn/yolo-vision-model-matrix.md`、`docs/articles/zh-cn/local-nuget-feed-consumer.md`、`docs/articles/zh-cn/linux-runner-evidence-checklist.md`、`docs/articles/zh-cn/runtime-packages.md`。它们是文档和采用路径，不是 public package proof、post-publish proof、package push 或 release close approval。

第三批正文继续向真实案例和宣发长文靠近：`docs/articles/zh-cn/yolovision-detection-real-model-tutorial.md`、`docs/articles/zh-cn/yolovision-classification-real-model-tutorial.md`、`docs/articles/zh-cn/yolovision-segmentation-real-model-tutorial.md`、`docs/articles/zh-cn/yolovision-pose-obb-sem-roadmap.md`、`docs/articles/zh-cn/tensorrtexec-gui-user-guide.md`、`docs/articles/zh-cn/tensorrtexec-trtexec-parity-deep-dive.md`、`docs/articles/zh-cn/onnx-to-engine-trtexec-conversion-guide.md`、`docs/articles/zh-cn/runtime-package-installation-deep-dive.md`、`docs/articles/zh-cn/cuda-error-35-troubleshooting.md`、`docs/articles/zh-cn/deferred-api-real-completion-review.md`、`docs/articles/zh-cn/plugin-inventory-readonly-api.md`、`docs/articles/zh-cn/package-consumer-runtime-proof-playbook.md`。它们仍是文档和采用路径，不是 runtime proof、public package proof、post-publish proof、package push 或 release close approval。

第四批正文继续补强可执行采用路径和发布边界解释：`docs/articles/zh-cn/yolovision-detection-yolov8n-download-export-run.md`、`docs/articles/zh-cn/yolovision-segmentation-mask-postprocess-guide.md`、`docs/articles/zh-cn/yolovision-pose-keypoint-output-guide.md`、`docs/articles/zh-cn/yolovision-obb-angle-output-guide.md`、`docs/articles/zh-cn/tensorrtexec-winforms-screenshot-walkthrough.md`、`docs/articles/zh-cn/tensorrtexec-report-schema-guide.md`、`docs/articles/zh-cn/runtime-package-windows-linux-install-faq.md`、`docs/articles/zh-cn/plugin-registry-inventory-user-guide.md`、`docs/articles/zh-cn/deferred-readonly-api-upgrade-playbook.md`、`docs/articles/zh-cn/csharp-wrapper-lifetime-design.md`、`docs/articles/zh-cn/release-evidence-non-substitute-guide.md`、`docs/articles/zh-cn/project-roadmap-to-public-release.md`。它们仍是文档和采用路径，不是 runtime proof、public package proof、post-publish proof、package push 或 release close approval。

第五批正文继续补上 schema、validator 和 owner checklist 路径：`docs/articles/zh-cn/yolovision-output-json-schema-guide.md`、`docs/articles/zh-cn/yolovision-real-asset-record-template-guide.md`、`docs/articles/zh-cn/yolovision-owner-asset-evidence-guide.md`、`docs/articles/zh-cn/yolovision-golden-output-validation-guide.md`、`docs/articles/zh-cn/tensorrtexec-report-json-schema-snapshot.md`、`docs/articles/zh-cn/tensorrtexec-report-proof-boundary.md`、`docs/articles/zh-cn/onnx-to-engine-trtexec-proof-boundary.md`、`docs/articles/zh-cn/callback-allocator-listener-readonly-safety-gates.md`、`docs/articles/zh-cn/tensorrtexec-gui-cli-field-map.md`、`docs/articles/zh-cn/runtime-package-minimal-smoke-commands.md`、`docs/articles/zh-cn/runtime-package-native-load-troubleshooting.md`、`docs/articles/zh-cn/plugin-inventory-field-metadata-smoke-guide.md`、`docs/articles/zh-cn/deferred-next-readonly-candidate-list.md`、`docs/articles/zh-cn/csharp-public-api-handle-exposure-audit.md`、`docs/articles/zh-cn/release-proof-strict-validator-playbook.md`、`docs/articles/zh-cn/release-proof-owner-input-dashboard.md`、`docs/articles/zh-cn/release-proof-sample-article-closure.md`、`docs/articles/zh-cn/public-release-owner-final-checklist.md`。它们仍是文档、validator guidance 和采用路径，不是 runtime proof、public package proof、post-publish proof、package push 或 release close approval。

发布前总检矩阵已经机器可读化：`samples/YoloVision/yolo-model-matrix.json`、`samples/OnnxToEngine/trtexec-parity-matrix.json`、`applications/TensorRtExec/tensor-rt-exec-feature-matrix.json`、`artifacts/cuda-runtime-compilation/capability-matrix.json`、`artifacts/cuda-runtime-compilation/local-smoke.json`、`artifacts/interface-coverage/release-api-readiness-audit.json`、`artifacts/final-release/release-proof-owner-input-dashboard.json`、`artifacts/final-release/owner-proof-execution-checklist.json`、`artifacts/final-release/post-publish-clean-consumer-owner-input.template.json`、`artifacts/final-release/yolovision-owner-asset-evidence.template.json`、`artifacts/final-release/yolovision-owner-asset-evidence.example.json` 和 `docs/articles/zh-cn/publishing/article-roadmap-30plus.json`。

Owner 执行指南继续新增：`docs/articles/zh-cn/owner-proof-execution-checklist.md`、`docs/articles/zh-cn/post-publish-clean-consumer-owner-input-guide.md` 和 `docs/articles/zh-cn/yolovision-owner-asset-evidence-example.md`。这些仍是 owner 输入指导和示例，不是 runtime proof、public package proof、post-publish proof、package push 或 release close approval。

Strict close 准备继续新增：`docs/articles/zh-cn/owner-real-proof-field-delta-dashboard.md`、`docs/articles/zh-cn/release-close-strict-gate-dashboard.md` 和 `docs/articles/zh-cn/owner-proof-import-preflight.md`，对应 `artifacts/final-release/owner-real-proof-field-delta-dashboard.json`、`artifacts/final-release/release-close-strict-gate-dashboard.json`、`artifacts/final-release/owner-proof-import-preflight.json`。这些 dashboard 仍是 blocked guidance，只有 owner 真实 proof 通过 strict validators 后才能进入 release close 审核。

ReleaseCandidate 公开材料终检继续新增：`docs/articles/zh-cn/release-candidate-freeze-manifest.md`、`docs/articles/zh-cn/public-material-final-scan.md` 和 `docs/articles/zh-cn/final-owner-proof-blocker-dashboard.md`，对应 `artifacts/final-release/release-candidate-freeze-manifest.json`、`artifacts/final-release/public-material-final-scan.json`、`artifacts/final-release/final-owner-proof-blocker-dashboard.json`。这些记录只冻结公开材料、扫描旧样例名和 proof overclaim、汇总 owner blocker；它们不发布包，不生成 runtime proof，不生成 post-publish proof，也不能关闭 release issue。

Owner 真实 Proof 导入审计继续新增：`docs/articles/zh-cn/owner-real-proof-import-audit-bundle.md`、`docs/articles/zh-cn/owner-evidence-file-manifest-template.md`、`docs/articles/zh-cn/strict-validator-command-runbook.md` 和 `docs/articles/zh-cn/release-issue-close-owner-input-final-checklist.md`，对应 `artifacts/final-release/owner-real-proof-import-audit-bundle.json`、`artifacts/final-release/owner-evidence-file-manifest.template.json`、`artifacts/final-release/strict-validator-command-runbook.json`、`artifacts/final-release/release-issue-close-owner-input-final-checklist.json`。这些文件只组织 owner 路径、hash、validator 命令和 close 输入，继续保持 `canPublishPublicly=false` 与 `canCloseReleaseIssue=false`。

## 当前验证状态

截至 2026-06-12：

- TensorRT interface coverage matrix：`0` 个 missing rows。
- CUDA runtime interface coverage matrix：`0` 个 missing rows。
- Manifest API inventory：`3271` 条 API 记录，分布在 `102` 份 manifests 中。
- 最新覆盖报告：`artifacts/interface-coverage/interface-coverage-summary.md`。
- 高版本原生验证：`win-x64-trt11-cuda13-release` 可配置并可编译。
- 托管验证：solution build 和 project quality tests 通过。
- DocFX 验证：文档构建为 `0` warning、`0` error。
- 当前发布 proof 边界：package-consumer runtime proof 仍需要兼容 CUDA host 的真实记录；`blocked-by-cuda-driver`、runbook、collection bundle、dependency probe、local feed、ProjectReference、bridge-only 日志、`Skipped=True`、mismatched log SHA256、sidecar-only report 和 build-only report 都不是 smoke passed evidence。

当前仓库已经为本工作区扫描到的 TensorRT 8/10/11 和 CUDA 11/12/13 头文件建立 manifest/native-source 覆盖。一部分高风险或低频 CUDA runtime API 被明确记录为 deferred boundary，而不是包装成高层托管 API。CUDA library metadata、texture/surface descriptor、primary execution context、IPC event/memory export token 以及无指针 graph memory allocation/free 流程已有 owner-safe copied 或 bridge-owned 路径；callback 生命周期、裸 driver entrypoint、external resource import、IPC open/close ownership、borrowed device pointer 和 user-object destructor ownership继续 deferred。

## 当前阶段

原始接口追平阶段已完成。当前主线是发布硬化：

- 持续保持覆盖矩阵、build、tests 和 DocFX 绿色。
- 保持 sample runners 准确、可复现。
- 对依赖外部资产的 sample 目录保留 README/roadmap，不放空壳目录。
- 验证 runtime-package asset collection、package consumer restore/build/run 路径。
- 只有在 ABI、所有权、版本保护和托管生命周期清晰时，才把 deferred boundary 晋升为高层 public API。

## 部署验证路径

本地 Windows 部署 sanity check 推荐顺序：

```powershell
dotnet restore .\TensorRtSharp.sln
powershell -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
dotnet build .\TensorRtSharp.sln -c Debug --no-restore
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build
dotnet docfx .\docs\docfx.json
```

如果修改了 manifest、native 或 generated 文件，还应运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
powershell -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
```

本轮还补跑了 TensorRT 8 兼容分支：

```powershell
cmake --build --preset win-x64-trt8-cuda11-release --parallel
cmake --preset win-x64-trt8-cuda12-release
cmake --build --preset win-x64-trt8-cuda12-release --parallel
```

推荐验证顺序：

1. `CudaSmokeRunner`
2. `TensorRtSmokeRunner`
3. `LifecycleSmokeRunner`
4. `OnnxToEngineSmokeRunner`
5. `NetworkBuilderSmokeRunner`
6. 各类 layer-specific network runners

常用用户示例请从 `samples/README.md` 进入，例如 `MultiStream`、`DynamicShape`、`InferenceBindings`、`OnnxToEngine`、`Classification` 和 `YoloVision`。

## Samples

可运行部署示例位于 `samples/`，入口清单见 `samples/README.md`。

近期 sample 成熟度状态：

- `MultiStream` 是真实 CUDA multi-stream/event ordering 示例，并已纳入 solution。
- `CudaRuntimeCompilation` 是 owner-safe NVRTC 编译与 Runtime-library/Driver-module 双路径 named typed-kernel launch/readback 样例，覆盖复制型 PTX/CUBIN/LTO IR、失败日志、lowered name、确定性、owner 提前释放和逐值 GPU correctness；仓库外 local-feed consumer 已通过 managed/bridge-only `PackageReference` 包复验 CUDA 12.9 双路径，CUDA 13.2 PTX load rejection、Linux、public package 与 post-publish 仍单独记录。
- `DynamicShape` 是真实 TensorRT dynamic-shape/profile/binding 示例，并已纳入 solution。
- `InferenceBindings` 是真实 TensorRT inference-binding 示例，并已纳入 solution。
- `OnnxToEngine` 现在是可运行的常用 ONNX-to-engine 示例，并已纳入 solution。
- `Classification` 和 `YoloVision` 是依赖用户自备 ONNX 模型、labels 和 input-shape metadata 的可运行示例。
- `applications/TensorRtExec` 是面向用户的 ONNX-to-engine CLI/WinForms 工具。它可以为外部 ONNX 生成 build/precheck report，但真实模型 runtime proof 仍归具体 sample runner，package-consumer-runtime proof 仍归 release proof record。
- custom-kernel launch 已提供两条 owner-safe named-kernel/typed-argument 路径：`CudaKernelLibrary.Launch(...)` 使用 CUDA 12.9+ Runtime library，`CudaDriverModule.Launch(...)` 动态加载 CUDA Driver 并统一 module owner。本机 Driver 12090 已验证 CUDA 11.8/12.1/12.9 PTX 的 launch/readback。bridge-only 包只包含 `jyppxtrtbridge.dll`，不捆 NVRTC/builtins；CUDA 12.9 clean local-feed consumer 在无 `ProjectReference`、无 `JYPPX_NATIVE_BRIDGE_PATH` 下验证缺失 NVRTC 诊断与双路径 correctness，但仍只是本地 candidate，不是 public-package/post-publish proof。

## Runtime Packages

自 2026-07-30 起，不再打包或发布 NVIDIA 原厂运行库。正式发布物只保留 C# 托管接口包、按版本编译的项目自有 `.Bridge` 包，以及由 Git 跟踪文件生成的源码归档。CUDA、cuDNN、TensorRT 和可选 NVRTC 由用户自行安装。

下面的 runtime key 继续作为 bridge 编译兼容矩阵，用于选择用户本机的 header 与 import library，不再表示可复制进包内的 vendor DLL 或 `.so`。

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 `
  -SourceRuntimeKey win-x64-trt11.0-cuda12.9-cudnn9.22 `
  -Version 4.0.0 `
  -SplitPackageRole bridge

powershell -ExecutionPolicy Bypass -File .\eng\Test-ExternalVendorRuntimePackagePolicy.ps1 `
  -PackagePath .\artifacts\runtime-split-nupkg\win-x64-trt11.0-cuda12.9-cudnn9.22
```

完整策略见 `docs/articles/zh-cn/external-vendor-runtime-package-policy.md` 与 `pack/external-vendor-runtime-policy.json`。

<details>
<summary>历史 full-runtime 说明（已退休，不得用于发布）</summary>

runtime 包为一个明确 TensorRT / CUDA / cuDNN 组合承载原生部署资产。当前 Windows runtime package keys：

- `win-x64-trt8.6-cuda11.8-cudnn8.9`
- `win-x64-trt8.6-cuda12.1-cudnn8.9`
- `win-x64-trt10.11-cuda11.8-cudnn8.9`
- `win-x64-trt10.11-cuda12.9-cudnn9.22`
- `win-x64-trt11.0-cuda12.9-cudnn9.22`
- `win-x64-trt11.0-cuda13.2-cudnn9.22`

当前本地状态：

- TensorRT 10 + CUDA 11.8 是稳定的真实 vendor-backed smoke 路径，已有 package-consumer smoke 证据。
- TensorRT 10 + CUDA 12.9 和 TensorRT 11 + CUDA 12.9 已完成本地 runtime/package 验证和 package-consumer smoke。
- TensorRT 11 + CUDA 13.2 bridge 可编译、可收集 assets、可打包，并通过 package consumer restore/build/native-copy；runtime/builder smoke 仍等待 CUDA 13-capable driver/runtime 环境。
- Linux runtime 包名现在显式包含系统版本和架构。Ubuntu 22.04 x64 是默认 hosted 矩阵并覆盖 6 个组合；Ubuntu 24.04 x64 只覆盖 NVIDIA 官方仓库已提供的 TensorRT 10/11 现代组合；Ubuntu 20.04 x64 通过 hosted-container 发布线使用 `ubuntu:20.04` job container。

相关文档：

- `docs/articles/zh-cn/runtime-packages.md`
- `docs/articles/zh-cn/runtime-distribution-strategy.md`
- `docs/articles/zh-cn/package-consumer-validation.md`
- `docs/articles/zh-cn/release-candidate-gate.md`
- `docs/articles/zh-cn/api-reference.md`

</details>

## 发布自动化

正式发布只允许在 `guojin-yan` 仓库执行。`grape-yan` 仓库仅用于日常 build/test，workflow 权限为只读，不包含 package push 或 Release upload job。

当前发布工作流只处理：

- `package-managed.yml`：`JYPPX.TensorRT.CSharp.API` 与纯 managed 扩展 `JYPPX.TensorRT.CSharp.API.YoloVision`；
- `runtime-windows.yml` / `runtime-linux.yml`：`split_package_roles=bridge` 的 `.Bridge` 包；
- `package-source.yml`：仅含 Git 跟踪文件的源码归档。

所有上传路径都会运行 `eng/Test-ExternalVendorRuntimePackagePolicy.ps1`。managed workflow 还会强制基础 API + YoloVision 两包的精确 ID/版本 allowlist、nuspec source commit 对齐、YoloVision surface audit 和仓库外纯 managed consumer。`release-bundle.yml` 的部署、package publish 与 Release attach 默认全部为 `false`；任何远端副作用还必须在正式仓库显式传入 `owner_publish_approved=true`。

<details>
<summary>Bridge-only 发布自动化说明</summary>

这个仓库支持两种发布执行方式：

1. 用 `gh` 从 GitHub 远端触发工作流，再由本机的 self-hosted Windows runner 执行 Windows runtime 打包。
2. 直接运行本地脚本，做纯工作站上的验证闭环，不在 GitHub Actions 中留下运行记录。

也可以用 `act` 在本机做 workflow dry-run，例如解析 `release-bundle.yml` 或 `runtime-linux.yml` 的调度图。`act` 适合做轻量检查，但不能替代正式发布证据：Windows hosted job 不能被 Linux 容器可靠复刻，self-hosted runtime job 仍依赖真实本机/runner 上的 CUDA、cuDNN、TensorRT 和签名环境。详见 `docs/articles/zh-cn/local-actions.md`。

基础 managed、YoloVision 与 bridge 可以独立构建，但同一发布 handoff 必须使用同一版本和源码提交，并记录三个 nupkg 的 SHA256。CUDA、cuDNN、TensorRT 和 NVRTC 由 consumer 自行安装，不再上传到 GitHub Packages 或 GitHub Releases。日常 Actions 先在 `grape-yan` 运行零发布 dry-run；正式发布只在 `guojin-yan` 执行。

历史远端 vendor 包清理结果：

| 项目 | 结果 |
| --- | --- |
| Owner 确认指纹 | `sha256:bbd87a8e018ca4b3dc62d1382a7c710e4f7548ec958c9e6f2aaee45f7ec05b97` |
| 已删除 | 65 个 GitHub Package versions + 65 个 Release assets |
| 删除后候选 | 0 |
| 保留 | managed、`.Bridge` 和 GitHub 自动源码归档 |

历史 package identity 只保留给清理与证据解释，不能重新 pack、push 或上传。

远端 managed bundle 发布示例：

```powershell
gh workflow run release-bundle.yml `
  --ref TensorRtSharp4.0 `
  -f version=4.0.1 `
  -f owner_publish_approved=true `
  -f publish_managed_to_nuget=true `
  -f publish_managed_to_github_packages=true
```

远端构建/发布 Windows bridge 示例：

Windows 远端示例默认使用完整 6 组合矩阵，不传 `windows_runtime_keys`，使用默认 Windows 6 组合矩阵。只有调试或修复单一依赖线时才单独传 key。

```powershell
gh workflow run release-bundle.yml `
  --ref TensorRtSharp4.0 `
  -f version=4.0.0 `
  -f runtime_version=4.0.0 `
  -f owner_publish_approved=true `
  -f run_windows_runtime_packaging=true `
  -f windows_runtime_delivery_mode=split `
  -f windows_split_package_roles=bridge `
  -f publish_runtime_to_github_packages=true `
  -f attach_runtime_to_github_release=true
```

单个 key 的 bridge 诊断示例：

```powershell
gh workflow run release-bundle.yml `
  --ref TensorRtSharp4.0 `
  -f version=4.0.1 `
  -f runtime_version=4.0.1 `
  -f run_windows_runtime_packaging=true `
  -f windows_runtime_delivery_mode=split `
  -f windows_runtime_keys=<runtime-key> `
  -f windows_split_package_roles=bridge `
  -f publish_runtime_to_github_packages=false `
  -f attach_runtime_to_github_release=false
```

本地 managed bundle 示例：

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Invoke-LocalReleaseBundle.ps1 `
  -Version 4.0.1 `
  -SkipWindowsRuntime
```

本地构建 bridge 示例：

下面的示例用单个 `<runtime-key>` 做快速迭代；正式发布时应省略 `windows_runtime_keys`，或一次性传入 6 个 key。

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 `
  -SourceRuntimeKey <runtime-key> `
  -Version 4.0.1 `
  -SplitPackageRole bridge

powershell -ExecutionPolicy Bypass -File .\eng\Test-ExternalVendorRuntimePackagePolicy.ps1 `
  -PackagePath .\artifacts\runtime-split-nupkg\<runtime-key>
```

`release-bundle.yml` 默认不触发 docs deploy、runtime 打包、package publish 或 Release attach。需要 runtime validation 时显式设置 `run_windows_runtime_packaging=true` 或 `run_linux_runtime_packaging=true`；只有同时启用 publish/attach 参数时才需要 `owner_publish_approved=true`。如果启用 Linux runtime 但 `linux_runtime_keys` 为空，Linux 模块会干净 no-op。

发布到 `nuget.org` 时，仓库 secret `NUGET_API_KEY` 应填写 NuGet 官网生成的纯文本 ASCII API key。这个 key 必须仍然有效，并且必须对 `JYPPX.TensorRT.CSharp.API`、`JYPPX.TensorRT.CSharp.API.YoloVision` 两个 package ID或其所属账号/组织拥有 push 权限。managed-package workflow 会在发布前校验该 secret；不要把加密后的本机凭据或机器导出的 token 片段填进 `NUGET_API_KEY`。如果推送阶段返回 nuget.org `403`，说明 key 无效、过期或没有对应包 ID 的权限，需要用包 owner 账号重新生成有 scope 的 key 后再重跑 managed bundle workflow。

</details>

## 仓库布局

```text
build/      CMake modules and build helpers
docs/       DocFX site and conceptual documentation
eng/        automation and dependency discovery scripts
native/     C ABI bridge and TensorRT/CUDA adapters
pack/       NuGet packaging projects
samples/    user-facing common examples and documented sample roadmaps
smoke/      validation runners for release gates, packaging, and regression checks
src/        managed libraries
tests/      managed integration and unit tests
third_party/local dependency drop folder (not committed)
```

## 构建前置条件

- .NET SDK 10.0.300 或更高版本。
- CMake 3.27 或更高版本。
- Windows 上需要 Visual Studio C++ toolchain。
- native build 和 runtime-package validation 需要匹配的本地 TensorRT / CUDA / cuDNN roots。

## 快速开始

```powershell
dotnet restore .\TensorRtSharp.sln
dotnet build .\TensorRtSharp.sln -c Debug --no-restore
powershell -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build
```

构建当前高版本 native preset：

```powershell
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
```

## 依赖发现

使用以下脚本检查本地 TensorRT/CUDA/cuDNN roots：

- `eng/Get-Dependencies.ps1`
- `eng/get-dependencies.sh`

Windows 本地 root 不写入公开 runtime manifest。请使用 `pack/runtime/runtime-packages.local.json` 保存机器本地覆盖配置；可从 `pack/runtime/runtime-packages.local.example.json` 复制后修改。

审计本机 NVRTC/builtins 身份时可以运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-CudaRtcFullRuntimePackagingPreflight.ps1
```

该脚本只做 host dependency identity audit，不复制、不打包、不发布。`cuda-rtc` role 已固定为 `retired-not-packable`；`.Bridge` 包永远不携带 NVRTC 或 matching builtins。

托管 runtime loading 面向生产部署：

- 常规探测检查 app base directory 和 `runtimes/<rid>/native`。
- 显式桥接库路径使用 `JYPPX_NATIVE_BRIDGE_PATH`。
- 显式 vendor roots 使用 `JYPPX_TENSORRT_ROOT` 和 `JYPPX_CUDA_ROOT`。
- 本地 `build-out` / `third_party` 开发扫描需要 `JYPPX_ENABLE_DEVELOPMENT_PROBING=1`。

## 文档入口

- `docs/index.md`
- `docs/articles/zh-cn/getting-started.md`
- `docs/articles/zh-cn/installation-layout.md`
- `docs/articles/zh-cn/api-coverage-and-deferred-boundaries.md`
- `docs/articles/zh-cn/sample-runners.md`
- `docs/articles/zh-cn/runtime-packages.md`
- `docs/articles/zh-cn/package-consumer-validation.md`
- `docs/articles/zh-cn/release-candidate-gate.md`

构建文档：

```powershell
dotnet docfx .\docs\docfx.json
```

## 注意事项

- NVIDIA 二进制文件不提交到仓库。
- `third_party/` 仅作为本地依赖投放目录。
- runtime 包只包含项目自有 bridge；用户需要在消费端按版本自行安装 TensorRT、CUDA、cuDNN 和可选 NVRTC。
- 手写 public C# wrapper 应包含有用的 XML documentation；生成 API 可以使用生成注释。

`public-release-owner-execution-package`、`external-clean-consumer-proof-kit`、`runtime-proof-compatible-host-kit`、`post-publish-owner-verification-kit` 和 `owner-public-release-execution-readiness-pack` 继续补上真实公开发布 Owner 执行包层。它们只提供命令模板、Owner 字段、外部 proof 采集路径和 readiness blocker；默认保持 blocked/non-proof，不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

`owner-external-real-proof-input-contract`、`owner-external-real-proof-import-validator`、`post-publish-clean-consumer-real-proof-gate`、`runtime-compatible-host-real-proof-gate` 和 `release-close-real-proof-readiness-gate` 继续补上外部真实 proof 回填与 ReleaseClose 准入层。它们只校验 Owner 回填字段、公开源/hash/log/host metadata、仓库外 clean consumer proof、兼容主机 runtime proof 和最终 close readiness；默认保持 blocked/non-proof，不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

`release-candidate-real-proof-final-freeze`、`owner-real-input-import-preflight`、`public-package-hash-cross-check-gate`、`clean-consumer-runtime-proof-cross-check-gate`、`post-publish-rollback-owner-decision-gate` 和 `release-close-final-real-input-admission-pack` 继续补上最终真实 Owner 输入准入层。它们只冻结本地证据 hash、预检 Owner 输入、交叉核对公开包 hash、clean consumer/runtime proof metadata、rollback Owner 决策和最终 ReleaseClose blocker；默认保持 blocked/non-proof，不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

`release-close-real-input-candidate-promotion-readiness` 继续补上最终候选晋级 readiness 层。它要求 11 条 promotion lane 先拿到真实 Owner 输入并通过对应 strict validator，继续拒绝 local feed、ProjectReference、direct nupkg、dry-run、dashboard、runbook、candidate、draft 和 build-only 替代物。

`owner-real-input-json-contract`、`owner-real-input-json-import`、`owner-real-input-hash-and-path-validator`、`owner-real-input-forbidden-substitute-validator`、`strict-close-real-input-dry-run`、`strict-close-real-input-finding-report`、`strict-close-owner-action-pack` 和 `release-close-real-input-final-blocker-ledger` 继续补上 StrictClose 真实输入验证层。它们只定义并本地校验 Owner 输入字段、报告 blocker、生成 Owner 行动指引；默认保持 blocked/non-proof，不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

`clean-external-package-consumer-owner-runbook` 和 `post-publish-owner-verification-runbook` 将剩余 Owner 工作拆成可直接执行的仓库外 consumer 清单。它们要求仓库外工程、公开或 Owner 批准的包源、stdout/stderr/merged transcript hash、包 hash、validator 输出 hash、host metadata、Owner review 和 non-substitute confirmations；在 Owner 回填真实执行结果并通过 strict validator 晋级前，仍然保持 blocked/non-proof。

`clean-consumer-proof-execution-bundle` 和 `clean-consumer-external-proof-closure-pack` 把 local smoke classification、仓库外 clean consumer 执行、compatible-host runtime metadata、Owner result import、post-publish clean consumer evidence 和 strict close admission 串成同一条 blocked owner-action 链路。它们只是执行映射和收敛包：不会运行 runtime smoke、不会发布包、不会批准公开发布、不会关闭 release issue，也不能在缺少真实外部 logs、SHA256、package metadata、native asset evidence、host metadata、Owner review 以及 `-RequireExistingLog` / `-FailOnNotProof` 严格 validator 通过前提升 runtime/post-publish proof。
