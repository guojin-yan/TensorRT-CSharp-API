# TensorRtSharp4.0

TensorRtSharp4.0 is the next-generation workspace for a production-oriented TensorRT and CUDA bridge on .NET.

## Scope

- Managed assemblies: `JYPPX.TensorRtSharp`, `JYPPX.CudaSharp`
- Public namespace roots: `JYPPX.TensorRtSharp` and `JYPPX.CudaSharp`; shared bridge types use `JYPPX.TensorRtSharp.Shared`.
- NuGet package: `JYPPX.TensorRT.CSharp.API`
- Native bridge library: `jyppxtrtbridge`
- First release targets: Windows x64 and Linux x64
- TensorRT lines: 8.x, 10.x, and 11.x
- CUDA lines: 11.x, 12.x, and 13.x

## Release Candidate Front Door

Current release-facing state:

- Evidence freeze state: `blocked-real-proof-required`.
- Publication automation state: `performsPublish=false`.
- Public channel approval state: `canPublishPublicly=false`.
- Release issue close state: `canCloseReleaseIssue=false`.
- Release issue close record validation: `release-issue-close-record-validation=blocked-template-only`; `release-issue-close-record-template.json` is not proof.
- Final quality freeze state: `blocked-final-quality-freeze-real-proof-required`; `final-quality-freeze-dashboard` is a non-proof dashboard, not publication approval.
- Public proof claim boundary audit: `public-proof-claim-boundary-audit-passed`; it scans stale public claims but is not runtime proof, post-publish proof, package push, or release close approval.
- Article roadmap 30+ validation: `article-roadmap-30plus-validation-passed-non-proof-planning`; the roadmap is content planning only.
- Owner real input landing pack state: `blocked-owner-real-input-required`; it maps the five final blockers to real Owner files, fields, strict validators, and forbidden substitutes.
- Final Owner execution checklist state: `blocked-final-owner-execution-checklist-real-owner-input-required`; it is the shortest manual execution/backfill path and does not run `dotnet nuget push`.
- Real proof import boundary audit: `real-proof-import-boundary-audit-passed`; it scans public/final-release surfaces for forbidden substitute proof claims but is not proof.
- Remaining owner proof blockers: owner authorization, `package-consumer-runtime`, Linux runner proof, `real-model-runtime`, and post-publish verification.
- Owner final backfill tracks are fixed as `package-consumer-runtime`, `linux-runner-proof`, `real-model-runtime`, and `post-publish verification`; `ownerProofFinalBackfillTracks` is an execution map, not proof.
- Non-substitute evidence remains non-promotable: local feed, ProjectReference, bridge-only logs, `Skipped=True`, mismatched log SHA256, build-only/precheck output, sidecar-only reports, runbooks, collection packages, and Windows handoff for Linux proof.

Use these front-door documents when evaluating the project:

- Owner one-screen Release Hold checklist: `docs/articles/zh-cn/owner-release-execution-package.md`
- Owner release execution package artifact: `artifacts/final-release/owner-release-execution-package.json`
- Owner release execution package validation: `artifacts/final-release/owner-release-execution-package-validation.json`
- Owner proof backfill execution pack artifact: `artifacts/final-release/owner-proof-backfill-execution-pack.json`
- Owner proof execution handoff artifact: `artifacts/final-release/owner-proof-execution-handoff.json`
- Owner external proof input preflight artifact: `artifacts/final-release/owner-external-proof-input-preflight.json`
- Owner proof input repair pack artifact: `artifacts/final-release/owner-proof-input-repair-pack.json`
- Owner proof input draft pack artifact: `artifacts/final-release/owner-proof-input-draft-pack.json`
- Owner external proof backfill orchestrator artifact: `artifacts/final-release/owner-external-proof-backfill-orchestrator.json`
- Package consumer runtime proof candidate artifact: `artifacts/final-release/package-consumer-runtime-proof-candidate.json`
- Package consumer runtime proof owner input template: `artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json`
- Package consumer runtime proof record validation: `artifacts/final-release/package-consumer-runtime-proof-record-validation.json`
- Package consumer external smoke scaffold: `artifacts/final-release/package-consumer-external-smoke-scaffold.json`
- Post-publish verification owner input template: `artifacts/final-release/post-publish-verification-owner-input.template.json`
- Post-publish verification record projection: `artifacts/final-release/post-publish-verification-record.json`
- Release issue close record candidate artifact: `artifacts/final-release/release-issue-close-record-candidate.json`
- Release issue close record owner input template: `artifacts/final-release/release-issue-close-record-owner-input.template.json`
- Final evidence freeze artifact: `artifacts/final-release/final-evidence-freeze.json`
- Final evidence freeze validation: `artifacts/final-release/final-evidence-freeze-validation.json`
- Release issue final close decision template: `artifacts/final-release/release-issue-final-close-decision.template.json`
- Release issue final close decision validation: `artifacts/final-release/release-issue-final-close-decision-validation.json`
- Real external proof overlay pack artifact: `artifacts/final-release/real-external-proof-overlay-pack.json`
- Real external proof overlay pack validation: `artifacts/final-release/real-external-proof-overlay-pack-validation.json`
- Release issue close record overlay candidate artifact: `artifacts/final-release/release-issue-close-record-overlay-candidate.json`
- Release issue close record overlay candidate validation: `artifacts/final-release/release-issue-close-record-overlay-candidate-validation.json`
- Owner external execution result backfill kit artifact: `artifacts/final-release/owner-external-execution-result-backfill-kit.json`
- Owner external execution result backfill kit validation: `artifacts/final-release/owner-external-execution-result-backfill-kit-validation.json`
- Owner input cross-hash audit artifact: `artifacts/final-release/owner-input-cross-hash-audit.json`
- Owner input cross-hash audit validation: `artifacts/final-release/owner-input-cross-hash-audit-validation.json`
- Release close strict record candidate artifact: `artifacts/final-release/release-close-strict-record-candidate.json`
- Release close strict record candidate validation: `artifacts/final-release/release-close-strict-record-candidate-validation.json`
- Owner proof real backfill execution pack artifact: `artifacts/final-release/owner-proof-real-backfill-execution-pack.json`
- Owner proof real backfill execution pack validation: `artifacts/final-release/owner-proof-real-backfill-execution-pack-validation.json`
- Release issue close record real input map artifact: `artifacts/final-release/release-issue-close-record-real-input-map.json`
- Release issue close record real input map validation: `artifacts/final-release/release-issue-close-record-real-input-map-validation.json`
- Owner real proof field delta pack artifact: `artifacts/final-release/owner-real-proof-field-delta-pack.json`
- Owner real proof field delta pack validation: `artifacts/final-release/owner-real-proof-field-delta-pack-validation.json`
- Real proof candidate promotion guard artifact: `artifacts/final-release/real-proof-candidate-promotion-guard.json`
- Real proof candidate promotion guard validation: `artifacts/final-release/real-proof-candidate-promotion-guard-validation.json`
- Real proof record validator artifact: `artifacts/final-release/real-proof-record-validator.json`
- Real proof record validator validation: `artifacts/final-release/real-proof-record-validator-validation.json`
- Owner real proof execution closure pack artifact: `artifacts/final-release/owner-real-proof-execution-closure-pack.json`
- Owner real proof execution closure pack validation: `artifacts/final-release/owner-real-proof-execution-closure-pack-validation.json`
- Runtime proof execution input record artifact: `artifacts/final-release/runtime-proof-execution-input-record.json`
- Runtime proof execution input record validation: `artifacts/final-release/runtime-proof-execution-input-record-validation.json`
- Final quality freeze dashboard artifact: `artifacts/final-release/final-quality-freeze-dashboard.json`
- Final quality freeze dashboard validation: `artifacts/final-release/final-quality-freeze-dashboard-validation.json`
- Public proof claim boundary audit artifact: `artifacts/final-release/public-proof-claim-boundary-audit.json`
- Article roadmap 30+ validation: `artifacts/final-release/article-roadmap-30plus-validation.json`
- Owner real input landing pack artifact: `artifacts/final-release/owner-real-input-landing-pack.json`
- Owner real input landing pack validation: `artifacts/final-release/owner-real-input-landing-pack-validation.json`
- Final Owner execution checklist artifact: `artifacts/final-release/final-owner-execution-checklist.json`
- Final Owner execution checklist validation: `artifacts/final-release/final-owner-execution-checklist-validation.json`
- Real proof import boundary audit artifact: `artifacts/final-release/real-proof-import-boundary-audit.json`
- Owner runtime proof execution runbook artifact: `artifacts/final-release/owner-runtime-proof-execution-runbook.json`
- Owner runtime proof execution runbook validation: `artifacts/final-release/owner-runtime-proof-execution-runbook-validation.json`
- Release close strict validation bridge artifact: `artifacts/final-release/release-close-strict-validation-bridge.json`
- Release close strict validation bridge validation: `artifacts/final-release/release-close-strict-validation-bridge-validation.json`
- Owner runtime proof result input template: `artifacts/final-release/owner-runtime-proof-result-input.template.json`
- Owner runtime proof result input validation: `artifacts/final-release/owner-runtime-proof-result-input-validation.json`
- Runtime proof lane dry-run summary: `artifacts/final-release/runtime-proof-lane-dry-run-summary.json`
- Runtime proof lane dry-run summary validation: `artifacts/final-release/runtime-proof-lane-dry-run-summary-validation.json`
- Release close strict dry-run summary: `artifacts/final-release/release-close-strict-dry-run-summary.json`
- Release close strict dry-run summary validation: `artifacts/final-release/release-close-strict-dry-run-summary-validation.json`
- Owner external proof execution bundle: `artifacts/final-release/owner-external-proof-execution-bundle.json`
- Owner external proof execution bundle validation: `artifacts/final-release/owner-external-proof-execution-bundle-validation.json`
- Owner external proof execution result import: `artifacts/final-release/owner-external-proof-execution-result-import.json`
- Owner external proof execution result import validation: `artifacts/final-release/owner-external-proof-execution-result-import-validation.json`
- Real external proof record import validator: `artifacts/final-release/real-external-proof-record-import-validator.json`
- Real external proof record import validator validation: `artifacts/final-release/real-external-proof-record-import-validator-validation.json`
- Release close owner input bridge: `artifacts/final-release/release-close-owner-input-bridge.json`
- Release close owner input bridge validation: `artifacts/final-release/release-close-owner-input-bridge-validation.json`
- Public package proof owner input template: `artifacts/final-release/public-package-proof-owner-input.template.json`
- Public package proof owner input validation: `artifacts/final-release/public-package-proof-owner-input-validation.json`
- Post-publish proof owner confirmation: `artifacts/final-release/post-publish-proof-owner-confirmation.json`
- Post-publish proof owner confirmation validation: `artifacts/final-release/post-publish-proof-owner-confirmation-validation.json`
- Release close public proof bridge: `artifacts/final-release/release-close-public-proof-bridge.json`
- Release close public proof bridge validation: `artifacts/final-release/release-close-public-proof-bridge-validation.json`
- Owner proof real input convergence artifact: `artifacts/final-release/owner-proof-real-input-convergence.json`
- Owner proof real input convergence validation: `artifacts/final-release/owner-proof-real-input-convergence-validation.json`
- Release close final owner runbook artifact: `artifacts/final-release/release-close-final-owner-runbook.json`
- Release close final owner runbook validation: `artifacts/final-release/release-close-final-owner-runbook-validation.json`
- Release issue close final owner decision audit: `artifacts/final-release/release-issue-close-final-owner-decision-audit.json`
- Release issue close final owner decision audit validation: `artifacts/final-release/release-issue-close-final-owner-decision-audit-validation.json`
- Final post-publish audit pack: `artifacts/final-release/final-post-publish-audit-pack.json`
- Final post-publish audit pack validation: `artifacts/final-release/final-post-publish-audit-pack-validation.json`
- Release docs and NuGet metadata audit: `artifacts/final-release/release-docs-and-nuget-metadata-audit.json`
- Release docs and NuGet metadata audit validation: `artifacts/final-release/release-docs-and-nuget-metadata-audit-validation.json`
- Post-publish user verification pack: `artifacts/final-release/post-publish-user-verification-pack.json`
- Post-publish user verification pack validation: `artifacts/final-release/post-publish-user-verification-pack-validation.json`
- Release candidate final freeze manifest: `artifacts/final-release/release-candidate-final-freeze-manifest.json`
- Release candidate final freeze manifest validation: `artifacts/final-release/release-candidate-final-freeze-manifest-validation.json`
- Public publish owner manual command handoff: `artifacts/final-release/public-publish-owner-manual-command-handoff.json`
- Public publish owner manual command handoff validation: `artifacts/final-release/public-publish-owner-manual-command-handoff-validation.json`
- Final release close blocker dashboard: `artifacts/final-release/final-release-close-blocker-dashboard.json`
- Final release close blocker dashboard validation: `artifacts/final-release/final-release-close-blocker-dashboard-validation.json`
- Public publish result owner input: `artifacts/final-release/public-publish-result-owner-input.template.json`
- Public publish result owner input validation: `artifacts/final-release/public-publish-result-owner-input-validation.json`
- Public publish result import: `artifacts/final-release/public-publish-result-import.json`
- Public publish result import validation: `artifacts/final-release/public-publish-result-import-validation.json`
- Post-publish clean consumer result convergence: `artifacts/final-release/post-publish-clean-consumer-result-convergence.json`
- Post-publish clean consumer result convergence validation: `artifacts/final-release/post-publish-clean-consumer-result-convergence-validation.json`
- StrictCloseReady convergence dashboard: `artifacts/final-release/strict-close-ready-convergence-dashboard.json`
- StrictCloseReady convergence dashboard validation: `artifacts/final-release/strict-close-ready-convergence-dashboard-validation.json`
- Public publish final owner execution pack: `artifacts/final-release/public-publish-final-owner-execution-pack.json`
- Public publish final owner execution pack validation: `artifacts/final-release/public-publish-final-owner-execution-pack-validation.json`
- Public publish command cross-check: `artifacts/final-release/public-publish-command-cross-check.json`
- Public publish command cross-check validation: `artifacts/final-release/public-publish-command-cross-check-validation.json`
- Release issue close owner decision input: `artifacts/final-release/release-issue-close-owner-decision-input.template.json`
- Release issue close owner decision input validation: `artifacts/final-release/release-issue-close-owner-decision-input-validation.json`
- Final evidence freeze non-proof audit: `artifacts/final-release/final-evidence-freeze-non-proof-audit.json`
- Final evidence freeze non-proof audit validation: `artifacts/final-release/final-evidence-freeze-non-proof-audit-validation.json`
- Public publish real result owner input contract: `artifacts/final-release/public-publish-real-result-owner-input-contract.json`
- Public publish real result owner input contract validation: `artifacts/final-release/public-publish-real-result-owner-input-contract-validation.json`
- Post-publish clean consumer proof record contract: `artifacts/final-release/post-publish-clean-consumer-proof-record-contract.json`
- Post-publish clean consumer proof record contract validation: `artifacts/final-release/post-publish-clean-consumer-proof-record-contract-validation.json`
- Release issue close strict owner decision import: `artifacts/final-release/release-issue-close-strict-owner-decision-import.json`
- Release issue close strict owner decision import validation: `artifacts/final-release/release-issue-close-strict-owner-decision-import-validation.json`
- Final close gate convergence: `artifacts/final-release/final-close-gate-convergence.json`
- Final close gate convergence validation: `artifacts/final-release/final-close-gate-convergence-validation.json`
- Public publish real result record draft: `artifacts/final-release/public-publish-real-result-record-draft.json`
- Public publish real result record draft validation: `artifacts/final-release/public-publish-real-result-record-draft-validation.json`
- Post-publish clean consumer proof record draft: `artifacts/final-release/post-publish-clean-consumer-proof-record-draft.json`
- Post-publish clean consumer proof record draft validation: `artifacts/final-release/post-publish-clean-consumer-proof-record-draft-validation.json`
- Public publish forbidden substitute scan: `artifacts/final-release/public-publish-forbidden-substitute-scan.json`
- Public publish forbidden substitute scan validation: `artifacts/final-release/public-publish-forbidden-substitute-scan-validation.json`
- Release close real proof import bridge: `artifacts/final-release/release-close-real-proof-import-bridge.json`
- Release close real proof import bridge validation: `artifacts/final-release/release-close-real-proof-import-bridge-validation.json`
- Final owner close readiness checkpoint: `artifacts/final-release/final-owner-close-readiness-checkpoint.json`
- Final owner close readiness checkpoint validation: `artifacts/final-release/final-owner-close-readiness-checkpoint-validation.json`
- Final release close record real validator: `artifacts/final-release/final-release-close-record-real-validator.json`
- Final release close record real validator validation: `artifacts/final-release/final-release-close-record-real-validator-validation.json`
- Final owner release close record projection: `artifacts/final-release/final-owner-release-close-record-projection.json`
- Final owner release close record projection validation: `artifacts/final-release/final-owner-release-close-record-projection-validation.json`
- Final release close hash consistency gate: `artifacts/final-release/final-release-close-hash-consistency-gate.json`
- Final release close hash consistency gate validation: `artifacts/final-release/final-release-close-hash-consistency-gate-validation.json`
- Final close owner approval boundary audit: `artifacts/final-release/final-close-owner-approval-boundary-audit.json`
- Final close owner approval boundary audit validation: `artifacts/final-release/final-close-owner-approval-boundary-audit-validation.json`
- Release candidate final publishability audit: `artifacts/final-release/release-candidate-final-publishability-audit.json`
- Release candidate final publishability audit validation: `artifacts/final-release/release-candidate-final-publishability-audit-validation.json`
- Release candidate owner action roadmap: `artifacts/final-release/release-candidate-owner-action-roadmap.json`
- Release candidate owner action roadmap validation: `artifacts/final-release/release-candidate-owner-action-roadmap-validation.json`
- Release candidate non-substitute final scan: `artifacts/final-release/release-candidate-non-substitute-final-scan.json`
- Release candidate non-substitute final scan validation: `artifacts/final-release/release-candidate-non-substitute-final-scan-validation.json`
- Release candidate final owner checklist: `artifacts/final-release/release-candidate-final-owner-checklist.json`
- Release candidate final owner checklist validation: `artifacts/final-release/release-candidate-final-owner-checklist-validation.json`
- Release proof readiness snapshot artifact: `artifacts/final-release/release-proof-readiness-snapshot.json`
- Release candidate freeze summary artifact: `artifacts/release/release-candidate-freeze-summary.json`
- Final audit map: `docs/articles/zh-cn/release-final-audit-map.md`
- Public story pack: `docs/articles/zh-cn/release-public-story-pack.md`
- Owner proof backlog: `docs/articles/zh-cn/release-owner-proof-backlog.md`
- Non-substitute proof list: `docs/articles/zh-cn/release-proof-non-substitutes.md`
- Article index and publishing order: `docs/articles/zh-cn/release-article-index-and-publishing-order.md`
- Technical article closure ledger: `docs/articles/zh-cn/publishing/technical-article-closure-ledger.md`
- Technical article foundations first-batch audit (articles 2-6, 10, 15-16, 19): `docs/articles/zh-cn/publishing/technical-article-foundations-first-batch-audit.md`
- Technical article foundations second-batch audit (articles 28-32, 38-45, 103): `docs/articles/zh-cn/publishing/technical-article-foundations-second-batch-audit.md`
- Technical article proof backlog (42 owner/runtime proof rows): `docs/articles/zh-cn/publishing/technical-article-proof-backlog.md`
- External model evidence case study: `docs/articles/zh-cn/external-model-evidence-case-study.md`
- Project capability and release-boundary story: `docs/articles/zh-cn/project-release-story-and-boundaries.md`
- README frontpage checklist: `docs/articles/zh-cn/release-readme-frontpage-checklist.md`
- Owner final action sequence: `docs/articles/zh-cn/release-final-owner-action-sequence.md`
- Release issue close record validator: `eng/Test-ReleaseIssueCloseRecord.ps1`
- Release issue close record validation artifact: `artifacts/final-release/release-issue-close-record-validation.json`
- Frontpage and proof boundary final audit: `docs/articles/zh-cn/release-frontpage-and-proof-boundary-final-audit.md`
- Final evidence freeze artifact: `artifacts/final-release/release-candidate-final-evidence-freeze.json`
- Stale release claims audit artifact: `artifacts/final-release/stale-release-claims-audit.json`
- Release candidate final cross-check: `docs/articles/zh-cn/release-candidate-final-cross-check.md`
- Release candidate article matrix summary: `docs/articles/zh-cn/release-candidate-article-matrix-summary.md`
- Release candidate publication summary: `docs/articles/zh-cn/release-candidate-publication-summary.md`
- Release candidate final hold and owner waiting state: `docs/articles/zh-cn/release-candidate-final-hold-owner-waiting.md`
- Owner action checklist for final hold: `docs/articles/zh-cn/release-owner-action-checklist-final-hold.md`
- Release hold final inspection: `docs/articles/zh-cn/release-hold-final-inspection.md`

The shortest owner action surface is the `owner-release-execution-package` `oneScreenReleaseHoldChecklist`. Its validator now checks the owner execution package shape while preserving `performsPublish=false`, `canPublishPublicly=false`, and `canCloseReleaseIssue=false`. It mirrors the five remaining blockers, but it is guidance only: `owner authorization`, `package-consumer-runtime`, Linux runner proof, `real-model-runtime`, and `post-publish verification` still require real records and validators. After those pass, owner must fill the final release issue close record and pass `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`; until then `canCloseReleaseIssue=false` remains the release boundary.

The `final-evidence-freeze` records SHA256 values for the current evidence bundle, post-publish validation, release-close candidate validation, owner execution package validation, and final close decision validation. It is a frozen audit snapshot, not a proof promotion surface. The `release-issue-final-close-decision` template is the owner-facing last input contract for rollback review, public package source confirmation, clean consumer confirmation, runtime smoke exit code, and log/hash review; its template validation remains `blocked-owner-final-close-decision-required`.

The `real-external-proof-overlay-pack` and `release-issue-close-record-overlay-candidate` now group the remaining real owner-input fields and close-record hash mappings. Both are blocked/non-proof surfaces: they keep `performsPublish=false`, `canPublishPublicly=false`, and `canCloseReleaseIssue=false` until real post-publish proof, clean external consumer smoke, rollback approval, final owner decision, and strict close validation pass.

The `owner-external-execution-result-backfill-kit` and `owner-input-cross-hash-audit` extend that chain into owner execution result backfill and local cross-hash consistency. The kit cannot collect proof by itself, publish packages, approve publication, or close the release issue. The audit can show local artifact/path/hash consistency, but matching hashes cannot substitute real external proof, post-publish verification, owner approval, or strict release-close validation.

The `release-close-strict-record-candidate` adds a stricter final candidate surface for the close record. It binds the local evidence bundle, final freeze, post-publish validation, close candidate validation, final close decision, overlay validation, owner backfill kit, and cross-hash audit, but it remains blocked/non-proof until real owner approval, post-publish proof, clean consumer runtime proof, rollback approval, and `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady` all pass.

The `owner-proof-real-backfill-execution-pack` converts that strict candidate into owner input tasks, real proof tasks, and hash check tasks. It is owner handoff only: it keeps `performsPublish=false`, `canPublishPublicly=false`, and `canCloseReleaseIssue=false`.

The `release-issue-close-record-real-input-map` maps those owner input tasks to final close record fields and target artifacts. It remains blocked/non-proof until real owner inputs and strict close validation pass.

The `owner-real-proof-field-delta-pack` converts blocked strict candidate field contracts into concrete Owner field deltas, and `real-proof-candidate-promotion-guard` blocks candidate promotion until those deltas, non-substitute checks, and later real proof validators are satisfied. Both are blocked/non-proof surfaces and keep `performsPublish=false`, `canPublishPublicly=false`, and `canCloseReleaseIssue=false`.

The `real-proof-record-validator` defines the strict validator contract for future real proof records, and `owner-real-proof-execution-closure-pack` turns those contracts into Owner execution closure items with first commands, expected artifacts, logs, SHA256 fields, validator commands, and release-close follow-up. Both remain blocked/non-proof and cannot publish, promote runtime proof, verify post-publish state, or close the release issue.

The `runtime-proof-execution-input-record`, `owner-runtime-proof-execution-runbook`, and `release-close-strict-validation-bridge` continue that chain into owner-filled execution inputs, per-lane command sequences, and strict close prerequisite aggregation. All three remain blocked/non-proof: placeholder fields, runbook commands, hash slots, bridge readiness flags, and local evidence aggregation cannot replace real runtime logs, post-publish proof, rollback approval, final close decision, or `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`.

The `owner-runtime-proof-result-input`, `runtime-proof-lane-dry-run-summary`, and `release-close-strict-dry-run-summary` add the next owner result backfill and strict dry-run layer. The `owner-external-proof-execution-bundle`, `owner-external-proof-execution-result-import`, `real-external-proof-record-import-validator`, and `release-close-owner-input-bridge` continue that chain into owner-executed proof commands, result import slots, strict real-proof import contracts, and release-close owner gate aggregation. They expose missing real files, hashes, host/package metadata, validator output, reviewer fields, lane blockers, close blockers, and owner gates while keeping `canPromoteRuntimeProof=false`, `canPublishPublicly=false`, and `canCloseReleaseIssue=false`.

The `owner-proof-real-input-convergence` turns the remaining owner inputs, validators, and proof blockers into a single owner-facing validation matrix. It is still non-proof and cannot close the release issue.

The `public-package-proof-owner-input`, `post-publish-proof-owner-confirmation`, and `release-close-public-proof-bridge` add the public package proof preparation layer. They keep `performsPublish=false`, `canPublishPublicly=false`, and `canCloseReleaseIssue=false` while exposing the missing NuGet package source, GitHub Release asset path/hash, public package URL/hash, repository-external clean consumer restore/build/smoke logs, stdout/stderr SHA256, host metadata, owner review, post-publish proof gates, and final public proof bridge gates.

The `release-close-final-owner-runbook` turns that convergence matrix into the final Owner execution manual: public package source confirmation, package hashes, clean external consumer runtime smoke, smoke log/hash/host metadata, rollback review, final close decision, strict candidate refresh, and `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`. It remains blocked/non-proof and keeps `performsPublish=false`, `canPublishPublicly=false`, and `canCloseReleaseIssue=false`.

The `release-issue-close-final-owner-decision-audit` and `final-post-publish-audit-pack` add the final blocked audit layer for owner close decision and post-publish verification. They aggregate gates only; they are not runtime proof, post-publish proof, publish approval, release close approval, or package push.

The `release-candidate-final-freeze-manifest`, `public-publish-owner-manual-command-handoff`, and `final-release-close-blocker-dashboard` provide the final owner handoff layer. They freeze local artifact hashes, list manual publish placeholders, and summarize remaining close blockers, while keeping `performsPublish=false`, `notExecutedByAutomation=true` where applicable, and `canCloseReleaseIssue=false`.

`public-publish-result-owner-input`, `public-publish-result-import`, `post-publish-clean-consumer-result-convergence`, and `strict-close-ready-convergence-dashboard` keep Owner real public publish result backfill, clean consumer proof gaps, and StrictCloseReady close conditions on the same blocked/non-proof evidence chain. They do not publish, upload packages, approve public release, or close the release issue.

`public-publish-final-owner-execution-pack`, `public-publish-command-cross-check`, `release-issue-close-owner-decision-input`, and `final-evidence-freeze-non-proof-audit` add the final manual execution and boundary audit layer. They keep the release gate blocked until real owner public publish results, post-publish clean consumer proof, rollback review, and strict close validation are supplied. They are not runtime proof, post-publish proof, publish approval, release close approval, or package push.

`public-publish-real-result-owner-input-contract`, `post-publish-clean-consumer-proof-record-contract`, `release-issue-close-strict-owner-decision-import`, and `final-close-gate-convergence` add the post-real-publish owner backfill contract layer. They require real public package source, downloaded package hashes, repository-external clean consumer smoke evidence, rollback review, final owner decision, and strict close validation before any release close promotion. They are not runtime proof, post-publish proof, publish approval, release close approval, or package push.

`public-publish-real-result-record-draft`, `post-publish-clean-consumer-proof-record-draft`, `public-publish-forbidden-substitute-scan`, `release-close-real-proof-import-bridge`, and `final-owner-close-readiness-checkpoint` add the execution-facing real proof backfill layer. They expose owner-fill draft fields, forbidden substitute blockers, proof import lanes, and final readiness checks while staying blocked/non-proof. They are not runtime proof, post-publish proof, publish approval, release close approval, or package push.

`final-owner-strict-close-execution-order` archives the final Owner execution sequence across 7 action worklist entries, 7 execution steps, clean/post-publish runbooks, 10 public publish lanes, 11 command checks, 12 readiness checks, 19 blockers, and Owner input contract convergence. It is blocked/non-proof guidance only, not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push.

`real-owner-evidence-strict-validator-orchestration` connects the real Owner input contract to strict validators across 11 source records and 16 readiness fields, including public package URL/hash, clean consumer logs, post-publish logs, stdout/stderr, host metadata, non-substitute confirmations, rollback review, and final close decision. It remains blocked/non-proof, rejects local feed, ProjectReference, direct nupkg, dry-run, dashboard, runbook, candidate, and build-only substitutes, and does not publish or promote proof.

`release-close-real-input-candidate-promotion-readiness` adds the ReleaseClose real input candidate promotion layer. It maps 11 blocked promotion lanes from public package proof through final publish proof gate, keeps every lane non-proof, and records which strict validators must accept real Owner inputs before any candidate can promote. It is not runtime proof, post-publish proof, publish approval, release close approval, or package push; `failedBlockerCount=0` is not ready.

`final-release-close-record-real-validator`, `final-owner-release-close-record-projection`, `final-release-close-hash-consistency-gate`, and `final-close-owner-approval-boundary-audit` add the final ReleaseCloseRecord real-validation layer. They project required owner fields, close-record lanes, current hash consistency, and owner approval boundaries while remaining blocked/non-proof. They are not runtime proof, post-publish proof, publish approval, release close approval, or package push.

`release-candidate-final-publishability-audit`, `release-candidate-owner-action-roadmap`, `release-candidate-non-substitute-final-scan`, and `release-candidate-final-owner-checklist` add the final publishability review layer. They consolidate publishability gates, owner execution order, forbidden substitute checks, and the one-page owner checklist while remaining blocked/non-proof. They are not runtime proof, post-publish proof, publish approval, release close approval, or package push.

The `owner-proof-backfill-execution-pack` is the focused command-and-input companion for that owner action surface. It enumerates the real fields, first commands, validators, expected artifacts, and non-substitute proof kinds for `owner-authorization`, `package-consumer-runtime`, `linux-runner-proof`, `real-model-runtime`, `post-publish-verification`, and `release-issue-close-record`, but it is still guidance only.

The `owner-proof-execution-handoff` turns the backfill pack into an owner-facing execution dashboard: each proof line carries current state, owner next action, candidate artifacts, missing real inputs, and validator commands. It remains guidance only and cannot publish or close the release issue.

The `owner-external-proof-input-preflight` audits candidate owner inputs before any release-close review: it classifies each proof line as missing, template-only, guidance-only, candidate-needs-owner-review, or validator-passed-real-proof while keeping `canPublishPublicly=false` and `canCloseReleaseIssue=false`.

The `owner-proof-input-repair-pack` turns those blocked preflight lines into a field-level repair checklist: placeholders, existing files, SHA256 values, clean consumer evidence, owner decisions, rollback plans, first repair commands, and validators. Repair packs and input drafts are still not proof.

The `owner-proof-input-draft-pack` is the non-proof drafting surface for those repair items: it lists per-line draft paths, strict validators, and proof-substitute blockers while keeping every draft non-promotable.

The `owner-external-proof-backfill-orchestrator` converts draft specs into owner command plans for real external proof records. It remains guidance only and cannot collect proof, publish, or close the release issue by itself.

The `package-consumer-runtime-proof-owner-input` and `release-issue-close-record-owner-input` templates define the exact owner-filled fields needed by the next candidate overlay pass. `package-consumer-runtime-proof-record` projects those fields into a strict runtime proof record and can bridge into `external-runtime-proof-record`; `package-consumer-external-smoke-scaffold` creates a clean consumer project shape outside the repository. Both remain blocked until real clean external consumer smoke evidence passes validation. The candidate and scaffold surfaces remain blocked/non-proof; none can publish, close the release issue, or substitute for validator-passing real proof.

The `release-proof-readiness-snapshot` is the compact five-blocker status view for the same proof chain. It is also guidance only: it does not publish packages, upload assets, or close the release issue.

`applications/TensorRtExec` can generate ONNX build/precheck reports, but build-only, parse-only, sidecar-only, local feed, ProjectReference, collection bundles, runbooks, and `blocked-by-cuda-driver` are not release proof records. `samples/YoloVision` is the unified YOLO-family sample for YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom and det/cls/seg/obb/pose/sem. The repository now carries audited source-tree real-model-runtime records for official YOLOv8n det/cls/seg/pose/obb and torchvision LRASPP sem. Other family/task/exporter combinations still require their own assets, logs, hashes, and validators, and no source-tree record is package-consumer, public-package, post-publish, or release proof. The long-form entrypoints are `docs/articles/zh-cn/yolovision-all-task-overview.md` and `docs/articles/zh-cn/yolovision-detection-tutorial.md`.

YOLOv8n-seg now has both source-tree runtime evidence and a repository-external bridge-only local package consumer. The latter references only the managed API, YoloVision, and one `.Bridge` package; compares 1,793,600 raw values, four source-image masks, and independent PyTorch IoU; and requires raw-reference and mask mutations to fail closed. Run `eng/Test-YoloVisionSegmentationLocalPackageConsumer.ps1` and inspect `samples/assets/yolovision-yolov8n-seg-local-package-consumer-runtime-evidence.json`. This remains false for public-package, post-publish, redistribution, Owner acceptance, and release proof.

Torchvision LRASPP semantic segmentation now has the same repository-external three-package path. `eng/Test-YoloVisionSemanticLocalPackageConsumer.ps1` compares all 2,150,400 logits and the 102,400-pixel int32 class-index map, then requires raw-reference and class-index mutations to fail closed. The ONNX remains under `<workspace-root>/models` for a future Model Zoo and is never packed or uploaded. See `samples/assets/yolovision-lraspp-semantic-local-package-consumer-runtime-evidence.json` and `docs/articles/zh-cn/yolovision-lraspp-semantic-local-package-consumer-tutorial.md`.

Official YOLOv8n classification now also runs through the repository-external three-package path. `eng/Test-YoloVisionClassificationLocalPackageConsumer.ps1` compares all 1,000 probabilities, fixes the independent Top-5 order, and checks a fail-closed reference mutation. The shared runner isolates exactly one selected nupkg per feed and verifies the restored package cache hashes, so duplicate local package IDs cannot silently invalidate evidence. See `samples/assets/yolovision-yolov8n-cls-local-package-consumer-runtime-evidence.json` and `docs/articles/zh-cn/yolovision-yolov8n-cls-local-package-consumer-tutorial.md`.

Official YOLOv8n Pose uses the same isolated three-package path. `eng/Test-YoloVisionPoseLocalPackageConsumer.ps1` requires a byte-identical C# letterbox tensor, compares all 470,400 raw values, validates four 17-keypoint poses against an independent Ultralytics/PyTorch reference, and checks a fail-closed reference mutation. The converted ONNX remains under the outer `models` directory for the future Model Zoo and is not uploaded. See `samples/assets/yolovision-yolov8n-pose-local-package-consumer-runtime-evidence.json` and `docs/articles/zh-cn/yolovision-yolov8n-pose-local-package-consumer-tutorial.md`.

Official YOLOv8n OBB now follows that isolated three-package path as well. `eng/Test-YoloVisionObbLocalPackageConsumer.ps1` requires a byte-identical 1024x1024 C# letterbox tensor, compares all 430,080 raw values, validates 40 ship oriented boxes against an independent Ultralytics/PyTorch rotated-IoU reference, and checks a fail-closed reference mutation. The converted ONNX remains under the outer `models` directory for the future Model Zoo and is not uploaded. See `samples/assets/yolovision-yolov8n-obb-local-package-consumer-runtime-evidence.json` and `docs/articles/zh-cn/yolovision-yolov8n-obb-local-package-consumer-tutorial.md`.

Official YOLOv8n detection now completes the same isolated three-package path. `eng/Test-YoloVisionDetectionLocalPackageConsumer.ps1` requires a byte-identical 640x640 C# letterbox tensor, compares all 705,600 raw values, validates four person detections and one bus against an independent Ultralytics/PyTorch box-IoU reference, and checks a fail-closed reference mutation. The converted ONNX remains under the outer `models` directory for the future Model Zoo and is not uploaded. See `samples/assets/yolovision-yolov8n-det-local-package-consumer-runtime-evidence.json` and `docs/articles/zh-cn/yolovision-yolov8n-det-local-package-consumer-tutorial.md`.

The GPU allocator now also has a repository-external two-package consumer. `eng/Test-GpuAllocatorLocalPackageConsumer.ps1` restores only the managed API and matching bridge-only package, removes development bridge probing, verifies the restored bridge hash, runs eight real TensorRT 10.11 builder callbacks, and requires zero final allocations plus fail-closed rejection and exception cases. The identity network is created in code, so model acquisition and ONNX conversion are explicitly not applicable. See `samples/GpuAllocator.PackageConsumer`, `samples/assets/gpu-allocator-local-package-consumer-tensorrt10.11-evidence.json`, and `docs/articles/zh-cn/gpu-allocator-local-package-consumer-tutorial.md`. This is local-package evidence only and does not publish a package, tag, or Release.

OutputAllocator uses the same shared two-package harness without duplicating its restore and package-audit implementation. `eng/Test-OutputAllocatorLocalPackageConsumer.ps1` runs a repository-external identity-network consumer, observes real `reallocateOutput` and `notifyShape` callbacks, pairs one allocation with one release, requires zero live allocations after detach, and proves that a rejected reallocation fails enqueue without allocating. See `samples/OutputAllocator.PackageConsumer`, `samples/assets/output-allocator-local-package-consumer-tensorrt10.11-evidence.json`, and `docs/articles/zh-cn/output-allocator-local-package-consumer-tutorial.md`. CUDA and TensorRT remain host-installed, and no package, tag, or Release is published.

DebugListener now has the same repository-external two-package proof. `eng/Test-DebugListenerLocalPackageConsumer.ps1` restores only the managed API and matching bridge-only package, installs a native listener on a programmatically created debug tensor, observes copied `[1,4]` metadata without exposing the borrowed pointer, and requires clean detach after both an accepted callback and a controlled rejection. See `samples/DebugListener.PackageConsumer`, `samples/assets/debug-listener-local-package-consumer-tensorrt10.11-evidence.json`, and `docs/articles/zh-cn/debug-listener-local-package-consumer-tutorial.md`. The identity graph needs no external model or conversion; this local proof does not publish a package, tag, or Release.

ProgressMonitor now closes the remaining owner-safe callback package path. `eng/Test-ProgressMonitorLocalPackageConsumer.ps1` builds a programmatic 1x1 convolution network from a repository-external two-package consumer, observes real `phaseStart`, `stepComplete`, and `phaseFinish` callbacks through thread-safe managed state, and requires a controlled `stepComplete=false` cancellation to stop the build without being misclassified as a managed callback exception. See `samples/ProgressMonitor.PackageConsumer`, `samples/assets/progress-monitor-local-package-consumer-tensorrt10.11-evidence.json`, and `docs/articles/zh-cn/progress-monitor-local-package-consumer-tutorial.md`. No model download or conversion applies, vendor runtimes remain host-installed, and nothing is published.

Profiler now has real inference proof through the same repository-external two-package boundary. `eng/Test-ProfilerLocalPackageConsumer.ps1` validates immediate layer timing with `EnqueueEmitsProfile=true`, deferred timing with `false` plus `ReportToProfiler()`, copied finite metadata, clean detach, and controlled managed handler exceptions. See `samples/Profiler.PackageConsumer`, `samples/assets/profiler-local-package-consumer-tensorrt10.11-evidence.json`, and `docs/articles/zh-cn/profiler-local-package-consumer-tutorial.md`. The programmatic convolution graph requires no model conversion, and this local proof performs no publication.

Logger now has native TensorRT message proof through a repository-external two-package consumer. `eng/Test-LoggerLocalPackageConsumer.ps1` builds, deserializes, and executes a programmatic convolution network without calling `EmitDiagnostic`; it verifies copied severity/message metadata, builder/runtime borrowing, deferred disposal, managed handler-exception isolation, and atomic native failure flags across TensorRT 8/10/11. See `samples/Logger.PackageConsumer`, `samples/assets/logger-local-package-consumer-tensorrt10.11-evidence.json`, and `docs/articles/zh-cn/logger-local-package-consumer-tutorial.md`. CUDA and TensorRT remain host-installed, and nothing is published.

The large `eng` directory is an engineering pipeline surface, not a flat end-user command list. Start from `eng/README.md` for the supported entrypoints and internal-script retention rules. The Chinese DocFX directory likewise contains user documentation, internal engineering records, proof templates, and publication drafts; `docs/articles/zh-cn/README.md` and `publication-catalog.json` define the distinction. A technical article is not content-complete until it includes real execution results and a source-attributed result image and passes `eng/Test-TechnicalArticleCompleteness.ps1 -Strict`.

The publication article matrix now includes direct user paths for `YoloVision`, `OnnxToEngine`, and `TensorRtExec`: start with `docs/articles/zh-cn/yolovision-sample-overview.md`, `docs/articles/zh-cn/onnx-to-engine-quickstart.md`, and `docs/articles/zh-cn/tensorrtexec-cli-parameter-map.md`. The first campaign-body batch is now available at `docs/articles/zh-cn/tensorrtsharp-4-project-overview-campaign.md`, `docs/articles/zh-cn/tensorrtsharp-4-architecture-abi-wrapper.md`, `docs/articles/zh-cn/cuda-tensorrt-cudnn-version-matrix-guide.md`, `docs/articles/zh-cn/tensorrtsharp-nuget-runtime-package-guide.md`, `docs/articles/zh-cn/windows-installation-and-troubleshooting-guide.md`, `docs/articles/zh-cn/linux-installation-runner-boundary-guide.md`, `docs/articles/zh-cn/tensorrtexec-gui-cli-parity-design.md`, `docs/articles/zh-cn/plugin-registry-inventory-readonly-design.md`, `docs/articles/zh-cn/package-consumer-runtime-proof-clean-consumer-guide.md`, `docs/articles/zh-cn/release-proof-and-post-publish-verification-guide.md`, `docs/articles/zh-cn/tensorrtsharp-4-faq.md`, and `docs/articles/zh-cn/tensorrtsharp-4-release-story.md`. The second campaign-body batch now expands the user path through `docs/articles/zh-cn/cuda-memory-wrapper.md`, `docs/articles/zh-cn/cuda-stream-event-multistream-tutorial.md`, `docs/articles/zh-cn/dynamic-shape-optimization-profile-tutorial.md`, `docs/articles/zh-cn/onnx-parser-to-serialized-engine-tutorial.md`, `docs/articles/zh-cn/network-layer-coverage-guide.md`, `docs/articles/zh-cn/blog-refit-weights-guide.md`, `docs/articles/zh-cn/tensorrtexec-cli-parameter-map.md`, `docs/articles/zh-cn/yolovision-sample-overview.md`, `docs/articles/zh-cn/yolo-vision-model-matrix.md`, `docs/articles/zh-cn/local-nuget-feed-consumer.md`, `docs/articles/zh-cn/linux-runner-evidence-checklist.md`, and `docs/articles/zh-cn/runtime-packages.md`. These guides are documentation and adoption aids; they are not public package proof, post-publish proof, package push, or release close approval.

The third campaign-body batch adds concrete real-case and release-story topics: `docs/articles/zh-cn/yolovision-detection-real-model-tutorial.md`, `docs/articles/zh-cn/yolovision-classification-real-model-tutorial.md`, `docs/articles/zh-cn/yolovision-segmentation-real-model-tutorial.md`, `docs/articles/zh-cn/tensorrtexec-gui-user-guide.md`, `docs/articles/zh-cn/tensorrtexec-trtexec-parity-deep-dive.md`, `docs/articles/zh-cn/onnx-to-engine-trtexec-conversion-guide.md`, `docs/articles/zh-cn/runtime-package-installation-deep-dive.md`, `docs/articles/zh-cn/cuda-error-35-troubleshooting.md`, `docs/articles/zh-cn/deferred-api-real-completion-review.md`, `docs/articles/zh-cn/plugin-inventory-readonly-api.md`, and `docs/articles/zh-cn/package-consumer-runtime-proof-playbook.md`. The obsolete Pose/OBB/Semantic roadmap was removed after all three tasks gained real-model tutorials and runtime evidence. These are still documentation and adoption aids, not runtime proof, public package proof, post-publish proof, package push, or release close approval.

The fourth campaign-body batch deepens the executable adoption story and release-boundary explanation: `docs/articles/zh-cn/yolovision-detection-yolov8n-download-export-run.md`, `docs/articles/zh-cn/yolovision-segmentation-mask-postprocess-guide.md`, `docs/articles/zh-cn/yolovision-pose-keypoint-output-guide.md`, `docs/articles/zh-cn/yolovision-obb-angle-output-guide.md`, `docs/articles/zh-cn/runtime-package-windows-linux-install-faq.md`, `docs/articles/zh-cn/plugin-registry-inventory-user-guide.md`, `docs/articles/zh-cn/deferred-readonly-api-upgrade-playbook.md`, `docs/articles/zh-cn/csharp-wrapper-lifetime-design.md`, `docs/articles/zh-cn/release-evidence-non-substitute-guide.md`, and `docs/articles/zh-cn/project-roadmap-to-public-release.md`. The former screenshot-only TensorRtExec draft was consolidated into `docs/articles/zh-cn/tensorrtexec-gui-user-guide.md`; the duplicate report-schema draft was replaced by the authoritative application schema and report tutorial. These guides remain documentation and adoption aids; they are not runtime proof, public package proof, post-publish proof, package push, or release close approval.

The fifth campaign-body batch retains validator and owner-checklist guidance for the remaining release blockers: `docs/articles/zh-cn/yolovision-owner-asset-evidence-guide.md`, `docs/articles/zh-cn/tensorrtexec-report-proof-boundary.md`, `docs/articles/zh-cn/onnx-to-engine-trtexec-proof-boundary.md`, `docs/articles/zh-cn/callback-allocator-listener-readonly-safety-gates.md`, `docs/articles/zh-cn/runtime-package-minimal-smoke-commands.md`, `docs/articles/zh-cn/runtime-package-native-load-troubleshooting.md`, `docs/articles/zh-cn/plugin-inventory-field-metadata-smoke-guide.md`, `docs/articles/zh-cn/deferred-next-readonly-candidate-list.md`, `docs/articles/zh-cn/csharp-public-api-handle-exposure-audit.md`, `docs/articles/zh-cn/release-proof-strict-validator-playbook.md`, `docs/articles/zh-cn/release-proof-owner-input-dashboard.md`, `docs/articles/zh-cn/release-proof-sample-article-closure.md`, and `docs/articles/zh-cn/public-release-owner-final-checklist.md`. Obsolete YoloVision schema/golden/template drafts were removed after the machine schemas, task contract, strict references, and complete model tutorials became authoritative. The authoritative TensorRtExec 84-option map and report schema remain under `applications/TensorRtExec`. These guides remain documentation, validator guidance, and adoption aids; they are not runtime proof, public package proof, post-publish proof, package push, or release close approval.

Release-readiness matrices are now machine-readable: `samples/YoloVision/yolo-model-matrix.json`, `samples/OnnxToEngine/trtexec-parity-matrix.json`, `applications/TensorRtExec/tensor-rt-exec-feature-matrix.json`, `artifacts/cuda-runtime-compilation/capability-matrix.json`, `artifacts/cuda-runtime-compilation/local-smoke.json`, `artifacts/interface-coverage/release-api-readiness-audit.json`, `artifacts/final-release/release-proof-owner-input-dashboard.json`, `artifacts/final-release/owner-proof-execution-checklist.json`, `artifacts/final-release/post-publish-clean-consumer-owner-input.template.json`, `artifacts/final-release/yolovision-owner-asset-evidence.template.json`, `artifacts/final-release/yolovision-owner-asset-evidence.example.json`, and `docs/articles/zh-cn/publishing/article-roadmap-30plus.json`.

Owner execution guides now include `docs/articles/zh-cn/owner-proof-execution-checklist.md`, `docs/articles/zh-cn/post-publish-clean-consumer-owner-input-guide.md`, and `docs/articles/zh-cn/yolovision-owner-asset-evidence-example.md`. These remain owner-input guidance and examples only; they are not runtime proof, public package proof, post-publish proof, package push, or release close approval.

Strict close preparation now adds `docs/articles/zh-cn/owner-real-proof-field-delta-dashboard.md`, `docs/articles/zh-cn/release-close-strict-gate-dashboard.md`, and `docs/articles/zh-cn/owner-proof-import-preflight.md`, backed by `artifacts/final-release/owner-real-proof-field-delta-dashboard.json`, `artifacts/final-release/release-close-strict-gate-dashboard.json`, and `artifacts/final-release/owner-proof-import-preflight.json`. These dashboards are blocked guidance only until owner-filled real proof passes strict validators.

ReleaseCandidate final public-material review now adds `docs/articles/zh-cn/release-candidate-freeze-manifest.md`, `docs/articles/zh-cn/public-material-final-scan.md`, and `docs/articles/zh-cn/final-owner-proof-blocker-dashboard.md`, backed by `artifacts/final-release/release-candidate-freeze-manifest.json`, `artifacts/final-release/public-material-final-scan.json`, and `artifacts/final-release/final-owner-proof-blocker-dashboard.json`. These records freeze public materials, scan stale sample/proof wording, and consolidate owner blockers while remaining non-proof: they do not publish packages, do not create runtime proof, do not create post-publish proof, and cannot close the release issue.

Owner real proof import audit now adds `docs/articles/zh-cn/owner-real-proof-import-audit-bundle.md`, `docs/articles/zh-cn/owner-evidence-file-manifest-template.md`, `docs/articles/zh-cn/strict-validator-command-runbook.md`, and `docs/articles/zh-cn/release-issue-close-owner-input-final-checklist.md`, backed by `artifacts/final-release/owner-real-proof-import-audit-bundle.json`, `artifacts/final-release/owner-evidence-file-manifest.template.json`, `artifacts/final-release/strict-validator-command-runbook.json`, and `artifacts/final-release/release-issue-close-owner-input-final-checklist.json`. These files organize owner paths, hashes, validator commands, and close inputs while keeping `canPublishPublicly=false` and `canCloseReleaseIssue=false`.

## Current Verified Status

Status as of 2026-06-12:

- TensorRT interface coverage matrix: `0` missing rows.
- CUDA runtime interface coverage matrix: `0` missing rows.
- Manifest API inventory: `3271` API records across `102` manifests.
- Latest coverage report: `artifacts/interface-coverage/interface-coverage-summary.md`.
- Native high-version validation: `win-x64-trt11-cuda13-release` configures and builds.
- Managed validation: solution build and project quality tests pass.
- DocFX validation: documentation builds with `0` warnings and `0` errors.
- Current release-proof boundary: package-consumer runtime proof still requires a compatible CUDA host record; `blocked-by-cuda-driver`, runbooks, collection bundles, dependency probes, local feed, ProjectReference, bridge-only logs, `Skipped=True`, mismatched log SHA256, sidecar-only reports, and build-only reports are not smoke passed evidence.

The project now has manifest/native-source coverage for the scanned TensorRT 8/10/11 and CUDA 11/12/13 headers in this workspace. A subset of unsafe or rarely used CUDA runtime APIs is intentionally recorded as deferred boundaries instead of being exposed as high-level managed APIs. Owner-safe copied or bridge-owned paths now cover CUDA library metadata, texture/surface descriptors, primary execution contexts, IPC event/memory export tokens, and pointer-free graph memory allocation/free flows. Callback lifetimes, raw driver entrypoints, external resource imports, IPC open/close ownership, borrowed device pointers, and user-object destructor ownership remain deferred.

## Current Stage

The raw interface coverage phase is complete for the locally scanned headers. Active work is release hardening:

- keep coverage matrix, build, tests, and DocFX green
- keep sample runners accurate and reproducible
- keep asset-dependent sample directories documented instead of empty
- validate runtime-package collection and package-consumer restore/build/run paths
- promote high-value deferred boundaries only when ABI, ownership, version guards, and managed lifetimes are clear

## Deployment Validation Path

For a local Windows deployment sanity check, use this order:

```powershell
dotnet restore .\TensorRtSharp.sln
powershell -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
dotnet build .\TensorRtSharp.sln -c Debug --no-restore
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build
dotnet docfx .\docs\docfx.json
```

If manifest/native/generated files changed, also run:

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
powershell -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
```

Then run the common examples from `samples/README.md` and the validation runners from `smoke/README.md`.

Recommended common-example order:

1. `MultiStream`
2. `DynamicShape`
3. `InferenceBindings`
4. `OnnxToEngine`

Recommended smoke order:

1. `CudaSmokeRunner`
2. `TensorRtSmokeRunner`
3. `LifecycleSmokeRunner`
4. `OnnxToEngineSmokeRunner`
5. `NetworkBuilderSmokeRunner`
6. Layer-specific network runners

## Samples

Runnable deployment samples are under `samples/`. Validation-oriented smoke runners are under `smoke/`.

Recent sample maturity updates:

- `MultiStream` is a real CUDA multi-stream/event ordering sample and is included in the solution.
- `CudaRuntimeCompilation` is an owner-safe NVRTC compile plus dual Runtime-library/Driver-module named typed-kernel launch/readback sample covering copied PTX/CUBIN/LTO IR, failure logs, lowered names, determinism, early owner disposal, and per-value GPU correctness. A repository-external local-feed consumer now verifies the same CUDA 12.9 paths from managed and bridge-only `PackageReference` packages; CUDA 13.2 PTX load rejection, Linux, public-package, and post-publish evidence remain separate.
- `DynamicShape` is a real TensorRT dynamic-shape/profile/binding sample and is included in the solution.
- `InferenceBindings` is a real TensorRT inference-binding sample and is included in the solution.
- `OnnxToEngine` is now a runnable common ONNX-to-engine example and is included in the solution.
- `Classification` and `YoloVision` are runnable asset-dependent ONNX examples; users provide their own model, labels, and input-shape metadata.
- `applications/TensorRtExec` is the user-facing ONNX-to-engine CLI/WinForms tool. It can create build/precheck reports for external ONNX assets, but real-model runtime proof still belongs to the relevant sample runner and package-consumer-runtime proof belongs to release proof records.
- Custom-kernel launch is available through owner-safe named-kernel and typed-argument APIs: `CudaKernelLibrary.Launch(...)` uses the CUDA 12.9+ Runtime library, while `CudaDriverModule.Launch(...)` dynamically loads the CUDA Driver for a unified module path. Local PTX launch/readback is verified for CUDA 11.8/12.1/12.9 artifacts on the current Driver 12090 host. The bridge-only package contains only `jyppxtrtbridge.dll`, not NVRTC or its builtins; the CUDA 12.9 clean local-feed consumer verifies missing-NVRTC diagnostics and dual-path correctness without `ProjectReference` or `JYPPX_NATIVE_BRIDGE_PATH`. This remains a local candidate, not public-package or post-publish proof.

## Runtime Packages

As of 2026-07-30, NVIDIA runtime redistribution is retired. Published artifacts are limited to the managed C# package, versioned project-owned `.Bridge` packages, and tracked source archives. Consumers install matching CUDA, cuDNN, TensorRT, and optional NVRTC dependencies themselves.

The runtime keys below remain build compatibility keys. They select which locally installed headers and import libraries compile the bridge; they do not authorize copying vendor DLLs or shared libraries into a package.

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 `
  -SourceRuntimeKey win-x64-trt11.0-cuda12.9-cudnn9.22 `
  -Version 4.0.0 `
  -SplitPackageRole bridge

powershell -ExecutionPolicy Bypass -File .\eng\Test-ExternalVendorRuntimePackagePolicy.ps1 `
  -PackagePath .\artifacts\runtime-split-nupkg\win-x64-trt11.0-cuda12.9-cudnn9.22
```

See `docs/articles/zh-cn/external-vendor-runtime-package-policy.md` and `pack/external-vendor-runtime-policy.json`.

<details>
<summary>Historical full-runtime notes (retired; do not use for publication)</summary>

Runtime packages carry native deployment assets for one explicit TensorRT / CUDA / cuDNN combination. Current Windows runtime package keys:

- `win-x64-trt8.6-cuda11.8-cudnn8.9`
- `win-x64-trt8.6-cuda12.1-cudnn8.9`
- `win-x64-trt10.11-cuda11.8-cudnn8.9`
- `win-x64-trt10.11-cuda12.9-cudnn9.22`
- `win-x64-trt11.0-cuda12.9-cudnn9.22`
- `win-x64-trt11.0-cuda13.2-cudnn9.22`

Current local status:

- TensorRT 10 + CUDA 11.8 is the stable real vendor-backed smoke path and has package-consumer smoke evidence.
- TensorRT 10 + CUDA 12.9 and TensorRT 11 + CUDA 12.9 have local runtime/package validation and package-consumer smoke evidence.
- TensorRT 11 + CUDA 13.2 bridge builds, collects assets, packs, and passes package consumer restore/build/native-copy; runtime/builder smoke is pending on a CUDA 13-capable driver/runtime environment. Any release containing this bridge must label it as `build-package-validated-runtime-unverified` and link the limitation in `docs/articles/zh-cn/runtime-package-matrix.md`.
- Linux runtime packages now include the OS/architecture in the package identity. Ubuntu 22.04 x64 is the default hosted matrix for all six combinations; Ubuntu 24.04 x64 is limited to the modern TensorRT 10/11 combinations that NVIDIA publishes for that distro; Ubuntu 20.04 x64 uses the hosted-container lane with an `ubuntu:20.04` job container.

See:

- `docs/articles/en/runtime-packages.md`
- `docs/articles/en/runtime-distribution-strategy.md`
- `docs/articles/en/package-consumer-validation.md`
- `docs/articles/en/release-candidate-gate.md`
- `docs/articles/en/api-reference.md`

</details>

## Release Automation

Formal releases run only from the `guojin-yan` repository. The `grape-yan` repository is build/test validation-only and has read-only workflow permissions with no package push or Release upload job.

Current release workflows publish only:

- `package-managed.yml`: `JYPPX.TensorRT.CSharp.API` plus the pure managed `JYPPX.TensorRT.CSharp.API.YoloVision` and `JYPPX.TensorRT.CSharp.API.Classification` extensions;
- `runtime-windows.yml` / `runtime-linux.yml`: `.Bridge` packages with `split_package_roles=bridge`;
- `package-source.yml`: a tracked-files-only source archive.

Every upload path runs `eng/Test-ExternalVendorRuntimePackagePolicy.ps1`. The managed workflow requires an exact three-package ID/version allowlist, matching nuspec source commits, extension surface checks, and a repository-external managed-only consumer. `release-bundle.yml` defaults all publication/deployment inputs to `false`; any docs, package, or Release side effect additionally requires `owner_publish_approved=true` in the formal repository.

For nuget.org publication, `NUGET_API_KEY` must be an active plain-text key with push permission for `JYPPX.TensorRT.CSharp.API`, `JYPPX.TensorRT.CSharp.API.YoloVision`, and `JYPPX.TensorRT.CSharp.API.Classification`, or for their owning account/organization. A nuget.org `403` is non-retryable until the package owner supplies a valid package-scoped key.

Public publication also requires `eng/Test-PublicationLicenseReadiness.ps1` to pass. Local pack/dry-run may continue while the license is an Owner decision, but no GitHub Release creation, Release upload, NuGet push, or GitHub Packages push may occur until every nupkg declares a non-placeholder license and the tracked source archive contains the selected root license file.

<details>
<summary>Historical vendor-package migration note</summary>

Before 2026-07-30, this repository modeled package roles that could carry CUDA, cuDNN, TensorRT, NVRTC, parser, plugin, builder-resource, collection, and meta assets. Those publication paths are retired and must not be replayed.

The historical package identities remain in selected manifests only for compatibility audits and interpretation of old evidence. Their project files have been removed. `eng/Invoke-LocalRuntimePackage.ps1` fails closed, and `eng/Invoke-LocalSplitRuntimePackage.ps1` accepts only `bridge`.

After explicit Owner fingerprint review, 65 retired GitHub Package versions and 65 matching Release assets were deleted from `guojin-yan/TensorRT-CSharp-API`. The post-delete inventory contained zero remaining deletion candidates. Managed and `.Bridge` package identities were preserved.

Current formal release rules:

- publish only `JYPPX.TensorRT.CSharp.API`, explicitly allowlisted pure managed extensions such as `JYPPX.TensorRT.CSharp.API.YoloVision`, matching `.Bridge` packages, and tracked-files-only source archives;
- keep NVIDIA libraries as consumer-installed machine prerequisites;
- run `eng/Test-ExternalVendorRuntimePackagePolicy.ps1` on every pack/upload path;
- run formal publication only from `guojin-yan`; use `grape-yan` for build/test validation only;
- require same-commit managed/YoloVision/bridge provenance, clean external consumer evidence, post-publish verification, and Owner approval before release closure.

</details>

## Repository Layout

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

## Build Prerequisites

- .NET SDK 10.0.300 or later
- CMake 3.27 or later
- Visual Studio C++ toolchain on Windows
- Matching local TensorRT / CUDA / cuDNN roots for native builds and runtime-package validation

## Quick Start

```powershell
dotnet restore .\TensorRtSharp.sln
dotnet build .\TensorRtSharp.sln -c Debug --no-restore
powershell -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build
```

To build the current high-version native preset:

```powershell
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
```

## Dependency Discovery

Use one of the scripts below to inspect local TensorRT/CUDA/cuDNN roots:

- `eng/Get-Dependencies.ps1`
- `eng/get-dependencies.sh`

Windows local roots are intentionally not stored in the public runtime manifest. Use `pack/runtime/runtime-packages.local.json` for machine-specific root overrides; start from `pack/runtime/runtime-packages.local.example.json`.

To audit locally installed NVRTC/builtins identity, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-CudaRtcFullRuntimePackagingPreflight.ps1
```

This script is a host-dependency identity audit only: it does not copy, package, or publish vendor files. The `cuda-rtc` role is permanently `retired-not-packable`; `.Bridge` packages never carry NVRTC or matching builtins.

Managed runtime loading is production-first:

- normal probing checks the app base directory and `runtimes/<rid>/native`
- explicit bridge loading uses `JYPPX_NATIVE_BRIDGE_PATH`
- explicit vendor roots use `JYPPX_TENSORRT_ROOT` and `JYPPX_CUDA_ROOT`
- local `build-out` / `third_party` development scanning requires `JYPPX_ENABLE_DEVELOPMENT_PROBING=1`

## Documentation Entry Points

- `docs/index.md`
- `docs/articles/en/getting-started.md`
- `docs/articles/en/installation-layout.md`
- `docs/articles/en/api-coverage-and-deferred-boundaries.md`
- `docs/articles/en/cuda-runtime-compilation-roadmap.md`
- `docs/articles/zh-cn/cuda-runtime-compilation-roadmap.md`
- `docs/articles/zh-cn/cuda-runtime-compilation-technical-article.md`
- `docs/articles/en/sample-runners.md`
- `docs/articles/en/runtime-packages.md`
- `docs/articles/en/package-consumer-validation.md`
- `docs/articles/en/release-candidate-gate.md`

Build documentation with:

```powershell
dotnet docfx .\docs\docfx.json
```

## Notes

- NVIDIA binaries are intentionally not committed.
- `third_party/` is only a local drop location.
- Runtime packages contain only the project-owned bridge. Install matching TensorRT, CUDA, cuDNN, and optional NVRTC components from NVIDIA on the consumer machine.
- Public hand-written C# wrappers should include useful XML documentation; generated APIs may use generated comments.

- Public release owner execution package: `artifacts/final-release/public-release-owner-execution-package.json`
- External clean consumer proof kit: `artifacts/final-release/external-clean-consumer-proof-kit.json`
- Runtime proof compatible host kit: `artifacts/final-release/runtime-proof-compatible-host-kit.json`
- Post-publish owner verification kit: `artifacts/final-release/post-publish-owner-verification-kit.json`
- Owner public release execution readiness pack: `artifacts/final-release/owner-public-release-execution-readiness-pack.json`
- Owner external real proof input contract: `artifacts/final-release/owner-external-real-proof-input-contract.json`
- Owner external real proof import validator: `artifacts/final-release/owner-external-real-proof-import-validator.json`
- Post-publish clean consumer real proof gate: `artifacts/final-release/post-publish-clean-consumer-real-proof-gate.json`
- Runtime compatible host real proof gate: `artifacts/final-release/runtime-compatible-host-real-proof-gate.json`
- Release close real proof readiness gate: `artifacts/final-release/release-close-real-proof-readiness-gate.json`
- Release candidate real proof final freeze: `artifacts/final-release/release-candidate-real-proof-final-freeze.json`
- Owner real input import preflight: `artifacts/final-release/owner-real-input-import-preflight.json`
- Public package hash cross-check gate: `artifacts/final-release/public-package-hash-cross-check-gate.json`
- Clean consumer runtime proof cross-check gate: `artifacts/final-release/clean-consumer-runtime-proof-cross-check-gate.json`
- ReleaseClose real input candidate promotion readiness: `artifacts/final-release/release-close-real-input-candidate-promotion-readiness.json`
- Post-publish rollback owner decision gate: `artifacts/final-release/post-publish-rollback-owner-decision-gate.json`
- Release close final real input admission pack: `artifacts/final-release/release-close-final-real-input-admission-pack.json`
- Owner real input JSON contract: `artifacts/final-release/owner-real-input-json-contract.json`
- Owner real input JSON import: `artifacts/final-release/owner-real-input-json-import.json`
- Owner real input hash and path validator: `artifacts/final-release/owner-real-input-hash-and-path-validator.json`
- Owner real input forbidden substitute validator: `artifacts/final-release/owner-real-input-forbidden-substitute-validator.json`
- Strict close real input dry-run: `artifacts/final-release/strict-close-real-input-dry-run.json`
- Strict close real input finding report: `artifacts/final-release/strict-close-real-input-finding-report.json`
- Strict close owner action pack: `artifacts/final-release/strict-close-owner-action-pack.json`
- Release close real input final blocker ledger: `artifacts/final-release/release-close-real-input-final-blocker-ledger.json`
- Clean external package consumer owner runbook: `artifacts/final-release/clean-external-package-consumer-owner-runbook.json`
- Clean external package consumer owner runbook validation: `artifacts/final-release/clean-external-package-consumer-owner-runbook-validation.json`
- Post-publish owner verification runbook: `artifacts/final-release/post-publish-owner-verification-runbook.json`
- Post-publish owner verification runbook validation: `artifacts/final-release/post-publish-owner-verification-runbook-validation.json`
- Clean consumer proof execution bundle: `artifacts/final-release/clean-consumer-proof-execution-bundle.json`
- Clean consumer external proof closure pack: `artifacts/final-release/clean-consumer-external-proof-closure-pack.json`
- Clean consumer external proof closure pack validation: `artifacts/final-release/clean-consumer-external-proof-closure-pack-validation.json`

`public-release-owner-execution-package`, `external-clean-consumer-proof-kit`, `runtime-proof-compatible-host-kit`, `post-publish-owner-verification-kit`, and `owner-public-release-execution-readiness-pack` add the real public release owner execution layer. They only provide command templates, owner fields, external proof collection paths, and readiness blockers while remaining blocked/non-proof. They are not runtime proof, post-publish proof, publish approval, release close approval, or package push.

`owner-external-real-proof-input-contract`, `owner-external-real-proof-import-validator`, `post-publish-clean-consumer-real-proof-gate`, `runtime-compatible-host-real-proof-gate`, and `release-close-real-proof-readiness-gate` add the real external proof backfill and ReleaseClose admission layer. They validate owner-filled fields, public source/hash/log/host metadata requirements, clean consumer proof, compatible-host runtime proof, and final close readiness while remaining blocked/non-proof. They are not runtime proof, post-publish proof, publish approval, release close approval, or package push.

`release-candidate-real-proof-final-freeze`, `owner-real-input-import-preflight`, `public-package-hash-cross-check-gate`, `clean-consumer-runtime-proof-cross-check-gate`, `post-publish-rollback-owner-decision-gate`, and `release-close-final-real-input-admission-pack` add the final real owner input admission layer. They freeze local evidence hashes, preflight owner input, cross-check public package hashes, cross-check clean consumer/runtime proof metadata, capture rollback owner decisions, and aggregate final ReleaseClose blockers while remaining blocked/non-proof. They are not runtime proof, post-publish proof, publish approval, release close approval, or package push.

`release-close-real-input-candidate-promotion-readiness` adds the final candidate-promotion readiness layer above those admission gates. It keeps all 11 promotion lanes blocked until real Owner inputs pass the required strict validators and explicitly rejects local feed, ProjectReference, direct nupkg, dry-run, dashboard, runbook, candidate, draft, and build-only substitutes.

`owner-real-input-json-contract`, `owner-real-input-json-import`, `owner-real-input-hash-and-path-validator`, `owner-real-input-forbidden-substitute-validator`, `strict-close-real-input-dry-run`, `strict-close-real-input-finding-report`, `strict-close-owner-action-pack`, and `release-close-real-input-final-blocker-ledger` add the strict close real input validation layer. They define and locally validate owner input fields, report blockers, and produce owner action guidance while remaining blocked/non-proof. They are not runtime proof, post-publish proof, publish approval, release close approval, or package push.

`clean-external-package-consumer-owner-runbook` and `post-publish-owner-verification-runbook` turn the remaining owner work into executable external-consumer checklists. They require repository-external projects, public or owner-approved package sources, stdout/stderr/merged transcript hashes, package hashes, validator output hashes, host metadata, owner review, and non-substitute confirmations. They remain blocked/non-proof until the owner supplies real execution results and strict validators promote concrete records.

`clean-consumer-proof-execution-bundle` and `clean-consumer-external-proof-closure-pack` connect local smoke classification, external clean consumer execution, compatible-host runtime metadata, owner result import, post-publish clean consumer evidence, and strict close admission into one blocked owner-action chain. They are execution maps and convergence packs only: they do not run runtime smoke, publish packages, approve public release, close the release issue, or promote runtime/post-publish proof without real external logs, SHA256 values, package metadata, native asset evidence, host metadata, owner review, and strict validators with `-RequireExistingLog` / `-FailOnNotProof`.
