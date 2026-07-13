# Owner External Proof Backfill Orchestrator

`owner-external-proof-backfill-orchestrator` 是真实外部 proof 回填的 owner command planner。它读取 `owner-proof-input-draft-pack.json`、`owner-proof-input-repair-pack.json` 和 `release-evidence-bundle.json`，把 6 条 proof line 转成 target proof record、strict validator、required files、SHA256 fields、owner decision、rollback fields 和下一步命令。

它不是 proof，不采集 proof，不执行发布，不批准公开发布，也不关闭 release issue。默认必须保持：

- `recordKind=owner-external-proof-backfill-orchestrator`
- `orchestratorState=blocked-owner-external-proof-required`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- 所有 line `canPromoteProof=false`

## 生成与校验

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerProofInputDraftPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerProofInputDraftPack.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerExternalProofBackfillOrchestrator.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerExternalProofBackfillOrchestrator.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
```

输出：

- `artifacts/final-release/owner-external-proof-backfill-orchestrator.json`
- `artifacts/final-release/owner-external-proof-backfill-orchestrator.md`
- `artifacts/final-release/owner-external-proof-backfill-orchestrator-validation.json`
- `artifacts/final-release/owner-external-proof-backfill-orchestrator-validation.md`

## 覆盖的 Proof Line

- `owner-authorization`
- `package-consumer-runtime`
- `linux-runner-proof`
- `real-model-runtime`
- `post-publish-verification`
- `release-issue-close-record`

## Package Consumer Runtime

`package-consumer-runtime` 必须保持 blocked，直到 owner 在兼容 CUDA/TensorRT host 上生成真实 clean external consumer proof。orchestrator 必须要求：

- clean external consumer root outside repository
- no ProjectReference
- no local feed as public proof
- no direct `.nupkg` as public proof
- managed/runtime nupkg SHA256
- runtime package key match
- compatible host metadata
- smoke command with `--runtime-package-key`
- smoke log path
- smoke log SHA256
- `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof`

## Release Issue Close Record

`release-issue-close-record` 必须保持 blocked，直到所有真实 proof gate 通过并有 owner final close decision。orchestrator 必须要求：

- release evidence bundle SHA256
- release close preflight path/hash
- stale claims audit path/hash
- post-publish proof validation path/hash
- rollback plan
- owner final close decision
- `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`

## 不可替代边界

Orchestrator、draft pack、repair pack、input draft、template、handoff、preflight、runbook、collection package 和 readiness snapshot 都不是 proof。真实 proof 只能来自 target proof record 中的真实文件、匹配 SHA256、clean consumer evidence、owner decision、rollback plan，以及对应 strict validator 通过。
