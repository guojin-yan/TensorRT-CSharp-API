# Owner Proof Real Backfill Execution Pack

`owner-proof-real-backfill-execution-pack` 是 `release-close-strict-record-candidate` 之后的 Owner 执行包。它不再只告诉 Owner “缺 proof”，而是把缺口拆成三类可执行任务：

- `ownerInputTasks`：来自 strict candidate 的 8 个 `requiredOwnerFields`，包括 rollback plan、rollback owner、rollback trigger、owner final close decision、release issue id、release issue url、public channel package source、clean consumer runtime smoke log。
- `realProofTasks`：来自 strict candidate 的 release close blockers，包括 post-publish verification、final close decision、release close candidate、overlay candidate、owner external execution result backfill kit。
- `hashCheckTasks`：来自 strict candidate 的 hash lines，用于确认本地 artifact/path/hash 一致，但 hash match 不能替代 proof。

## 当前边界

- `packState=blocked-owner-real-proof-backfill-required`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

该执行包是 owner handoff，不执行 publish / push / upload，不关闭 release issue，不把 template、draft、candidate、schema-only、preflight-only、dependency-probe-only、blocked-by-cuda-driver、local feed、ProjectReference 或 direct `.nupkg` 当作 proof。

## 产物

- `artifacts/final-release/owner-proof-real-backfill-execution-pack.json`
- `artifacts/final-release/owner-proof-real-backfill-execution-pack.md`
- `artifacts/final-release/owner-proof-real-backfill-execution-pack-validation.json`
- `artifacts/final-release/owner-proof-real-backfill-execution-pack-validation.md`

## 生成与验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerProofRealBackfillExecutionPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerProofRealBackfillExecutionPack.ps1 -Strict
```

严格验证只证明执行包结构合法、任务拆分完整、无发布/关闭副作用。只要真实 Owner 输入和真实 proof 没有回填，`failedActionRequiredCount>=1` 与 `blockedTaskCount>=1` 就是预期状态。
