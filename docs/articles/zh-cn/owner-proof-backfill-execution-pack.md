# Owner Proof Backfill Execution Pack

`owner-proof-backfill-execution-pack` 是面向 release owner 的真实 proof 回填执行包。它不发布包、不上传资产、不关闭 release issue，只把当前仍缺的真实 proof 拆成可执行的字段、命令、validator 和不可替代边界。

生成脚本：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerProofBackfillExecutionPack.ps1
```

输出：

- `artifacts/final-release/owner-proof-backfill-execution-pack.json`
- `artifacts/final-release/owner-proof-backfill-execution-pack.md`

默认边界必须保持：

- `recordKind=owner-proof-backfill-execution-pack`
- `packageState=blocked-real-proof-required`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `releaseIssueCloseRecordValidationState=blocked-template-only`
- `releaseIssueCloseRecordCanPromote=false`

## 回填范围

执行包当前固定覆盖 6 条 owner proof 回填线：

- `owner-authorization`
- `package-consumer-runtime`
- `linux-runner-proof`
- `real-model-runtime`
- `post-publish-verification`
- `release-issue-close-record`

每一项都必须写清：

- `currentState`
- `firstCommand`
- `validatorCommand`
- `requiredRealInputs`
- `expectedArtifacts`
- `sourceArtifacts`
- `cannotUse`
- `blockerReason`

这让 owner 可以从一个材料里看到下一步先跑什么命令、需要准备哪些真实字段、最终由哪个 validator 判定，以及哪些已有材料不能被误升格为 proof。

## 不可替代材料

以下材料只能作为 guidance、template、diagnostic、precheck 或 collection context，不能替代真实 proof：

- local feed
- ProjectReference
- direct `.nupkg` reference
- helper scan
- build-only
- parse-only
- sidecar-only
- dependency-probe-only
- readiness snapshot
- template-only release issue close record
- preflight-only release issue close record
- schema-only release issue close record
- `release-issue-close-record-template.json`
- Windows handoff for Linux proof

## 与 Evidence Bundle 的关系

`release-evidence-bundle` 会把 `owner-proof-backfill-execution-pack.json` 和 `.md` 纳入 `sourceEvidence`、`sourceArtifacts` 和 `evidenceItems`。这只是传播 owner 指引，不会改变 release gate：

- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `readyBackfillItemCount=0`
- `release-issue-close-record-validation=blocked-template-only`

只有真实 package-consumer runtime、real model runtime、Linux runner、post-publish verification、owner authorization 和 final close record 全部回填并通过 validator 后，才能进入 release issue close review。

`owner-proof-execution-handoff` 是该执行包的下一层交接看板。它复用这里的 6 条 backfill line，并额外展示每条 line 的 existing candidate artifacts、missing real inputs、owner next action 和 canPromoteProof。它同样只是 guidance，不能发布包、不能批准公开发布、不能关闭 release issue。

## 最后关闭门槛

Release issue close 仍必须由真实 close record 和 validator 决定：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady
```

该记录必须绑定真实 post-publish proof、release close preflight、stale claim audit、release evidence bundle SHA256、rollback plan 和 owner final close decision。`owner-proof-backfill-execution-pack` 本身不是 close record，也不能把 `canCloseReleaseIssue` 改成 true。
