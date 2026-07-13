# Owner Release Execution Package

`owner-release-execution-package` 是发布候选进入真实 owner 执行前的总调度材料。它把 final gap review、release close preflight、release evidence bundle、owner action、external runtime proof、post-publish verification、Classification/YoloVision 样例 evidence 和 Linux runner blocker 放到同一个执行顺序里，方便 release owner 按步骤补齐真实 proof。

它不是发布脚本，也不是 release proof record。生成脚本 `eng/Export-OwnerReleaseExecutionPackage.ps1` 只写出 `artifacts/final-release/owner-release-execution-package.json` 和 `artifacts/final-release/owner-release-execution-package.md`，并明确保持：

- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `packageState=blocked-real-proof-required`
- `releaseIssueCloseRecordValidationState=blocked-template-only`

## 为什么还需要这个执行包

前面的材料已经分别解决了多个问题：

- `release-candidate-final-gap-review` 说明当前 release candidate 还缺哪些真实 proof。
- `release-close-preflight` 聚合关闭 release issue 前的 blocker。
- `owner-action-required.md` 给 owner 一个待办清单。
- `owner-authorized-publish-command-plan` 约束真实发布命令只能由 owner 手动授权。
- `external-runtime-proof-*` 和 `post-publish-verification-*` 分别处理发布前 runtime proof 与发布后 clean consumer proof。
- `release-issue-close-record-*` 是最后的 owner close gate，用于把真实 proof、evidence bundle SHA256、rollback plan 和最终关闭决定绑到同一条审计记录。

Owner execution package 的价值不是替代这些材料，而是把它们排成一个可执行顺序，减少 owner 在最后一公里来回翻文件。它回答的是：“现在应该先跑哪个 validator、准备哪个 JSON、哪个动作必须等真实发布之后才能做、哪些材料绝不能被当作 proof。”

`owner-proof-backfill-execution-pack` 是它的聚焦 companion。前者负责总调度和一屏 Release Hold 清单，后者把 `owner-authorization`、`package-consumer-runtime`、`linux-runner-proof`、`real-model-runtime`、`post-publish-verification` 和 `release-issue-close-record` 拆成可执行的真实输入字段、first command、validator、expected artifacts、source artifacts 和 cannot-use 材料。两者都保持 `performsPublish=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`。

## 一屏 Release Hold 清单

生成产物现在会输出 `oneScreenReleaseHoldChecklist`，把 owner 当前最短执行面压缩成一张表。它不是新的 proof 类型，而是把五个仍处于 `blocked-real-proof-required` 的 blocker 放到同一视图：

- owner authorization。
- `package-consumer-runtime`。
- Linux runner proof。
- `real-model-runtime`。
- post-publish verification。

每一项都会列出 owner-visible blocker、当前状态、下一步动作、first command、validator command、required real inputs 和 cannot-use 材料。这样 owner 不需要在 release close preflight、evidence bundle、owner authorization、sample evidence、Linux runner 与 post-publish 文档之间来回切换，也不会把 local feed、ProjectReference、blocked-by-cuda-driver、build-only、parse-only 或 sidecar-only 当成可关闭 release issue 的真实 proof。

## 执行顺序

推荐顺序如下：

1. 刷新 stale claim audit，确保文章、README 和 artifact 没有提前写成已发布或已关闭。
2. 在兼容 CUDA/TensorRT 主机采集 `package-consumer-runtime` proof。
3. 回填 Classification/YoloVision 的 `real-model-runtime` proof。
4. 回填 Linux runner proof。
5. 校验 owner authorization 与手动 publish command plan。
6. owner 在外部手动执行真实发布命令。
7. 发布后扫描 clean consumer project，确认没有 `ProjectReference`。
8. 回填并校验 post-publish verification proof。
9. 刷新 release close preflight 和 release evidence bundle。
10. 回填并校验 release issue close record。
11. owner 手工复核 evidence bundle SHA256、rollback plan 和最终关闭决定。
12. 所有真实 proof 与 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady` 通过后，再手工关闭 release issue。

## 发布前 proof

发布前最关键的是 `package-consumer-runtime`。它必须来自干净 consumer 的真实 runtime smoke 记录，而不是本仓库内的 build-only、local feed 或 ProjectReference 路径。

必须使用：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof
```

有效记录需要包含真实 managed/runtime nupkg SHA256、consumer project identity、runtime package key、host metadata、smoke command、stdout/stderr summary 和日志 SHA256。`DependencyProbe`、`blocked-by-cuda-driver`、`build-only`、`sidecar-only` 和 `parse-only` 都不能晋级为 `package-consumer-runtime` proof。

## 样例 proof

Classification 和 YoloVision 的真实样例运行只能证明 `real-model-runtime`，不能替代 release proof record。YoloVision 当前覆盖 YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom 的 det、cls、seg、obb、pose、sem 等任务面，但真实 proof 仍需要 owner 提供：

- 模型来源和 license。
- labels 或类别定义。
- 输入图片或预处理 tensor。
- model/input/labels/log 的 SHA256。
- TensorRtExec build sidecar。
- sample runner log。
- sample-run-evidence record。

校验路径是 `Test-SampleAssetManifest.ps1` 与 `Test-SampleRunEvidenceRecord.ps1`。即使这些全部通过，也只说明样例真实运行成立，不能写成 `package-consumer-runtime`。

## 发布后 proof

Post-publish verification 只能在真实渠道发布之后完成。它需要 clean consumer project 从真实渠道下载 package，记录 package identity、downloaded nupkg SHA256、restore/build/smoke logs、stdout/stderr summary 和对应日志 hash。

必须使用：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof
```

`local feed`、`ProjectReference`、draft、template、collection package 和 helper scan 不能关闭 release issue。clean consumer scan 只是 post-publish proof 的组成部分，不是完整 proof。

## Release Issue Close Record

Release issue close record 是 post-publish 之后的最后一道 owner close gate，不是 proof 采集模板。当前模板和 validation 必须保持：

- `release-issue-close-record-validation=blocked-template-only`
- `proofClassification=template-only`
- `canPromoteReleaseIssueCloseRecord=false`
- `canCloseReleaseIssue=false`

真实关闭前必须运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady
```

该记录必须引用真实 post-publish proof、release close preflight、stale claim audit、release evidence bundle SHA256、rollback plan 和 owner final close decision。`release-issue-close-record-template.json`、schema-only、preflight-only、缺 SHA256 或缺 owner 决定的记录都不能关闭 release issue。

## 手动发布占位

执行包会列出 `dotnet nuget push`、GitHub Packages upload 和 GitHub Release upload 的手动命令模板，但这些模板只用于 owner 审阅和材料化。脚本不会执行任何真实发布、撤回、删除、delist 或 withdraw 操作。

执行包中的英文边界短语是 `owner manually executes publish commands outside this script`。它用于提醒后续自动化和 quality tests：发布命令只能由 owner 在脚本之外手动执行。

这条边界很重要：自动化可以帮助 owner 减少漏项，但不能替 owner 做授权，也不能把真实渠道动作伪造成已经完成。

## 不可替代材料

以下材料可以作为诊断、脚手架或执行指南，但不能作为 release close proof：

- helper
- template
- draft
- runbook
- collection package
- local inventory
- local feed
- ProjectReference
- DependencyProbe
- dependency-probe-only
- blocked-by-cuda-driver
- build-only
- parse-only
- sidecar-only
- Windows handoff for Linux proof
- schema-only release issue close record
- template-only release issue close record
- preflight-only release issue close record

## 与 TensorRtExec 的边界

TensorRtExec 已经提供 console 和 WinForms 双入口，并把大量 trtexec-like 参数接入 parser/report/GUI。但高级参数仍需要保持 `TrtexecAlignmentStatus=parse-only` 的保守边界：参数能被记录，不等于底层 TensorRT 行为和模型级 smoke 已经证明。

因此 TensorRtExec build report、evidence sidecar 和 parse-only report 不能替代 `package-consumer-runtime` 或 post-publish verification proof。

## 完成判断

只有当以下条件同时成立时，owner 才能考虑 release close review：

- `Test-StaleReleaseClaims.ps1` findingCount 为 0。
- `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof` 通过。
- owner authorization 明确存在，且 publish command 不再是 placeholder-only。
- 真实发布已经由 owner 手动完成。
- clean consumer project 没有 `ProjectReference`，并使用真实渠道 package identity。
- `Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof` 通过。
- Linux runner proof 和必要样例 proof 已按 release scope 回填。
- `Export-ReleaseClosePreflight.ps1` 不再显示 `blocked-real-proof-required`。
- `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady` 通过，且 release issue close record 包含 release evidence bundle SHA256、rollback plan 和 owner final close decision。

在这些条件满足前，`owner-release-execution-package` 必须保持 owner guidance，而不是 release proof。
