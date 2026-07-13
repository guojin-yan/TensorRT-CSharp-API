# Release Owner 最后一公里执行顺序

这篇文章把 release owner 最后一公里按真实执行顺序展开。它不是发布授权，也不会执行上传；它只说明 owner 在具备兼容主机、真实模型资产、Linux runner 和真实发布渠道后，应该如何把剩余 blocker 逐项转成可验证的 proof record。

当前仍需要保留：

- `blocked-real-proof-required`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

## 总览

最终 release close 不是单个脚本动作，而是一组顺序化 proof gate。owner 只能在所有 validator 通过后，才可以手动关闭 release issue。

| 顺序 | 阶段 | 关键 validator | 当前边界 |
| --- | --- | --- | --- |
| 1 | refresh stale claim audit | `Test-StaleReleaseClaims.ps1` | stale claim 为 0 只是文档干净，不是 runtime proof |
| 2 | fill owner proof input | `Test-ReleaseOwnerProofInputRecord.ps1` | 模板和 readiness snapshot 不是 owner proof |
| 3 | collect package-consumer-runtime proof | `Test-ExternalRuntimeProofRecord.ps1` | local feed / ProjectReference 不能替代 clean consumer |
| 4 | collect real-model-runtime proof | `Test-SampleAssetManifest.ps1` / `Test-SampleRunEvidenceRecord.ps1` | build-only / sidecar-only 不是真实模型运行 |
| 5 | collect Linux runner proof | `Test-LinuxRunnerEvidenceRecord.ps1` | Windows handoff 不是 Linux proof |
| 6 | validate owner authorization | `Test-OwnerAuthorizedPublishCommandPlan.ps1` | placeholder command 不代表授权 |
| 7 | owner manually executes publish commands | owner 手工执行 | 自动化不得执行真实发布 |
| 8 | scan clean post-publish consumer | `Test-PostPublishCleanConsumerProject.ps1` | scan 是辅助证据，不是 post-publish proof |
| 9 | validate post-publish verification proof | `Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof` | draft / collection package 不是 proof |
| 10 | refresh release close preflight | `Export-ReleaseClosePreflight.ps1` | preflight 是聚合门禁，不是 final close proof |
| 11 | fill release issue close record | `Test-ReleaseIssueCloseRecord.ps1` | template-only close record 不是 proof |
| 12 | owner manually closes release issue | owner 手工关闭 | 只有 validator 全部通过后才允许 |

## Step 1：Refresh Stale Claim Audit

目标是确认 README、docs、release notes 和文章没有把 blocked 状态写成 ready。

输入：

- release-facing docs。
- README / README.zh-CN。
- release articles。
- artifact markdown。

命令：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StaleReleaseClaims.ps1
```

输出：

- `artifacts/final-release/stale-release-claims-audit.json`
- `artifacts/final-release/stale-release-claims-audit.md`

通过条件：

- `findingCount=0`。

不可替代材料：

- 人工口头确认。
- 只检查 README 不检查 docs。
- 删除规则来绕过 stale claim。

## Step 2：Fill And Validate Owner Proof Input

目标是把 owner 授权、selected channel、package URL/hash、clean consumer、runtime log/hash、stdout/stderr 和 host metadata 填成一个可验证记录。

输入：

- `artifacts/final-release/release-owner-proof-input-record.json`
- owner identity。
- selected channel。
- managed/runtime package URL 和 SHA256。
- clean consumer restore/native asset/dependency probe/runtime smoke log。
- OS/GPU/driver/CUDA/TensorRT/cuDNN metadata。

命令：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseOwnerProofInputRecordTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseOwnerProofInputRecord.ps1 `
  -InputPath .\artifacts\final-release\release-owner-proof-input-record.json `
  -RequireExistingLogs `
  -FailOnNotProof
```

输出：

- `artifacts/final-release/release-owner-proof-input-record-validation.json`
- `artifacts/final-release/release-owner-proof-input-record-validation.md`

当前边界：

- 默认模板仍是 `blocked-template-only`。
- `release-owner-proof-input-record-template.json` 不是 proof。
- `owner-proof-input-readiness` 不是 proof。

## Step 3：Package Consumer Runtime Proof

目标是证明 `package-consumer-runtime` 包消费者路径，而不是源码工程路径。

最低输入：

- 仓库外 clean consumer。
- 无 `ProjectReference`。
- 真实 package source 或 owner 指定待发布 package input。
- managed package SHA256。
- runtime package SHA256。
- runtime package key。
- restore/build/native listing/dependency probe/runtime smoke log。
- stdout/stderr summary。

验证：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 `
  -InputPath .\artifacts\final-release\external-runtime-proof-record.json `
  -RequireExistingLog `
  -FailOnNotProof
```

失败处理：

- 如果是 `blocked-by-cuda-driver`，保留 blocker，换兼容主机。
- 如果是 dependency missing，补 native asset 或 package source。
- 如果 consumer 使用了 ProjectReference，重建 clean consumer。

## Step 4：Real Model Runtime

目标是证明 `real-model-runtime` 级别的 Classification / YoloVision 真实模型样例，而不是证明 release package consumer。

最低输入：

- ONNX 模型。
- labels。
- input image 或 input data。
- license / redistribution note。
- SHA256。
- TensorRtExec build-only report 或 sidecar。
- sample runner log。
- sample-run-evidence record。

YoloVision 范围固定：

- family：`YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom`
- task：`det、cls、seg、obb、pose、sem`

验证：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleAssetManifest.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog
```

失败处理：

- 如果只有 support matrix，不能晋级。
- 如果只有 sidecar-only，继续要求真实 runner log。
- 如果 output metadata 不清楚，先补 profile 和 postprocess。

## Step 5：Linux Runner Proof

目标是证明 Linux x64 package/runtime lane。

输入：

- Linux OS / arch。
- CUDA driver/runtime。
- TensorRT line/version。
- cuDNN version。
- runtime package key。
- restore/build/probe/smoke log。
- validator output。

验证：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidenceRecord.ps1
```

Windows handoff、template-only、runbook 和 dry-run summary 都不能替代真实 Linux runner proof。

## Step 6：Validate Owner Authorization

owner 确认是否允许进入真实发布流程。

输入：

- package id 和 version。
- 目标 package source / channel。
- NVIDIA TensorRT / CUDA / cuDNN redistribution disposition。
- 签名策略。
- 回滚策略。
- 发布命令由谁手工执行。

验证：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerAuthorizedPublishCommandPlan.ps1
```

自动生成的 command plan、execution package、template 和 approval input 只能帮助 owner 填字段，不能替代授权。

## Step 7：Owner Manually Executes Publish Commands

真实发布只能由 owner 在脚本之外手动执行。

典型命令类型：

- `dotnet nuget push <owner-reviewed-managed.nupkg> ...`
- `dotnet nuget push <owner-reviewed-runtime.nupkg> ...`
- `gh release upload <tag> <owner-reviewed-artifacts> ...`

边界：

- `owner-release-execution-package` 只保存 command template。
- 自动化脚本不得执行真实上传。
- 没有 owner 凭据和授权时，必须保持 `canPublishPublicly=false`。

## Step 8：Scan Clean Post-Publish Consumer

这一步只能在 owner 完成真实渠道发布后执行。它用于检查 clean consumer 项目是否仍然干净。

验证：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishCleanConsumerProject.ps1
```

边界：

- scan passed 只是辅助条件。
- 它不能替代 `post-publish-verification-record.json`。
- local feed、ProjectReference、direct `.nupkg` 都必须继续 blocked。

## Step 9：Post-Publish Verification

`post-publish verification` 用于证明真实渠道包下载后的 clean consumer restore/build/smoke 路径。

输入：

- package id/version/channel URL。
- downloaded nupkg SHA256。
- clean consumer project。
- no ProjectReference。
- restore/build/native listing/dependency probe/runtime smoke log。
- stdout/stderr summary。

验证：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 `
  -InputPath .\artifacts\final-release\post-publish-verification-record.json `
  -RequireExistingLog `
  -FailOnNotProof
```

input draft、clean consumer scan、collection package 和 local feed 都不是 post-publish proof。它们只帮助 owner 填好 record。

## Step 10：Refresh Release Close Preflight

真实 proof 回填后刷新聚合门禁：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseClosePreflight.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StaleReleaseClaims.ps1
```

当前默认状态：

- `release-close-preflight.preflightState=blocked-real-proof-required`
- `release-close-preflight.failedItemCount=9`
- `releaseIssueCloseRecordValidationState=blocked-template-only`
- `canCloseReleaseIssue=false`

只有 release close preflight 消除对应 blocker，且 stale claim audit 没有发现越级声明，才能继续 owner 的 release close 判断。

## Step 11：Fill And Validate Release Issue Close Record

这是最终 close issue 前的最后一份 owner 记录。它把真实 post-publish proof、release close preflight、stale audit、owner final close decision、evidence bundle SHA256 和 rollback plan 绑定在一起。

输入：

- `artifacts/final-release/release-issue-close-record.json`
- release issue id / URL。
- owner identity / approval timestamp。
- selected channel。
- managed/runtime package URL 和 SHA256。
- post-publish verification validation path。
- release close preflight path。
- stale release claims audit path。
- release evidence bundle SHA256。
- rollback / yanking / deprecation plan。

命令：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseRecordTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 `
  -InputPath .\artifacts\final-release\release-issue-close-record.json `
  -FailOnNotCloseReady
```

当前默认状态：

- `release-issue-close-record-validation.validationState=blocked-template-only`
- `release-issue-close-record-validation.proofClassification=template-only`
- `canPromoteReleaseIssueCloseRecord=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

不可替代材料：

- `release-issue-close-record-template.json`
- `release-close-preflight.json`
- `release-evidence-bundle.json`
- owner handoff。
- collection package。
- readiness snapshot。
- schema-only / dry-run-only / precheck-only。

## Step 12：Owner Manually Closes Release Issue

只有以下条件同时成立时，owner 才能手动关闭 release issue：

- `Test-StaleReleaseClaims.ps1` findingCount 为 0。
- `Test-ReleaseOwnerProofInputRecord.ps1 -RequireExistingLogs -FailOnNotProof` 通过。
- `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof` 通过。
- `Test-SampleAssetManifest.ps1` 和 `Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog` 通过。
- `Test-LinuxRunnerEvidenceRecord.ps1` 通过。
- `Test-OwnerAuthorizedPublishCommandPlan.ps1` 通过。
- owner 已经手动执行真实发布。
- `Test-PostPublishCleanConsumerProject.ps1` 通过。
- `Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof` 通过。
- `Export-ReleaseClosePreflight.ps1` 不再显示 `blocked-real-proof-required`。
- `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady` 通过。

在这些条件满足前，`owner-release-execution-package` 必须保持 owner guidance，而不是 release proof。

## 最后一公里原则

1. 先证明包消费者，再证明真实模型，再证明真实渠道。
2. 每一步都保留日志和 SHA256。
3. validator 不通过时保留 blocker。
4. 不修改 quality test 来制造完成。
5. 不把 helper、template、draft、runbook、collection package、input package、local feed、ProjectReference、build-only、parse-only、sidecar-only、precheck-only、dry-run-only、schema-only 或 `blocked-by-cuda-driver` 写成 proof。
6. 不把 `release-close-preflight` 当 final close proof。
7. 不把 `release-issue-close-record-template` 当 final close proof。
