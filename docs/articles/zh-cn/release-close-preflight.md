# Release Close Preflight

`release-close-preflight` 是 release issue 关闭前的聚合预检。它不发布、不上传、不删除包；它只读取现有 proof/validation/summary artifact，判断是否仍缺真实 external runtime proof、owner authorization、post-publish verification proof 或 stale claim 清理。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseClosePreflight.ps1
```

默认输出：

- `artifacts/final-release/release-close-preflight.json`
- `artifacts/final-release/release-close-preflight.md`

## 输入来源

预检会读取：

- `artifacts/final-release/external-runtime-proof-validation.json`
- `artifacts/final-release/owner-authorized-publish-command-plan-validation.json`
- `artifacts/final-release/post-publish-verification-validation.json`
- `artifacts/final-release/release-issue-close-record-validation.json`
- `artifacts/final-release/stale-release-claims-audit.json`
- `artifacts/final-release/release-candidate-full-acceptance-summary.json`
- `artifacts/final-release/post-publish-clean-consumer-project-scan.json`

## 预检项

`preflightItems` 固定覆盖：

- `external-runtime-proof-record`
- `owner-authorized-command-plan`
- `post-publish-clean-consumer-scan`
- `post-publish-verification-record`
- `stale-release-claims`
- `full-acceptance-close-readiness`
- `release-issue-close-record`

当前没有真实 external runtime proof 和真实 post-publish proof 时，状态必须保持：

- `preflightState=blocked-real-proof-required`
- `failedItemCount=9`
- `releaseIssueCloseRecordValidationState=blocked-template-only`
- `releaseIssueCloseRecordCanPromote=false`
- `canCloseReleaseIssue=false`
- `performsPublish=false`

## 不可替代 Proof 清单

以下内容不能作为 release close proof：

- template
- draft
- runbook
- collection package
- local inventory
- local feed
- ProjectReference
- helper
- build-only
- parse-only
- sidecar-only
- dependency-probe-only
- blocked-by-cuda-driver
- schema-only release issue close record
- template-only release issue close record
- preflight-only release issue close record
- release-issue-close-record-template.json

这些产物可以帮助 owner 采集证据，但不能把 release issue 变成可关闭。

## 推荐收口顺序

1. 生成 external runtime proof template/input draft/collection package。
2. owner 在兼容 CUDA/TensorRT 主机采集真实 external runtime proof。
3. 通过 `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof`。
4. owner 授权并手工执行真实发布。
5. 在源码仓库外创建 clean consumer 并运行 clean consumer scan。
6. 生成 post-publish input draft，回填真实 package/channel/log/host/smoke 字段。
7. 通过 `Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof`。
8. 刷新 stale release claims audit、full acceptance summary、release evidence bundle 和 release close preflight。
9. owner 回填 `release-issue-close-record.json`，必须引用真实 post-publish proof、release close preflight、stale claim audit、evidence bundle SHA256、rollback plan 和最终关闭决定。
10. 运行 `Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady`。
11. 只有所有 preflight items 与 release issue close record validator 通过后，才进入 owner 手工关闭 release issue。

## 不能误读

- `release-close-preflight` 不是发布授权。
- `ready-for-owner-proof-collection` 不是 release close readiness。
- clean consumer scan 是 helper evidence，不是 post-publish proof。
- TensorRtExec build-only report、parse-only option coverage、evidence sidecar 都不能替代 clean package consumer runtime proof。
- owner command plan 仍是 placeholder-only，不得由自动化执行真实发布命令。
- `release-issue-close-record-template.json`、`release-issue-close-record-validation=blocked-template-only` 或缺 evidence bundle SHA256 的 close record 都不能关闭 release issue。
