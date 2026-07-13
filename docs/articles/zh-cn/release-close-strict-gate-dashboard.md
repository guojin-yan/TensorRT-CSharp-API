# ReleaseClose Strict Gate Dashboard

`release-close-strict-gate-dashboard` 是 release issue 关闭前的一页式 strict gate。它聚合 release evidence bundle、release close preflight、release issue close record、package-consumer-runtime、post-publish verification、sample-run-evidence 和 owner approval 的阻塞状态。

机器可读文件：

`artifacts/final-release/release-close-strict-gate-dashboard.json`

## 适用读者

- release owner。
- 最终 release close 审核者。
- 需要确认哪些 lane 仍然 blocked 的维护者。

## 解决问题

发布前材料已经很多，如果只看单个模板或报告，很容易误以为 release close 可以通过。这个 dashboard 只做一件事：把所有 blocked lane 放到同一页，并明确对应 validator command 与 blocked reason。

覆盖 lane：

- release evidence bundle。
- release close preflight。
- release issue close record。
- package-consumer-runtime。
- post-publish verification。
- sample-run-evidence。
- owner approval。

## 边界说明

以下内容不能关闭 release issue：

- build-only
- dry-run
- template
- local feed
- ProjectReference
- direct `.nupkg`
- TensorRtExec report
- YoloVision matrix
- OnnxToEngine report
- readonly diagnostics
- design gate
- blocked-by-cuda-driver

任何 template-only record、placeholder owner input、report、matrix、runtime proof substitute 或 dry-run 都不能成为 release close proof。该 dashboard 固定 `performsPublish=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`。

## 可复制命令

严格关闭前至少需要验证：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseClosePreflight.ps1

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady
```

这些命令不执行真实包发布，只验证已有记录。

## 下一步

1. 逐个解除 blocked lane，而不是合并 proof ladder。
2. 先完成 package-consumer-runtime 和 post-publish verification。
3. 再填 release issue close record 和 owner approval。
4. 最终使用 strict validator 判断是否可以 close。
