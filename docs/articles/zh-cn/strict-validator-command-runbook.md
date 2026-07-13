# Strict Validator Command Runbook

`strict-validator-command-runbook` 是 owner 真实 proof 输入的严格验证命令手册。它列出四条 lane 的 validator command，并明确这些命令只验证 owner 输入，不执行真实发布。

机器可读文件：

`artifacts/final-release/strict-validator-command-runbook.json`

## 适用读者

- 准备逐条运行 strict validator 的 owner。
- 发布前最终审计者。
- 需要确认 sample、package、post-publish 和 close 不能互相替代的维护者。

## 解决问题

真实 proof 回填不是一次性动作。sample-run、package-consumer-runtime、post-publish verification 和 release close approval 的 validator 顺序、前置条件和边界不同。本 runbook 把命令、必须先满足的条件和不能替代的 lane 写清楚。

## 边界说明

runbook 固定 `performsPublish=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`。它不是 runtime proof，也不是 post-publish proof。它不会执行 package push，也不能让 template、dashboard、dry-run、report 或 matrix 晋级为 proof。

以下材料只能作为审计线索或执行输入，不能单独关闭发布门禁：

- TensorRtExec report
- YoloVision matrix
- OnnxToEngine report
- readonly diagnostics

核心边界：

- `sample-run-evidence` 不能替代 `package-consumer-runtime`。
- `package-consumer-runtime` 不能替代 `post-publish verification`。
- `post-publish verification` 必须在 selected-channel 真实发布后执行。
- `release-close-owner-approval` 必须等所有 strict validators 通过后才能填写。

## 可复制验证命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog -FailOnNotProof

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RequireExistingLog -FailOnNotProof

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json -RequireExistingLog -FailOnNotProof

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady
```

## 下一步

1. 用 owner evidence manifest 填写真实文件。
2. 逐条执行 validator。
3. 记录 validator 输出和 SHA256。
4. 进入 release issue close owner input final checklist。
