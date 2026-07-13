# Post Publish Clean Consumer Proof Record Draft

`post-publish-clean-consumer-proof-record-draft` 是发布后仓库外 clean consumer 验证的 Owner 草稿填写面。它要求记录外部 consumer 项目、公开包源、restore/build/smoke 命令、host metadata、日志 SHA256、native asset listing 和 reviewer 信息。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishCleanConsumerProofRecordDraft.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishCleanConsumerProofRecordDraft.ps1 -Strict
```

## 产物

- `artifacts/final-release/post-publish-clean-consumer-proof-record-draft.json`
- `artifacts/final-release/post-publish-clean-consumer-proof-record-draft.md`
- `artifacts/final-release/post-publish-clean-consumer-proof-record-draft-validation.json`
- `artifacts/final-release/post-publish-clean-consumer-proof-record-draft-validation.md`

## 边界

默认状态为 `blocked-post-publish-clean-consumer-proof-record-required`，`RequiredFields=19` 且 `Blocked=19`。它不是 post-publish proof，只是等待 Owner 回填真实仓库外运行结果的 draft。

必须保持：

- `notExecutedByAutomation=true`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isPostPublishProof=false`
- `isReleaseCloseProof=false`

## Owner 下一步

Owner 必须在仓库外 clean consumer 项目中使用公开包源完成 restore/build/smoke，并回填真实日志、hash、exit code、host/runtime metadata 和 reviewer。local feed、ProjectReference、direct nupkg、template、runbook 或本仓库内 smoke 不能替代该 proof record。
