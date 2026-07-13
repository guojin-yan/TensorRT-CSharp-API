# Final Release Close Hash Consistency Gate

`final-release-close-hash-consistency-gate` 是最终关闭记录的本地 hash 一致性门禁。它覆盖 release evidence bundle、final freeze、post-publish validation、公开发布草稿、clean consumer 草稿、真实字段合同、Owner 投影和 release issue close validation 共 8 条 hash lane。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalReleaseCloseHashConsistencyGate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalReleaseCloseHashConsistencyGate.ps1 -Strict
```

## 产物

- `artifacts/final-release/final-release-close-hash-consistency-gate.json`
- `artifacts/final-release/final-release-close-hash-consistency-gate.md`
- `artifacts/final-release/final-release-close-hash-consistency-gate-validation.json`
- `artifacts/final-release/final-release-close-hash-consistency-gate-validation.md`

## 边界

默认状态为 `blocked-final-release-close-hash-consistency-owner-proof-required`，`HashLanes=8`、`Mismatched=0` 且 `Blocked=7`。它只检查当前本地 artifact hash 一致性，不发布包、不批准公开发布、不生成 runtime proof、post-publish proof 或 release close proof，也不关闭 release issue。

必须保持：

- `notExecutedByAutomation=true`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isPostPublishProof=false`
- `isReleaseCloseProof=false`
- `isReleaseCloseRecordProof=false`

## Owner 下一步

Owner 必须先补齐真实公开包、clean consumer proof、最终关闭记录和 strict validation，再重新生成 hash gate。hash 一致不等于可关闭；只要 proof 或 Owner approval 仍缺失，release issue 必须保持打开。
