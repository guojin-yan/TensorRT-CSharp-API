# Release Close Real Proof Import Bridge

`release-close-real-proof-import-bridge` 是最终 release close 前的真实 proof 导入桥。它把公开发布结果、post-publish clean consumer proof、runtime proof、rollback review、final owner decision 和 strict close validator 聚合成 6 条关闭 lane。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCloseRealProofImportBridge.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCloseRealProofImportBridge.ps1 -Strict
```

## 产物

- `artifacts/final-release/release-close-real-proof-import-bridge.json`
- `artifacts/final-release/release-close-real-proof-import-bridge.md`
- `artifacts/final-release/release-close-real-proof-import-bridge-validation.json`
- `artifacts/final-release/release-close-real-proof-import-bridge-validation.md`

## 边界

默认状态为 `blocked-release-close-real-proof-import-required`，`Lanes=6` 且 `Blocked=6`。它只是 blocked import bridge，不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。

必须保持：

- `notExecutedByAutomation=true`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isPostPublishProof=false`
- `isReleaseCloseProof=false`

## Owner 下一步

Owner 必须先补齐真实公开包 proof、仓库外 clean consumer proof、runtime proof、rollback review 和最终关闭决策，再让 strict validator 收敛。bridge 只负责暴露导入缺口，不能替代任何真实 proof。
