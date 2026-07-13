# Post Publish Proof Owner Confirmation

`post-publish-proof-owner-confirmation` 将 public package proof、post-publish owner input、post-publish record validator、Owner 外部执行结果导入和 release close owner bridge 聚合为一个 Owner-facing confirmation gate。

## 覆盖范围

- 聚合 5 个 post-publish proof confirmation gate。
- 默认所有 gate blocked，`confirmationState=blocked-post-publish-proof-owner-confirmation-required`。
- 每个 gate 都必须达到自己的 required state，才能进入 public proof close bridge。
- 聚合结果只说明哪些真实 proof 输入仍缺失，不会晋级 proof。

## 命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishProofOwnerConfirmation.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishProofOwnerConfirmation.ps1 -Strict
```

## 产物

- `artifacts/final-release/post-publish-proof-owner-confirmation.json`
- `artifacts/final-release/post-publish-proof-owner-confirmation.md`
- `artifacts/final-release/post-publish-proof-owner-confirmation-validation.json`
- `artifacts/final-release/post-publish-proof-owner-confirmation-validation.md`

## 边界

该 confirmation 是 blocked gate aggregation，不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。它不能把 owner input、validator contract、hash slot、local package 或 dry-run 结果当成真实发布后 proof。
