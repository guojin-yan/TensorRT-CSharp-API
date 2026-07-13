# Final Post-Publish Audit Pack

`final-post-publish-audit-pack` 是公开发布后最终审计准备包。它把 public package proof、post-publish verification、post-publish owner confirmation、package-consumer runtime proof、release close public proof bridge、final owner decision audit 和 strict release close validator 收束成 7 条 audit lane。

该包不是 post-publish proof，也不是 release close approval。它只用于把最终发布后审计仍缺失的真实证据列清楚，避免把本地包、local feed、ProjectReference、模板、dry-run、runbook、hash slot、candidate record 或 validator contract 误判为真实 proof。

## 产物

- `artifacts/final-release/final-post-publish-audit-pack.json`
- `artifacts/final-release/final-post-publish-audit-pack.md`
- `artifacts/final-release/final-post-publish-audit-pack-validation.json`
- `artifacts/final-release/final-post-publish-audit-pack-validation.md`

## 当前边界

- `performsPublish=false`
- `canPromoteRuntimeProof=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isReleaseCloseProof=false`
- `isPostPublishProof=false`

## Audit Lane

| Lane | 真实完成条件 |
|---|---|
| `public-package-proof` | 公开包来源、registry URL、package URL、nupkg SHA256 和 Owner review 字段齐备。 |
| `post-publish-verification-proof` | 真实 public-channel clean consumer restore/build/smoke log、SHA256 和 host metadata 齐备。 |
| `post-publish-proof-owner-confirmation` | Owner confirmation 所有 gate ready。 |
| `package-consumer-runtime-proof` | 仓库外 clean external package consumer runtime smoke proof ready。 |
| `release-close-public-proof-bridge` | public proof bridge 所有 gate ready。 |
| `final-owner-decision-audit` | final owner decision audit 所有 gate ready。 |
| `strict-release-close-validator` | `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady` 通过。 |

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalPostPublishAuditPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalPostPublishAuditPack.ps1 -Strict
```

当前预期状态仍是 blocked，直到真实公开包发布后的 clean consumer proof、post-publish verification 和 strict close record 全部由 Owner 填入并通过 validator。
