# Final Evidence Freeze

`final-evidence-freeze` 是当前发布证据链的 SHA256 冻结记录。它把 release evidence bundle、post-publish validation、release-close candidate validation、Owner execution package validation 和 final close decision validation 的路径、状态与 SHA256 固化到同一份审计快照里。

它不是 release proof，不执行发布，不上传包，也不关闭 release issue。当前状态必须继续保持：

- `freezeState=blocked-evidence-frozen-owner-action-required`
- `validationState=final-evidence-freeze-valid-blocked-owner-action-required`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

## 生成与校验

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalEvidenceFreeze.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalEvidenceFreeze.ps1 -Strict
```

输出：

- `artifacts/final-release/final-evidence-freeze.json`
- `artifacts/final-release/final-evidence-freeze.md`
- `artifacts/final-release/final-evidence-freeze-validation.json`
- `artifacts/final-release/final-evidence-freeze-validation.md`

## 边界

Freeze 只能回答“当前证据链是否可审计、可复查、可哈希对齐”。它不能回答“是否已经可发布”或“是否可以关闭 release issue”。真实晋级仍要求 Owner 在公开包源上完成 clean external consumer restore/build/runtime smoke，回填 package SHA256、日志 SHA256、host metadata、rollback plan 和最终 close decision，并通过 strict validators。
