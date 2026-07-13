# Release Candidate Final Publishability Audit

`release-candidate-final-publishability-audit` 是发布候选最终可发布性总检。它把 release evidence、classification audit、公开发布、clean consumer、runtime proof、post-publish、package/docs/sample readiness 和 release close gate 汇总为 17 条 publishability gate。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateFinalPublishabilityAudit.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCandidateFinalPublishabilityAudit.ps1 -Strict
```

## 产物

- `artifacts/final-release/release-candidate-final-publishability-audit.json`
- `artifacts/final-release/release-candidate-final-publishability-audit.md`
- `artifacts/final-release/release-candidate-final-publishability-audit-validation.json`
- `artifacts/final-release/release-candidate-final-publishability-audit-validation.md`

## 边界

默认状态为 `blocked-release-candidate-final-publishability-owner-proof-required`，`Gates=17` 且 `Blocked=17`。它只是最终可发布性审计，不发布包、不批准公开发布、不生成 runtime proof、post-publish proof 或 release close proof，也不关闭 release issue。

## Owner 下一步

Owner 必须补齐真实公开发布结果、公开包 hash、仓库外 clean consumer smoke、兼容主机 runtime proof、Linux proof、真实模型 proof、post-publish verification、rollback review 和 strict close record validation。
