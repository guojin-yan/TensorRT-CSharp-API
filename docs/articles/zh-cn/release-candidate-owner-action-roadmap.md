# Release Candidate Owner Action Roadmap

`release-candidate-owner-action-roadmap` 是最终发布前的 Owner 行动路线图。它把真实公开发布、公开包下载 hash、clean consumer、runtime proof、post-publish verification、rollback review 和 release issue close decision 排成 12 个可执行步骤。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateOwnerActionRoadmap.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCandidateOwnerActionRoadmap.ps1 -Strict
```

## 产物

- `artifacts/final-release/release-candidate-owner-action-roadmap.json`
- `artifacts/final-release/release-candidate-owner-action-roadmap.md`
- `artifacts/final-release/release-candidate-owner-action-roadmap-validation.json`
- `artifacts/final-release/release-candidate-owner-action-roadmap-validation.md`

## 边界

默认状态为 `blocked-release-candidate-owner-action-roadmap-owner-proof-required`，`Actions=12` 且 `Blocked=12`。它只是 Owner guidance，不发布包、不批准公开发布、不生成 runtime proof、post-publish proof 或 release close proof，也不关闭 release issue。

## Owner 下一步

Owner 按路线图逐步补齐真实证据。每一步必须带真实文件、日志、hash、环境和 validator 输出；dry-run、runbook、local feed、ProjectReference 和 dashboard 都不能替代 proof。
