# Public Publish Forbidden Substitute Scan

`public-publish-forbidden-substitute-scan` 是公开发布后 proof 回填链路的替代物扫描。它把 local `.nupkg`、local feed、ProjectReference、direct nupkg、dry-run、template、runbook、dashboard 和 candidate 等材料统一标记为不可替代真实公开发布 proof。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PublicPublishForbiddenSubstituteScan.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PublicPublishForbiddenSubstituteScan.ps1 -Strict
```

## 产物

- `artifacts/final-release/public-publish-forbidden-substitute-scan.json`
- `artifacts/final-release/public-publish-forbidden-substitute-scan.md`
- `artifacts/final-release/public-publish-forbidden-substitute-scan-validation.json`
- `artifacts/final-release/public-publish-forbidden-substitute-scan-validation.md`

## 边界

默认状态为 `blocked-public-publish-forbidden-substitute-scan-owner-proof-required`，`Checks=9` 且 `Blocked=9`。它只做 substitute 风险扫描，不发布包、不生成 proof、不批准公开发布，也不关闭 release issue。

必须保持：

- `notExecutedByAutomation=true`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isPostPublishProof=false`
- `isReleaseCloseProof=false`

## Owner 下一步

Owner 必须用真实公开渠道发布结果、公开包下载 hash、仓库外 clean consumer smoke 和 strict close validator 替换所有 substitute。只要任一 substitute 仍被引用为 proof，release gate 必须保持 blocked。
