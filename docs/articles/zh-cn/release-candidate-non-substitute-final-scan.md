# Release Candidate Non Substitute Final Scan

`release-candidate-non-substitute-final-scan` 是最终发布前的禁止替代物扫描。它列出 14 类不能晋级为 proof 的材料，包括 local `.nupkg`、local feed、ProjectReference、direct nupkg、template、draft、dry-run、runbook、dashboard、audit pack、hash-only lane、candidate、local-only scan 和 manual handoff。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateNonSubstituteFinalScan.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCandidateNonSubstituteFinalScan.ps1 -Strict
```

## 产物

- `artifacts/final-release/release-candidate-non-substitute-final-scan.json`
- `artifacts/final-release/release-candidate-non-substitute-final-scan.md`
- `artifacts/final-release/release-candidate-non-substitute-final-scan-validation.json`
- `artifacts/final-release/release-candidate-non-substitute-final-scan-validation.md`

## 边界

默认状态为 `blocked-release-candidate-non-substitute-final-scan-owner-proof-required`，`Checks=14`、`Blocked=14` 且 `Promoted=0`。它只是禁止替代物审计，不发布包、不批准公开发布、不生成 runtime proof、post-publish proof 或 release close proof，也不关闭 release issue。

## Owner 下一步

Owner 必须用真实公开渠道、真实外部 consumer、真实 runtime/post-publish 记录替换所有替代物。替代物可以帮助定位缺口，但不能作为完成证据。
