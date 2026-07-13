# Release Candidate Final Owner Checklist

`release-candidate-final-owner-checklist` 是最终发布前的一页式 Owner checklist。它把 public channel、manual publish、public package hash、clean consumer、runtime proof、Linux proof、real model proof、post-publish verification、non-substitute scan 和 final close approval 汇总为 10 项。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateFinalOwnerChecklist.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCandidateFinalOwnerChecklist.ps1 -Strict
```

## 产物

- `artifacts/final-release/release-candidate-final-owner-checklist.json`
- `artifacts/final-release/release-candidate-final-owner-checklist.md`
- `artifacts/final-release/release-candidate-final-owner-checklist-validation.json`
- `artifacts/final-release/release-candidate-final-owner-checklist-validation.md`

## 边界

默认状态为 `blocked-release-candidate-final-owner-checklist-owner-proof-required`，`Items=10` 且 `Blocked=10`。它只是 checklist，不发布包、不批准公开发布、不生成 runtime proof、post-publish proof 或 release close proof，也不关闭 release issue。

## Owner 下一步

Owner 必须逐项完成真实证据、运行 validator，并确认所有 non-substitute 边界仍然成立。全部真实 proof 通过前，不能公开发布、不能 close release issue。
