# Owner Release Execution Package Validation

`owner-release-execution-package-validation` 校验 Owner 最短执行包的结构是否完整。它检查 `owner-release-execution-package.json` 是否包含一屏 Release Hold 清单、执行步骤、manual publish placeholder、required owner inputs、validator commands、source artifacts 和 non-substitute proof 边界。

它不是发布授权，也不是 runtime proof、post-publish proof 或 release close proof。当前模板状态下必须保持：

- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `packageState=blocked-real-proof-required`

## 生成与校验

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerReleaseExecutionPackage.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerReleaseExecutionPackage.ps1 -Strict
```

输出：

- `artifacts/final-release/owner-release-execution-package-validation.json`
- `artifacts/final-release/owner-release-execution-package-validation.md`

## 当前意义

当 validationState 为 `owner-execution-package-ready` 时，只说明 Owner 执行包已经足够清晰，可以作为执行指导交给 Owner。它仍不能替代真实公开包源、clean external consumer、package hash、runtime smoke log、Linux runner proof、real-model-runtime proof 或最终 release issue close record。

如果后续有人把 local feed、ProjectReference、direct `.nupkg`、dependency-probe-only、blocked-by-cuda-driver、template 或 build-only 当作 proof，validator 和 release evidence bundle 都必须继续阻断。
