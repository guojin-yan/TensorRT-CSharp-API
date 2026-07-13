# Public Package Proof Owner Input

`public-package-proof-owner-input` 是真实公开包发布之后由 Owner 回填 managed/runtime `.nupkg` 公开渠道证据的输入合同。它收集公开包 URL、NuGet package source、GitHub Release asset、包版本、SHA256、仓库外 clean external consumer restore/build/smoke 日志、stdout/stderr hash、host metadata、发布时间和人工复核字段，不会执行发布命令。

## 覆盖范围

- managed package 与 runtime package 的公开源、registry、package URL、`.nupkg` 路径和 SHA256。
- NuGet package source、GitHub Release URL/tag、managed/runtime release asset path/hash。
- 仓库外 clean external consumer 的 restore/build/smoke command、日志路径和 SHA256。
- stdout/stderr 日志路径和 SHA256；如果 stderr 没有输出，也要用显式 owner 复核标记记录。
- host metadata：Owner、机器、OS、架构、GPU、CUDA/cuDNN/TensorRT 版本线。
- Owner 对非 local feed、非 ProjectReference、非 direct `.nupkg` 的确认。
- 默认 `templateState=blocked-public-package-proof-owner-input-required`。
- 默认 validation 仍为 blocked，直到 Owner 提供真实公开包字段。

## 命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PublicPackageProofOwnerInputTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PublicPackageProofOwnerInput.ps1 -Strict
```

## 产物

- `artifacts/final-release/public-package-proof-owner-input.template.json`
- `artifacts/final-release/public-package-proof-owner-input.template.md`
- `artifacts/final-release/public-package-proof-owner-input-validation.json`
- `artifacts/final-release/public-package-proof-owner-input-validation.md`

## 边界

该模板和 validator 都不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。它们会显式拒绝 local feed、ProjectReference、direct `.nupkg`、仓库内 consumer、build-only 和 dry-run 作为公开包 proof 替代项。只有 Owner 在真实公开包发布后填入可审计字段，并通过后续 post-publish proof/close validator，才能进入 release close 证明链。
