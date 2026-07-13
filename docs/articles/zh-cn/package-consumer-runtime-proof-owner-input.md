# Package Consumer Runtime Proof Owner Input

`package-consumer-runtime-proof-owner-input` 是 package-consumer runtime proof candidate 的 owner 回填模板与验证面。它把 clean external consumer root、public package source、managed/runtime nupkg 路径与 SHA256、compatible host metadata、`--runtime-package-key` smoke command、smoke log hash 和 stdout/stderr summary 固定为可审计字段。

该 surface 只用于 candidate overlay，不是 proof，不执行发布，不关闭 release issue，也不能把 local feed、ProjectReference 或 direct `.nupkg` 当作 public package proof。

## 产物

- `artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json`
- `artifacts/final-release/package-consumer-runtime-proof-owner-input.template.md`
- `artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json`
- `artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.md`

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofCandidate.ps1 -OwnerInputPath artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofCandidate.ps1 -Strict
```

## 边界

- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `canPromoteProof=false`
- strict 模式只因 blocker 失败而失败；无真实 owner 输入时保持 `blocked-owner-input-required` 与 action-required。

Boundary keywords: not proof, not public package proof, not post-publish proof, not package push, not release close approval.
