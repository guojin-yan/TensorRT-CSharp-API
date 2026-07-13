# Package Consumer Runtime Proof Worklist

`package-consumer-runtime-proof-worklist` 是 package-consumer-runtime proof 的 Owner 执行清单。它把以下四段状态合并到一个可读、可验证的 worklist：

- `package-consumer-runtime-proof-owner-input`
- `package-consumer-runtime-proof-candidate`
- `package-consumer-runtime-proof-record`
- `package-consumer-external-smoke-scaffold`

它不是 runtime proof，不执行发布，不关闭 release issue，也不能替代真实 clean external consumer smoke。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofWorklist.ps1
```

输出：

- `artifacts/final-release/package-consumer-runtime-proof-worklist.json`
- `artifacts/final-release/package-consumer-runtime-proof-worklist.md`

## 当前期望状态

在 owner 尚未提供真实外部 clean consumer smoke 前，该 worklist 必须保持：

- `worklistState=blocked-real-package-consumer-runtime-proof-required`
- `canPromoteRuntimeProof=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`

## Owner 需要补齐的真实材料

- 仓库外 clean consumer root 与 `.csproj`
- 无 `ProjectReference`
- 无 local feed 作为 public proof
- 无 direct `.nupkg` 引用
- public package source
- managed/runtime nupkg path 与 SHA256
- runtime package key
- compatible host metadata
- restore/build/dependency probe/runtime smoke 命令与日志
- smoke log SHA256
- stdout/stderr 摘要

这些材料全部满足后，仍必须通过：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofCandidate.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -FailOnNotProof
```

只有 strict proof record validator 晋级后，才能继续桥接到 `external-runtime-proof-record.json` 和 release close 链路。

Boundary keywords: not proof, not public package proof, not post-publish proof, not package push, not release close approval.
