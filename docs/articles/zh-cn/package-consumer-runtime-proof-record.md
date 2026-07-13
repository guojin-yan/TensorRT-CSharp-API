# Package Consumer Runtime Proof Record

`package-consumer-runtime-proof-record` 用于把 Owner 回填的 clean external consumer smoke evidence，整理成 strict runtime proof record。它对齐现有 `external-runtime-proof-record` schema，并提供 bridge exporter 生成 `artifacts/final-release/external-runtime-proof-record.json`。

该记录不是发布动作，不推送 package，不关闭 release issue。只有在真实 public package source、repo 外 clean consumer、无 ProjectReference/local feed/direct `.nupkg`、compatible CUDA/TensorRT host、runtime-key smoke command、matching smoke log SHA256 和 reviewed stdout/stderr summary 全部通过后，才能让 `canPromoteRuntimeProof=true`。

## 产物

- `artifacts/final-release/package-consumer-runtime-proof-record.template.json`
- `artifacts/final-release/package-consumer-runtime-proof-record.template.md`
- `artifacts/final-release/package-consumer-runtime-proof-record.json`
- `artifacts/final-release/package-consumer-runtime-proof-record.md`
- `artifacts/final-release/package-consumer-runtime-proof-record-validation.json`
- `artifacts/final-release/package-consumer-runtime-proof-record-validation.md`
- `artifacts/final-release/external-runtime-proof-record.json`

## 命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofRecordTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -InputPath artifacts/final-release/package-consumer-runtime-proof-record.template.json -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofRecordFromOwnerInput.ps1 -OwnerInputPath artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -InputPath artifacts/final-release/package-consumer-runtime-proof-record.json -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofRecordFromPackageConsumerProof.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json
```

## 边界

- `template-only`、owner input、candidate、local feed、ProjectReference、direct `.nupkg` 不是 proof。
- `Test-PackageConsumerRuntimeProofRecord.ps1 -Strict` 在 template 状态下 blocker=0，但 proof-required/action-required 仍会阻塞晋级。
- release evidence bundle 只能把该记录作为 evidence item 聚合，不能单独授权发布或关闭 release issue。

Boundary keywords: not proof, not public package proof, not post-publish proof, not package push, not release close approval.
