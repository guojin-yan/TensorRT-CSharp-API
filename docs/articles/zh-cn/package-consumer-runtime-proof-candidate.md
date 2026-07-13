# Package Consumer Runtime Proof Candidate

`package-consumer-runtime-proof-candidate` 是 `package-consumer-runtime` proof line 的 owner input surface，用于把 clean external consumer、public package source、managed/runtime nupkg SHA256、compatible host metadata、runtime-key smoke command 和 smoke log SHA256 集中到一个候选记录中。

它不是 proof，不执行发布，不批准公开发布，也不关闭 release issue。只有真实外部 consumer、真实公开 package source、匹配的 package/hash、兼容 CUDA/TensorRT host metadata、包含 `--runtime-package-key` 的 smoke command、真实 smoke log 与 strict validator 全部通过后，才可能进入 proof review。

## 输出

- `artifacts/final-release/package-consumer-runtime-proof-candidate.json`
- `artifacts/final-release/package-consumer-runtime-proof-candidate.md`
- `artifacts/final-release/package-consumer-runtime-proof-candidate-validation.json`
- `artifacts/final-release/package-consumer-runtime-proof-candidate-validation.md`

## 默认状态

- `recordKind=package-consumer-runtime-proof-candidate`
- `candidateState=blocked-real-package-consumer-smoke-required`
- `canPromoteProof=false`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

## 必需真实输入

- clean external consumer root 必须位于仓库外。
- 不允许 `ProjectReference` 作为 public proof。
- 不允许 local feed 作为 public proof。
- 不允许 direct `.nupkg` 引用作为 public proof。
- managed/runtime `.nupkg` 必须有 SHA256。
- smoke command 必须包含 `--runtime-package-key`。
- smoke log path 与 smoke log SHA256 必须可验证。
- compatible host metadata 必须包括 OS、architecture、CUDA driver/runtime 和 TensorRT version。

## 边界

该 candidate 只描述 owner 需要补齐的输入，不会自行采集 proof。local feed、ProjectReference、direct `.nupkg`、dependency-probe-only、blocked-by-cuda-driver、template/draft/preflight-only 输出均不能替代真实 package consumer runtime proof。
