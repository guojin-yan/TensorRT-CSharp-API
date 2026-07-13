# 发布候选 Owner 一屏执行包

`artifacts/final-release/release-candidate-owner-one-screen-execution-pack.json` 是给 Owner 使用的一屏执行入口。它把真实模型运行、仓库外包消费、公开发布后验证和最终 release issue close 四条 lane 放到一个页面里，方便按顺序执行和回填。

这份执行包不是 proof，不会执行模型，不会发布包，也不会关闭 release issue。当前仍然必须保持：

- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `blocked-owner-action-required`

## 执行顺序

| 顺序 | Lane | 要填的记录 | Validator |
| --- | --- | --- | --- |
| 1 | real-model-runtime | `real-case-evidence-record.json` | `eng/Test-RealCaseEvidenceRecord.ps1 -FailOnNotProof` |
| 2 | package-consumer-runtime | `package-consumer-runtime-proof-record.json` | `eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof` |
| 3 | post-publish-verification | `post-publish-verification-record.json` | `eng/Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof` |
| 4 | release-issue-close | `release-issue-close-record.json` | `eng/Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady` |

## 每条 Lane 都需要什么

每条 lane 都列出：

- expected record path
- template/schema path
- command to run
- validator command
- required logs
- required SHA256 fields
- required host metadata
- blocker dashboard link
- forbidden substitutes
- current state
- `canPromote=false`

## 最终审计边界

以下说法在当前状态下都必须被阻止：

- 当前已经可以公开发布。
- 当前可以关闭 release issue。
- template/report/matrix/article/dry-run/build-only 都不能作为真实证据。
- `blocked-by-cuda-driver` 或 `Skipped=True` 可以当作 passed。
- local feed / ProjectReference / direct `.nupkg` 不能替代 clean external consumer 运行记录。

正确口径仍是：真实 proof 缺失，发布候选未冻结为可公开发布版本，Owner 需要继续回填真实外部执行证据并通过严格 validator。
