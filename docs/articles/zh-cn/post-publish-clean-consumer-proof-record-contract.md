# Post Publish Clean Consumer Proof Record Contract

`post-publish-clean-consumer-proof-record-contract` 是公开发布后的仓库外 clean consumer proof 记录合同。它要求 Owner 在真实公开包可下载后，用仓库外项目记录 restore、native asset listing、dependency probe、runtime smoke command、stdout/stderr 摘要和日志 SHA256，避免把本地包或项目引用误判为 post-publish proof。

当前合同已经扩展为最终 PostPublish clean consumer 回填面，必须覆盖：

- `cleanExternalConsumer.root` / `cleanExternalConsumer.projectPath`：仓库外 clean consumer 根目录和项目路径。
- `cleanExternalConsumer.restoreLogPath` / `cleanExternalConsumer.restoreLogSha256`：真实 restore 日志路径和 hash。
- `cleanExternalConsumer.buildLogPath` / `cleanExternalConsumer.buildLogSha256`：真实 build 日志路径和 hash。
- `cleanExternalConsumer.smokeLogPath` / `cleanExternalConsumer.smokeLogSha256`：真实 runtime smoke 日志路径和 hash。
- `cleanExternalConsumer.stdoutLogPath` / `cleanExternalConsumer.stdoutLogSha256`：stdout 日志路径和 hash。
- `cleanExternalConsumer.stderrLogPath` / `cleanExternalConsumer.stderrLogSha256`：stderr 日志路径和 hash；没有 stderr 时必须显式记录 `no-stderr-emitted`。
- `hostMetadata.osDescription` / `hostMetadata.architecture` / `hostMetadata.gpuName`：执行主机基础信息。
- `hostMetadata.cudaDriverVersion` / `hostMetadata.cudaRuntimeVersion` / `hostMetadata.cudnnVersion`：CUDA/cuDNN 运行环境信息。
- `hostMetadata.tensorRtVersion` / `hostMetadata.tensorRtLine`：TensorRT 版本和 TRT8/TRT10/TRT11 line。
- `ownerReview.reviewer` / `ownerReview.reviewedAtUtc` / `ownerReview.approvalState`：Owner 对 post-publish clean consumer proof 的复核记录。
- `forbiddenSubstituteCounts.projectReferenceCount`、`localFeedReferenceCount`、`directNupkgReferenceCount`、`buildOnlyCount`、`dependencyProbeOnlyCount`、`blockedByDriverOnlyCount`：禁止替代项扫描计数，预期均为 0。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishCleanConsumerProofRecordContract.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishCleanConsumerProofRecordContract.ps1 -Strict
```

## 产物

- `artifacts/final-release/post-publish-clean-consumer-proof-record-contract.json`
- `artifacts/final-release/post-publish-clean-consumer-proof-record-contract.md`
- `artifacts/final-release/post-publish-clean-consumer-proof-record-contract-validation.json`
- `artifacts/final-release/post-publish-clean-consumer-proof-record-contract-validation.md`

## 边界

该合同默认状态为 `blocked-post-publish-clean-consumer-proof-record-required`。它只定义 clean consumer proof record 的字段、禁用替代物和 validation 形状，不运行 sample、不下载公开包、不执行 smoke，也不批准 release close。

必须保持：

- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `isRuntimeExecutionProof=false`
- `isPostPublishProof=false`
- `isReleaseCloseProof=false`

## Owner 下一步

Owner 必须在仓库外 clean consumer 项目中使用真实公开渠道包执行 restore 与 runtime smoke，并回填日志、hash、host metadata、package identity 和 reviewer 字段。local feed、ProjectReference、direct `.nupkg`、build-only、template-only 或 scaffold-only 结果不能晋级为 post-publish proof。

如果 runtime smoke 被 CUDA driver、GPU 环境或依赖探测阻断，只能记录为 blocker/diagnostic，不能作为 post-publish proof 或 release close approval。
