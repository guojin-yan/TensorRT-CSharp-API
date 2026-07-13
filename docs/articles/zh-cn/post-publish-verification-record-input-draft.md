# Post Publish Verification Record Input Draft

`post-publish-verification-record-input-draft` 是真实发布后 proof 回填的输入草稿生成器。它会把 clean consumer scan、owner 填写的 package/channel 信息和日志 SHA256 汇总到一份 draft 中，减少手工复制错误。

它不是 proof；默认必须保持 `inputDraftOnly=true`、`isPostPublishVerificationProof=false`、`canCloseReleaseIssue=false`。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationRecordTemplate.ps1

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishCleanConsumerProject.ps1 `
  -ProjectPath C:\release-proof\TensorRtSharpCleanConsumer\TensorRtSharpCleanConsumer.csproj

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationRecordInputDraft.ps1 `
  -CleanConsumerProjectScanPath artifacts\final-release\post-publish-clean-consumer-project-scan.json `
  -SelectedChannel nuget.org `
  -ChannelSourceUri https://api.nuget.org/v3/index.json `
  -ManagedPackageId JYPPX.TensorRtSharp `
  -ManagedPackageVersion 4.0.0-rc.1 `
  -RuntimePackageId JYPPX.TensorRtSharp.win-x64-trt11.0-cuda13.2-cudnn9.22 `
  -RuntimePackageVersion 4.0.0-rc.1 `
  -RestoreLogPath C:\release-proof\logs\restore.log `
  -NativeAssetListingPath C:\release-proof\logs\native-assets.log `
  -DependencyProbeLogPath C:\release-proof\logs\dependency-probe.log `
  -SmokeLogPath C:\release-proof\logs\smoke.log
```

默认输出：

- `artifacts/final-release/post-publish-verification-record.input-draft.json`
- `artifacts/final-release/post-publish-verification-record.input-draft.md`

## 草稿会回填什么

- `recordKind=post-publish-verification-record-input-draft`
- clean consumer root / project name / project path
- `cleanConsumerProjectScanPath`
- `cleanConsumerProjectScanPassed`
- `noProjectReference`
- selected channel 和 channel source URI
- managed/runtime package id、version、URL
- managed/runtime package SHA256 source
- restore/native asset listing/dependency probe/smoke log path
- restore/native asset listing/dependency probe/smoke log SHA256
- restore/build/smoke command

如果日志路径存在，脚本会直接计算 64 位 SHA256；如果文件不存在，对应 SHA256 为空，owner 必须在真实 proof 回填前补齐。

## 边界

input draft 只是 owner 填写真实 record 的中间产物。它不能改变：

- `smokeStatus=owner-action-required`
- `runtimeSmokePassed=false`
- `isPostPublishVerificationProof=false`
- `canCloseReleaseIssue=false`

真实 close readiness 只能来自填好的 `post-publish-verification-record.json`，并且必须通过：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 `
  -InputPath artifacts/final-release/post-publish-verification-record.json `
  -RequireExistingLog `
  -FailOnNotProof
```

## Owner 回填提醒

input draft 生成后，owner 仍需补齐：

- 真实 package URL 与下载时间
- managed/runtime nupkg SHA256
- host OS/GPU/driver/CUDA/TensorRT/cuDNN metadata
- stdoutSummary / stderrSummary；stderr 为空时写明 `no-stderr-emitted`
- runtime smoke exit code 和 `smokeStatus=passed`
- owner / reviewer
- published version

没有这些字段时，draft 不能升级为 proof。
