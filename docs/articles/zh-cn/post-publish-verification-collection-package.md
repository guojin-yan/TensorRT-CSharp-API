# Post Publish Verification Collection Package

`post-publish-verification-collection-package` 是真实发布之后给 release owner 使用的 clean consumer proof 收集包。它提供从真实 channel 下载包、记录 URL / SHA256、创建仓库外 consumer、restore/build/smoke、回填 stdout/stderr summary、再执行 `-RequireExistingLog -FailOnNotProof` 的可复制流程。

它不执行 `dotnet nuget push`、GitHub Packages 上传、GitHub Release 上传、delete、delist 或 withdraw，也不能批准发布或关闭 release issue。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationCollectionPackage.ps1
```

输出：

- `artifacts/final-release/post-publish-verification-collection-package.json`
- `artifacts/final-release/post-publish-verification-collection-package.md`

默认边界必须保持：

- `recordKind=post-publish-verification-collection-package`
- `packageState=blocked-real-publication-required`
- `performsPublish=false`
- `approvesPublicRelease=false`
- `isPostPublishVerificationProof=false`
- `canCloseReleaseIssue=false`

## 执行顺序

collection package 的 `copyableExecutionOrder` 必须在 owner 已经授权并完成真实 channel 发布之后执行：

1. 复核 owner approval、owner decision 和 owner authorized publish command plan。
2. 复制 `post-publish-verification-record-template.json` 为真实 record。
3. 从真实 channel 下载 managed/runtime package，记录 package URL 与 SHA256。
4. 在源码仓库外创建 clean consumer，禁止 `ProjectReference`。
5. 从真实 channel restore/build 并保存日志。
6. 使用 `--runtime-package-key` 执行 package consumer runtime smoke。
7. 回填 stdout/stderr summary；无 stderr 时写 `no-stderr-emitted`。
8. 执行 `Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof`。
9. validator 晋级后刷新 release evidence、freeze summary 和 owner command plan。

## 证据边界

- collection package / backfill plan / template / draft / example / runbook 不是 post-publish proof。
- local feed、final package review、ProjectReference、dependency-probe-only 都不能关闭 release issue。
- build success 不是 runtime smoke proof。
- `blocked-by-cuda-driver` 不是 smoke passed。
- 只有真实 `post-publish-verification-record.json` 来自真实发布 channel、clean consumer、真实 logs 和 SHA256-backed validation，才能使 `canCloseReleaseIssue=true`。

## PostPublish Required Evidence 对齐

collection package 必须显式覆盖 owner command plan 中的 `postPublishRequiredEvidence`，包括：

- `selectedChannel`
- `channelSourceUri`
- `publishedPackageUrl`
- `managedPackageUrl`
- `runtimePackageUrl`
- `managedNupkgSha256`
- `runtimeNupkgSha256`
- `cleanConsumerRootOutsideRepository`
- `consumerProjectPath`
- `noProjectReference`
- `restoreLogPath`
- `nativeAssetListingSha256`
- `dependencyProbeLogPath`
- `dependencyProbeLogSha256`
- `runtimeSmokeLogPath`
- `runtimeSmokeLogSha256`
- `runtimeSmokePassed`
- `runtimeSmokeExitCode`
- `stdoutSummary`
- `stderrSummary`
- `hostMetadata`

## 不可替代 Proof 清单

以下内容必须继续视为“准备材料”或“阻塞状态”，不能关闭 release issue：

- collection package、backfill plan、template、draft、example、runbook。
- owner approval、owner decision、publish command plan；它们是授权链，不是发布后 consumer proof。
- local feed、本地 nupkg 输出、本地 inventory、ProjectReference consumer。
- dependency-probe-only、precheck、build-only、blocked-by-cuda-driver。
- 未保存 restore/native asset listing/dependency probe/smoke log 的记录。
- 未使用 `-RequireExistingLog` 校验 SHA256 的记录。

真实发布完成后，owner 应在源码仓库外创建 clean consumer，并执行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 `
  -InputPath artifacts/final-release/post-publish-verification-record.json `
  -RequireExistingLog `
  -FailOnNotProof
```

## 与聚合记录的关系

release evidence、promotion issue、freeze summary 和 freeze checklist 会展示本 collection package，帮助 owner 看清发布后 proof 回填步骤。但这些记录必须继续保持 `canCloseReleaseIssue=false`，直到真实 post-publish verification proof 与 owner 授权均齐全。
