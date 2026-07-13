# Post Publish Clean Consumer Project Scan

`post-publish-clean-consumer-project-scan` 是发布后验证链路里的 clean consumer 项目扫描器。它只检查消费端项目形态，不执行 restore/build/smoke，也不能替代真实 `post-publish-verification-record.json`。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishCleanConsumerProject.ps1 `
  -ProjectPath C:\release-proof\TensorRtSharpCleanConsumer\TensorRtSharpCleanConsumer.csproj
```

默认输出：

- `artifacts/final-release/post-publish-clean-consumer-project-scan.json`
- `artifacts/final-release/post-publish-clean-consumer-project-scan.md`

## 检查内容

扫描器会读取 `.csproj` 并确认：

- project 文件存在，且扩展名是 `.csproj`
- project 位于源码仓库之外
- 存在 `PackageReference`
- 存在目标 TensorRtSharp package reference
- 不存在 `ProjectReference`
- XML 可解析

典型状态：

- `clean-consumer-project-scan-passed`
- `blocked-project-reference-present`
- `blocked-target-package-reference-missing`
- `blocked-project-inside-repository`
- `missing-project`

## 边界

该扫描只证明 consumer project 形态符合发布后验证要求。它始终保持：

- `canCloseReleaseIssue=false`
- `isPostPublishVerificationProof=false`
- `performsPublish=false`

因此，scan passed 只能作为 input draft 和 post-publish verification 的前置辅助证据，不能单独关闭 release issue。

## 推荐使用顺序

1. release owner 完成真实发布后，在源码仓库之外创建全新 consumer。
2. consumer 只引用发布 channel 上的 managed/runtime package，不引用源码项目。
3. 运行 `Test-PostPublishCleanConsumerProject.ps1`。
4. 将扫描输出传给 `Export-PostPublishVerificationRecordInputDraft.ps1`。
5. 在 clean consumer 上继续执行 restore/build/native asset listing/dependency probe/runtime smoke。
6. 回填真实 `post-publish-verification-record.json` 并运行 `Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof`。

## 不能误读

- scan passed 不是 package runtime smoke passed。
- scan passed 不是 published package proof。
- scan passed 不是 release close approval。
- `ProjectReference` 出现时必须阻断发布后 proof。
- 仓库内 consumer 不能作为 clean consumer。
