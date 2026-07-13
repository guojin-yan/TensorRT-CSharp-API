# Post Publish Verification Proof Playbook

这篇文章说明真实渠道发布动作完成后，如何采集 `post-publish verification proof`。它不执行发布，也不授权发布；发布命令必须由 release owner 在外部审批后手工执行。

## Proof 定义

`post-publish verification proof` 只能来自真实渠道上的包：验证人从 NuGet.org、GitHub Packages、private feed 或 owner 指定渠道下载包，在干净 consumer 中 restore/build/smoke，并回填 package identity、URL、downloaded nupkg SHA256、clean consumer identity、host metadata、命令、stdout/stderr summary 和日志 hash。

以下内容不是 proof：

- post-publish template
- input draft
- backfill plan
- collection package
- clean consumer scan 本身
- local feed
- ProjectReference
- build-only report
- `blocked-by-cuda-driver`

## Owner 执行顺序

1. owner 完成真实渠道发布动作，并保留审批记录、渠道 URL、package id/version。
2. 在仓库外创建干净 consumer。
3. 从真实渠道 restore managed/runtime packages。
4. 下载或定位实际 nupkg 文件，计算 SHA256。
5. 运行 clean consumer restore/build/native asset listing/dependency probe/runtime smoke。
6. 记录 stdout/stderr summary，保留日志文件并计算 SHA256。
7. 先运行 clean consumer scan：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishCleanConsumerProject.ps1 `
  -ProjectPath <clean-consumer.csproj> `
  -OutputRoot .\artifacts\final-release
```

8. 使用 input draft 降低漏填风险，但不要把 draft 当 proof：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationRecordInputDraft.ps1
```

9. 回填真实 `post-publish-verification-record.json` 后验证：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 `
  -InputPath .\artifacts\final-release\post-publish-verification-record.json `
  -RequireExistingLog `
  -FailOnNotProof
```

10. 刷新 release evidence bundle 和 release close preflight。

## 字段复核表

| 字段 | 复核要求 |
| --- | --- |
| package identity | managed/runtime package id、version、channel URL |
| downloaded hashes | managed/runtime nupkg SHA256 均为真实下载文件 hash |
| clean consumer | 不在仓库内、无 ProjectReference、只从 package source restore |
| runtime key | smoke command 中包含目标 `--runtime-package-key` |
| host metadata | CUDA/TensorRT/cuDNN/OS/arch 完整 |
| log hashes | restore/native listing/dependency probe/smoke log 均可被 `-RequireExistingLog` 复核 |
| stdout/stderr | 有人工可读 summary，不能只写“见日志” |

## Release close 边界

post publish verification 能证明真实渠道发布后的 clean consumer 路径，但它仍需要 release close preflight 汇总 owner authorization、package-consumer-runtime proof、real-model-runtime proof 和其它 blockers。clean consumer scan、input draft、local feed、ProjectReference 或 `blocked-by-cuda-driver` 都不能单独关闭 release issue。

可以写：当前 post publish verification 需要 owner 在真实渠道发布动作后回填。不要写：template 或 draft 已经完成 post publish verification，或 local feed consumer 可以替代真实渠道验证。
