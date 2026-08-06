# Release Candidate Final Hold 与 Owner 等待状态

本文用于固定 TensorRtSharp4.0 当前发布候选的最终等待状态。它不是新的 proof record，也不是发布完成声明，而是给 owner、维护者和后续大模型执行者看的边界说明：项目已经完成 release candidate 前台收口，但公开发布仍被真实外部 proof 阻塞。

## 当前结论

当前发布状态必须继续保持：

```text
FreezeState=blocked-real-proof-required
PerformsPublish=False
CanPublishPublicly=False
CanCloseReleaseIssue=False
```

这意味着：

- 可以继续维护源码、文档、样例、quality tests、release runbook 和 proof schema。
- 可以把项目描述为 release candidate 或 final hold。
- 不能写成已经公开发布、已经可关闭 release issue、已经通过 post-publish verification。
- 不能用本地构建、模板工程、ProjectReference consumer、local feed、runbook、sidecar、parse-only report 或 build-only report 替代真实 proof。

## 等待 Owner 的五个真实动作

| 阻塞项 | 当前状态 | Owner 需要提供或执行 | 可接受证据 |
|---|---:|---|---|
| owner authorization | 未完成 | 明确授权 package ID、版本、feed、发布时间和发布范围 | 授权记录、时间、授权人、目标 feed |
| package-consumer-runtime | 未完成 | 在 clean consumer 中安装真实 nupkg 或真实 feed 包并运行 smoke | `external-runtime-proof-record.json` 和 validator log |
| Linux runner proof | 未完成 | 在真实 Linux x64 runner 上执行 build/test 或 runtime proof | OS、driver、CUDA、TensorRT、commit、commands、logs |
| real-model-runtime | 未完成 | 使用真实模型资产运行 Classification 或 YoloVision 等样例 | 模型来源、license、SHA256、input、labels、sample run log |
| post-publish verification | 未完成 | 真实发布后从目标 feed 拉取并验证 | post-publish verification record 和 validator log |

## 不可替代清单

最终关闭 release issue 还必须通过 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`。当前 `release-issue-close-record-validation=blocked-template-only`，`release-issue-close-record-template.json` 只是字段模板；缺真实 post-publish proof、release close preflight、stale claim audit、evidence bundle SHA256、rollback plan 或 owner final close decision 时，`canCloseReleaseIssue=false` 必须保持。

以下材料可以作为准备工作、说明文档或辅助诊断，但不能写成 release proof：

- `dotnet build`、DocFX build、project quality tests。
- `samples` 或 `applications` 的 dry-run、parse-only、build-only 输出。
- `applications/TensorRtExec` 的 ONNX build/precheck report。
- 本地 feed、draft package、collection bundle、input package、helper project。
- ProjectReference consumer 或同仓库 consumer。
- `blocked-by-cuda-driver`、`blocked-by-application-control` 等环境阻塞说明。
- 手写 runbook、模板 JSON、sidecar-only 文件。

这些材料仍然有价值，但它们只能证明“准备状态”或“构建/解析状态”，不能证明 package-consumer-runtime、real-model-runtime、Linux runner 或 post-publish verification 已完成。

## 允许继续做的工作

在 owner 尚未提供真实外部条件前，可以继续做以下工作：

1. 维护 README、README.zh-CN、docs index、toc 和 technical article roadmap 的一致性。
2. 复跑 `eng/Test-StaleReleaseClaims.ps1`，确保没有 stale release claim。
3. 扩展 quality tests，防止发布候选文档误写成正式发布。
4. 完善样例说明、资产 manifest、license checklist 和 proof schema。
5. 准备真实 proof 回填脚本和 owner action checklist。

不应继续做的工作：

1. 新增看起来像 proof、但没有真实外部执行日志的 wrapper。
2. 把 build-only、parse-only、dry-run、sidecar-only 的结果升级成 runtime proof。
3. 删除 blocker 或质量门禁来制造 release complete。
4. 执行 `dotnet nuget push`、GitHub Packages 上传或 GitHub Release 上传。

## 真实 Proof 到来后的执行顺序

当 owner 提供真实条件后，建议顺序如下：

1. 回填 owner authorization，明确目标 package、版本、feed 和授权时间。
2. 执行 package-consumer-runtime，并运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof
```

3. 回填 real-model-runtime，优先选择 `samples\ComputerVision\01.Classification` 或 `applications\YoloVision`，并运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleAssetManifest.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog
```

4. 在真实 Linux x64 runner 上回填 Linux proof，并运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidenceRecord.ps1
```

5. 真实发布完成后，执行 post-publish verification，并运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof
```

## 对外表述边界

推荐表述：

- “项目处于 release candidate final hold。”
- “发布候选文档、样例说明、quality gates 和 proof schema 已收口。”
- “公开发布仍等待 owner authorization、package-consumer-runtime、Linux runner proof、real-model-runtime 和 post-publish verification。”

禁止表述：

- “已经公开发布。”
- “NuGet 包已经可用。”
- “release issue 可以关闭。”
- “post-publish verification 已完成。”
- “blocked-by-cuda-driver 等同于 smoke passed。”

## 维护者检查清单

每次继续维护 final hold 状态时，至少检查：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StaleReleaseClaims.ps1
dotnet build .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-restore /m:1 /p:UseSharedCompilation=false /nr:false
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~TechnicalArticleRoadmap|FullyQualifiedName~ReleaseCandidateFinalEvidenceFreeze|FullyQualifiedName~CompatibleHostProofExecutionPack|FullyQualifiedName~ReleaseCloseGapDashboard"
```

如果新增 release-facing 文章，应同步更新：

- `README.md`
- `README.zh-CN.md`
- `docs/index.md`
- `docs/toc.yml`
- `docs/articles/zh-cn/technical-article-roadmap.md`
- `tests/JYPPX.ProjectQuality.Tests/TechnicalArticleRoadmapTests.cs`
