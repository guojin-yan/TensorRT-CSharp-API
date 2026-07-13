# 完整项目发布前最后一公里

TensorRtSharp4.0 已经从“接口追平”进入“发布候选收口”。最后一公里不是继续堆接口数量，而是把 API、deferred 边界、样例、文档、runtime package、release owner approval 和真实 package-consumer proof 串成一条可复核链路。

最终包审阅先运行 `Export-FinalPackageReviewBundle.ps1`，生成 `artifacts/final-release/final-package-review-bundle.json` 和 `.md`，用于核对 managed/runtime/split-runtime `.nupkg` 的 package id/version、大小、SHA256、runtime package key 和 native asset count。该步骤不发布包，也不是 public channel proof。

本文是一份发布前执行清单。它不会替代 release owner 决策，也不会把 `blocked-by-cuda-driver` 改成通过；它帮助你确认还差什么，哪些能自动验证，哪些必须由 owner 在兼容主机上补齐。

## 1. 代码与接口状态

先确认基础工程仍可构建：

```powershell
dotnet build .\TensorRtSharp.sln -c Debug --no-restore /p:UseSharedCompilation=false /nr:false
```

如果修改 manifest、native 或生成代码，再追加：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
```

`manifest/source 100%` 不等于全部 public API 可用。真实完成度仍要看非 deferred 实现、高层 C# wrapper、smoke 和 package-consumer 验证。

## 2. 样例与应用入口

发布前至少确认这些入口文档是可发现的：

- `samples/README.md`
- `samples/OnnxToEngine/README.md`
- `samples/Classification/README.md`
- `samples/YoloVision/README.md`
- `applications/README.md`
- `applications/TensorRtExec/README.md`

样例证据必须分层：

- `OnnxToEngine` 证明最小 identity ONNX round-trip。
- `TensorRtExec` 生成 external ONNX build/precheck report。
- `Classification` 和 `YoloVision` 需要用户或 release owner 自备真实资产。
- `sample-run-evidence` 最多晋级 sample-level `real-model-runtime`。
- `package-consumer-runtime` belongs to release proof records。

## 3. 文档封版

文档封版要检查：

```powershell
rg -n "sample-evidence-ladder|onnxtoengine-and-tensorrtexec-boundary|tool-report-to-release-proof-record|stale-claim-prepublish-audit|publish-final-mile-checklist" .\docs\index.md .\docs\toc.yml .\docs\articles\zh-cn\technical-article-roadmap.md
```

如环境支持，运行 DocFX：

```powershell
dotnet docfx .\docs\docfx.json
```

如果 docfx 环境不可用，至少用 ProjectQuality 静态测试保证新增文章存在、入口链接存在、proof 边界不被误写。

## 4. Release readiness

运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCandidateReadiness.ps1
```

当前缺少真实 `external-runtime-proof-record.json` 时，它应返回 blocker。这是正确行为。不要把 `Test-ReleaseCandidateReadiness.ps1` 改成无条件 green，也不要把 `-WarnOnly` 的 dry-run 结果写成 release gate 通过。

dry-run 审阅可以使用：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalReleaseDryRun.ps1 -AllowRuntimeSmokeBlocked -WarnOnly
```

`ready-needs-manual-approval` 不是 public release approval。它只说明材料可进入 owner 审阅，还需要真实 owner 输入和 release proof。

## 5. External runtime proof

真实 release gate 需要兼容 CUDA host 上的 package consumer runtime proof。记录文件应是：

```text
artifacts/final-release/external-runtime-proof-record.json
```

它至少应证明：

- 使用包源消费，不使用 project reference。
- managed nupkg 和 runtime nupkg SHA256 对齐。
- native asset copy 成功。
- smoke log 存在，SHA256 匹配。
- `results.smokeStatus=passed`。
- `proofClassification=package-consumer-runtime`。
- validator 严格模式通过。

`blocked-by-cuda-driver` is not smoke passed。当前 host 若仍被 CUDA driver/runtime mismatch 阻塞，应保留 blocker，并在兼容 host 上执行 proof 回填。

## 6. Owner approval

发布 owner 输入不应由脚本伪造。模板、example 和 draft 都不是 approval record。

检查：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseOwnerApprovalInputTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseOwnerApprovalInput.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseOwnerDecisionRecord.ps1
```

只有非模板、显式 owner、必需决策齐全且 proof 状态一致时，才可以进入真实 public promotion 讨论。否则保持 pending 或 blocked。

## 7. Publish execution

没有用户明确授权时，不发布 NuGet、GitHub Packages、GitHub Release 或 private feed。发布执行清单可以生成：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleasePublishExecutionChecklist.ps1
```

清单是执行材料，不是执行动作。它不会调用 `dotnet nuget push`，也不会批准 public release。

## 8. Post-publish verification

真实发布后还需要 post-publish verification：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationRecordTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof
```

post-publish record 只有在真实渠道发布后才有意义。发布前模板只能作为 owner guidance，不能写成已经验证。

## 最后一公里判断

可以进入发布审阅的状态：

- build 和 ProjectQuality tests 通过。
- stale claim audit 通过。
- docs/index/toc/roadmap 同步。
- release readiness blocker 解释清楚。
- owner approval input 已准备。
- external runtime proof 若缺失，则明确记录为 blocker。

不能进入 public publish 的状态：

- `blocked-by-cuda-driver` 被写成 passed。
- build-only 被写成 inference proof。
- collection bundle 被写成 runtime execution evidence。
- `ready-needs-manual-approval` 被写成 approved。
- 没有真实 `external-runtime-proof-record.json` 却声明 `package-consumer-runtime` 完成。

## 小结

完整项目发布前最后一公里是证据治理，不是文字包装。项目可以带着明确的 known limitation 进入 owner 审阅，但不能带着过度声明进入 public release。只要 release proof record、owner approval 和 post-publish verification 的边界保持清晰，TensorRtSharp4.0 就能以可信方式走向完整发布。
