# 最终发布 Dry Run

最终发布 dry run 用来回答一个发布负责人真正关心的问题：当前包能不能进入发布流程，哪些事项已经由自动证据证明，哪些仍需要人工确认或外部环境验证。

它不会发布包，也不会修改远端 package source。它只读取当前 workspace 中已经生成的证据，汇总 managed package、runtime package、package consumer、local feed consumer、release candidate readiness、release checklist、DocFX、双语 API 文档审计、签名状态、Linux handoff、CUDA runtime smoke 和真实 callback proof。

## 命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PublicApiBilingualDocumentation.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PublicApiBilingualDocumentationBacklog.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateChecklist.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalReleaseDryRun.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -AllowRuntimeSmokeBlocked -WarnOnly
```

输出文件：

- `artifacts/final-release/final-release-dry-run-summary.json`
- `artifacts/final-release/final-release-dry-run-summary.md`
- `artifacts/api-doc-audit/public-api-bilingual-documentation-backlog.json`
- `artifacts/api-doc-audit/public-api-bilingual-documentation-backlog.md`

## 状态解释

`ready` 表示没有 blocker、manual approval 或 warning。当前阶段更常见的是：

- `ready-needs-manual-approval`：自动证据链没有阻塞项，但仍有签名、Linux runner、CUDA driver/runtime 或 release owner 决策需要确认。
- `ready-with-warnings`：没有人工审批项，但存在非阻塞 warning。
- `blocked`：存在自动证据无法通过的发布阻塞项。

当前 `win-x64-trt11.0-cuda13.2-cudnn9.22` dry run 的核心边界是：

- package/readiness 证据可用，local feed consumer 可从包源 restore/build/copy native assets。
- full package runtime smoke 在当前机器上仍为 `blocked-by-cuda-driver`。
- package consumer evidence schema 当前记录为 `packageConsumerEvidenceKind=full-runtime-package-consumer-smoke-driver-blocked`、`runtimeSmokeClassification=runtime-smoke-driver-blocked`、`isRuntimeExecutionEvidence=false`、`isDependencyProbeOnly=true`、`isRealCallbackRuntimeProof=false`。
- final dry run 现在同时输出 `runtimeProofStatus=blocked-by-cuda-driver` 和 `runtimeProofRequiredForRelease=true`；这两个字段才是发布运行证明是否补齐的机器可读判断。
- `allowRuntimeSmokeBlocked=true` 只记录本次 dry run 接受环境阻塞继续汇总；它不会把 `blocked-by-cuda-driver` 提升为 `ready`。
- 如果 `release-candidate-readiness-summary.json` 唯一 blocker 是 `blocked-by-cuda-driver`，`Test-FinalReleaseDryRun.ps1 -AllowRuntimeSmokeBlocked` 会把它显示为 `blocked-by-cuda-driver-owner-action` manual approval；这只是消除 artifact 顺序依赖，不会把 runtime proof 标记为 ready。
- external runtime proof 的默认状态为 `externalRuntimeProofState=template-only`、`externalRuntimeProofRuntimePackageKeyMatches=true`、`externalRuntimeProofLogSha256FormatReady=false`、`externalRuntimeProofLogSha256Matches=false`；它说明 key 已对齐但真实 smoke log hash 仍未回填。
- post-publish verification 的默认状态为 `postPublishVerificationState=template-only`，并且 `postPublishConsumerProjectIdentityReady=false`、`postPublishHostReady=false`、`postPublishCommandsReady=false`、`postPublishSmokeCommandRuntimeKeyReady=false`、`postPublishStdoutStderrSummaryReady=false`；它只能在真实渠道发布后用于关闭 release issue。
- `DebugListener real callback runtime proof` 仍为 `false`。
- signing/trust 当前为 `unsigned-or-not-requested`。
- public API bilingual documentation audit 已清零，`findingCount=0`，backlog 为空工作队列。

`Export-PublicApiBilingualDocumentationBacklog.ps1` 只把双语文档 finding 分组为可执行工作队列；它不能单独把 `Public API bilingual documentation` gate 变为 ready。当前该 gate 已由 `Test-PublicApiBilingualDocumentation.ps1` 报告 `findingCount=0` 后通过。

## Summary 字段读法

| 字段 | 当前含义 | 不能证明 |
| --- | --- | --- |
| `overallStatus` | final dry-run 汇总状态。当前可为 `ready-needs-manual-approval`。 | 不等于公开发布完成。 |
| `blockingIssueCount` | 自动门禁 blocker 数量。当前为 `0` 时说明自动证据没有阻塞项。 | 不替代人工审批、签名、Linux runner 或 runtime smoke。 |
| `manualApprovalCount` | 需要 release owner 明确确认的事项数量。 | 不表示这些事项已经批准。 |
| `warningCount` | 非阻塞 warning 数量。 | 不表示 warning 可以从发布说明中隐藏。 |
| `packageConsumerSmokeStatus` | full package consumer runtime smoke 状态。当前为 `blocked-by-cuda-driver`。 | 不等于 smoke passed。 |
| `packageConsumerEvidenceKind` | package consumer evidence 的细分类型。当前为 `full-runtime-package-consumer-smoke-driver-blocked`。 | 不等于 full runtime smoke passed。 |
| `runtimeSmokeClassification` | runtime smoke 分类。当前为 `runtime-smoke-driver-blocked`。 | 不表示 runtime API 已完整执行。 |
| `runtimeProofStatus` | 发布运行证明状态。当前为 `blocked-by-cuda-driver`。 | 不会被 `overallStatus=ready-needs-manual-approval` 或 `allowRuntimeSmokeBlocked=true` 覆盖。 |
| `runtimeProofRequiredForRelease` | 是否仍需要为公开发布补齐 runtime proof。当前为 `true`。 | 不表示已经得到 release owner 豁免。 |
| `runtimeProofBlockerOwnerActionStatus` | runtime proof 阻塞的 owner action 状态。默认 `owner-action-required`。 | 不等于 proof 已解决。 |
| `runtimeProofBlockerCategory` | 当前阻塞类别。默认 `cuda-driver-runtime-compatibility`。 | 不等于 API 缺陷分类。 |
| `externalRuntimeProofState` | external proof record 的 validator 状态。默认 `template-only`。 | 模板不是 runtime proof。 |
| `externalRuntimeProofRuntimePackageKeyMatches` | external proof runtime key 是否匹配 release target。默认模板为 `true`。 | 单独 key 匹配不证明 smoke 通过。 |
| `externalRuntimeProofLogSha256FormatReady` | smoke log SHA256 是否已按格式回填。默认 `false`。 | 不能替代真实日志文件。 |
| `externalRuntimeProofLogSha256Matches` | 使用真实日志校验时 hash 是否匹配。默认 `false`。 | false 时不能晋级 proof。 |
| `externalRuntimeProofOwnerActionStatus` | external proof 是否仍需 owner 回填。默认 `owner-action-required`。 | 不等于发布批准。 |
| `postPublishVerificationState` | 发布后验证记录状态。默认 `template-only`。 | 模板不是发布后验证证明。 |
| `postPublishCommandsReady` | restore/build/smoke 等命令是否已完整记录。默认 `false`。 | 不能替代真实 post-publish consumer 执行。 |
| `postPublishStdoutStderrSummaryReady` | stdout/stderr 摘要是否已回填。默认 `false`。 | false 时不能关闭 release issue。 |
| `isRuntimeExecutionEvidence` | 当前 package consumer evidence 是否能作为 runtime execution 证据。当前为 `false`。 | 不能被 native-copy、dependency probe 或 driver-blocked 输出替代。 |
| `isDependencyProbeOnly` | 当前 evidence 是否只证明 dependency probe/native load 层面。当前为 `true`。 | 不等于 TensorRT/CUDA runtime smoke passed。 |
| `isRealCallbackRuntimeProof` | 当前 evidence 是否可作为真实 callback runtime proof。当前为 `false`。 | 不等于 `realCallbackRuntimeProof=true`。 |
| `allowRuntimeSmokeBlocked` | 本次 dry run 接受环境阻塞继续汇总，便于输出完整证据。 | 不会把 `blocked-by-cuda-driver` 提升为 `ready`。 |
| `localFeedConsumerStatus` | local feed consumer 的 restore/build/native-copy 或 dependency probe 状态。 | 不等于 public package channel 已发布。 |
| `realCallbackRuntimeProof` | 是否已有真实 callback runtime proof。当前为 `false`。 | 不能被 schema-ready、precheck 或 design gate 替代。 |
| `signingStatus` | 当前签名/信任状态。 | `unsigned-or-not-requested` 不等于 signed。 |
| `userAcceptanceCatalogStatus` | sample/smoke catalog 是否完整登记。 | asset-required sample 不等于真实模型 smoke passed。 |

## 不能误读

不要把 final release dry run 理解为正式发布完成。它不能证明：

- nuget.org 或 GitHub Packages 已经发布成功。
- NVIDIA CUDA/cuDNN/TensorRT 再分发条款已完成法律复核。
- 当前机器已经通过 CUDA 13.2 runtime smoke。
- `isDependencyProbeOnly=true` 的 package consumer evidence 可以当作 runtime execution proof。
- `IDebugListener::processDebugTensor` 已被 TensorRT runtime 真实调用。
- Linux Ubuntu runtime 包已经在目标 runner 上完成真实 build/pack/consumer 验证。

只有 full package consumer 在兼容 runtime 环境中输出 `InvocationCount>0` 且 `IsRealCallbackRuntimeProof=True`，才能把真实 callback runtime proof 晋级。

## 发布前建议

发布负责人应逐项确认 dry run 的 `Manual Approval Items`：

- release checklist 是否还有 pending 项。
- public API bilingual documentation 是否仍保持 `findingCount=0`。
- signing/trust 是否符合当前发布渠道策略。
- CUDA error 35 是否已在已知限制和排障文档中清楚说明。
- Linux dry-run/handoff 是否已由目标 runner 的真实 evidence 替代。
