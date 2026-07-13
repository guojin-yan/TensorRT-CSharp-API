# Release Candidate 发布总结

本文是 TensorRtSharp4.0 当前发布候选状态的对外总结稿。它面向维护者、release owner、博客读者和准备试用项目的 .NET 用户，集中说明项目已经完成的工程化收口、仍需真实外部环境补齐的 blocker，以及哪些材料不能被写成 release proof。

当前总结结论：

- `blocked-real-proof-required`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

这表示项目已经形成完整的发布候选证据链和文档矩阵，但真实 close 仍等待 owner proof。本文不授权发布、不执行上传、不替代真实 proof record。

## 1. 已完成的收口面

当前项目已经完成以下面向发布候选的收口工作：

| 收口面 | 当前状态 |
| --- | --- |
| 接口覆盖 | TensorRT / CUDA manifest、source、generated surface 已可审计，deferred boundary 已显式记录 |
| 高层入口 | C# wrapper、samples、smoke、TensorRtExec、YoloVision、runtime package 文档已形成前台入口 |
| 文档矩阵 | 中文文章矩阵已覆盖项目定位、安装、样例、应用、runtime package、release proof 和 callback 边界 |
| 证据链 | release evidence bundle、close preflight、gap dashboard、final freeze、frontpage audit、final cross-check 已形成 |
| 质量门禁 | stale claim audit、TechnicalArticleRoadmap tests 和 release boundary tests 持续约束越级声明 |

这些材料让项目更接近“可交接、可理解、可复现”，但它们不是 `package-consumer-runtime`、`real-model-runtime`、Linux runner proof 或 `post-publish verification`。

## 2. README 与前台入口

Release issue 关闭仍有最后一道独立门禁：`release-issue-close-record-validation=blocked-template-only`，`release-issue-close-record-template.json` 不是 proof。只有 owner authorization、`package-consumer-runtime`、Linux runner proof、`real-model-runtime`、`post-publish verification` 和 release close preflight 全部通过后，才能回填 `release-issue-close-record.json` 并运行 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`；在此之前 `canCloseReleaseIssue=false` 必须保持不变。

README / README.zh-CN 当前已经提供一组发布候选前台入口：

- `release-final-audit-map.md`
- `release-public-story-pack.md`
- `release-owner-proof-backlog.md`
- `release-proof-non-substitutes.md`
- `release-article-index-and-publishing-order.md`
- `release-readme-frontpage-checklist.md`
- `release-final-owner-action-sequence.md`
- `release-frontpage-and-proof-boundary-final-audit.md`
- `release-candidate-final-cross-check.md`
- `release-candidate-article-matrix-summary.md`
- `release-candidate-publication-summary.md`

这些入口的目标是让不同读者能直接进入正确路径：用户看安装和样例，模型用户看 Classification / YoloVision，owner 看 proof backlog 和 final action sequence，维护者看 final cross-check 和 stale claim audit。

## 3. 样例与应用状态

当前样例和应用可按以下方式对外说明：

| 入口 | 可以说明 | 必须保留的边界 |
| --- | --- | --- |
| `samples/OnnxToEngine` | 最小 ONNX round-trip 和 engine build 路径 | 不代表任意外部模型 runtime proof |
| `applications/TensorRtExec` | CLI / WinForms、build/precheck report、normalized command、sidecar | `build-only`、`parse-only`、`sidecar-only` 不是 release proof |
| `samples/Classification` | 自备分类模型、labels、input 的样例入口 | 真实 `real-model-runtime` 需要资产、hash、license、runner log 和 validator |
| `samples/YoloVision` | YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom，det、cls、seg、obb、pose、sem 的统一样例框架 | support matrix 不能替代真实模型 proof |

当前统一 YOLO-family 样例名是 `YoloVision`。不要把旧检测样例名或旧项目文件作为当前入口重新写入文档、README 或测试。

## 4. 仍需 owner 完成的真实 proof

release close 仍需要五类真实输入：

1. owner authorization
2. `package-consumer-runtime`
3. Linux runner proof
4. `real-model-runtime`
5. `post-publish verification`

推荐执行顺序：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleAssetManifest.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidenceRecord.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof
```

如果 owner 没有提供真实外部条件，应保留 blocker，不要用本地 helper、文章、runbook 或 collection package 代替 proof。

## 5. 不可替代 proof 的材料

以下材料可以帮助执行，但不能关闭 release blocker：

- helper
- template
- draft
- runbook
- collection package
- input package
- local feed
- ProjectReference
- dependency probe
- build-only
- parse-only
- sidecar-only
- `blocked-by-cuda-driver`

这些材料出现时，应写清它们的用途：准备、诊断、交接或构建报告。它们不能被写成 runtime smoke、clean package consumer、真实模型运行或真实渠道验证。

## 6. 对外叙事建议

可以对外说：

- 项目已经进入发布候选收口阶段。
- API surface、samples、applications、runtime package 和 release evidence 已经有完整前台入口。
- YoloVision 已作为统一 YOLO-family 样例入口。
- TensorRtExec 已提供 CLI / WinForms 的 trtexec-like build/precheck/report 体验。
- 当前仍等待 owner 在真实环境补齐 release close proof。

不要对外说：

- 当前不得写成已经具备公开渠道放行结论。
- `package-consumer-runtime` 不能由 runbook 或 local feed 证明。
- `real-model-runtime` 不能由 support matrix 或 sidecar 证明。
- `post-publish verification` 不能在真实渠道发布前完成。
- `blocked-by-cuda-driver` 表示 smoke 通过。

## 7. 最终检查命令

发布候选总结更新后，至少运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StaleReleaseClaims.ps1
dotnet build .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-restore /m:1 /p:UseSharedCompilation=false /nr:false
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~TechnicalArticleRoadmap|FullyQualifiedName~ReleaseCandidateFinalEvidenceFreeze|FullyQualifiedName~CompatibleHostProofExecutionPack|FullyQualifiedName~ReleaseCloseGapDashboard"
```

如果这些检查通过，在没有真实外部 proof 条件时，合理结论是：发布候选说明、文章矩阵、README 前台和 proof boundary 已完成一致性收口；真实 release close 仍等待 owner proof。
