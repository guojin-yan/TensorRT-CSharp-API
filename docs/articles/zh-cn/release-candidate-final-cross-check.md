# Release Candidate 最终总检

本文是 TensorRtSharp4.0 发布候选收口阶段的最终总检。它把最终冻结、README 前台审计、owner backlog、proof 不可替代清单、文章矩阵和质量门禁放到同一个检查面，方便维护者在没有真实外部 proof 条件时确认项目表达一致；也方便 owner 在具备真实环境后直接切入 proof 回填。

当前总检结论仍是：

- `blocked-real-proof-required`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

这不是失败结论，而是证据边界结论。项目已经具备大量可审计材料，但 release close 仍依赖真实 proof，而不是更多 wrapper、template 或文章。

## 1. 总检输入

本轮总检至少读取以下材料：

| 材料 | 作用 |
| --- | --- |
| `release-candidate-final-evidence-freeze.md` | 冻结当前证据链和剩余 blocker |
| `release-frontpage-and-proof-boundary-final-audit.md` | 检查 README / docs / samples / applications 前台一致性 |
| `release-final-audit-map.md` | 汇总最终审计入口和执行顺序 |
| `release-owner-proof-backlog.md` | 拆分 owner 剩余真实 proof backlog |
| `release-proof-non-substitutes.md` | 固定不可替代 proof 清单 |
| `technical-article-roadmap.md` | 确认中文文章矩阵和发布顺序 |

这些材料可以证明项目已经有清晰的收口面，但不能证明真实 runtime proof 已经完成。

## 2. 仍未消失的五个 blocker

当前 release close 仍依赖以下五个真实条件：

1. owner authorization
2. `package-consumer-runtime`
3. Linux runner proof
4. `real-model-runtime`
5. `post-publish verification`

每个 blocker 都需要真实输入、真实日志、真实 hash、真实主机 metadata 和 validator 输出。没有这些证据时，应保留 blocker，不要修改测试或文档制造完成。

## 3. 不可替代 proof 清单

以下材料仍然只能作为 guidance、precheck、diagnostics 或 handoff：

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

它们的价值是降低 owner 执行成本，而不是替 owner 执行真实 proof。尤其要注意：TensorRtExec 的 build/precheck report 不能替代 `package-consumer-runtime`；YoloVision support matrix 不能替代 `real-model-runtime`；真实渠道发布后的 clean consumer 验证才属于 `post-publish verification`。

## 4. README 与前台入口检查

README / README.zh-CN 应继续保留以下入口：

- `release-final-audit-map.md`
- `release-public-story-pack.md`
- `release-owner-proof-backlog.md`
- `release-proof-non-substitutes.md`
- `release-article-index-and-publishing-order.md`
- `release-readme-frontpage-checklist.md`
- `release-final-owner-action-sequence.md`
- `release-frontpage-and-proof-boundary-final-audit.md`
- `release-candidate-final-cross-check.md`

README 第一屏还应保留：

- `blocked-real-proof-required`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

如果 README、docs index、toc 或 roadmap 中缺少其中任意入口，应先补齐入口，再继续写更多文章。

## 5. 样例与应用检查

样例和应用仍按以下边界解释：

| 入口 | 当前价值 | 不能替代 |
| --- | --- | --- |
| `samples/OnnxToEngine` | 最小 ONNX round-trip 和 engine build 路径 | 任意外部模型 runtime proof |
| `applications/TensorRtExec` | CLI / WinForms build-only、precheck、report、sidecar | `package-consumer-runtime` |
| `samples/Classification` | 用户自备分类模型样例 | release package consumer proof |
| `samples/YoloVision` | YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom，det、cls、seg、obb、pose、sem 统一样例 | `real-model-runtime`，除非真实资产和日志通过 validator |

当前统一 YOLO-family 样例入口是 `YoloVision`。不要把旧检测样例名或旧项目文件重新写成当前入口。

## 6. 真实 proof 回填入口

如果 owner 提供真实外部条件，按以下顺序执行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleAssetManifest.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidenceRecord.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof
```

真实 proof 回填后，再刷新：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleasePromotionIssueRecord.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseClosePreflight.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StaleReleaseClaims.ps1
```

如果任意 validator 失败，应保留对应 blocker。

## 7. 最终质量门禁

最终总检至少运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StaleReleaseClaims.ps1
dotnet build .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-restore /m:1 /p:UseSharedCompilation=false /nr:false
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~TechnicalArticleRoadmap|FullyQualifiedName~ReleaseCandidateFinalEvidenceFreeze|FullyQualifiedName~CompatibleHostProofExecutionPack|FullyQualifiedName~ReleaseCloseGapDashboard"
```

这三个命令分别验证：

1. 文档没有越级声明。
2. 质量测试项目仍能构建。
3. release/frontpage/article roadmap 关键门禁仍通过。

如果没有真实外部条件，本阶段可以得出的结论只能是：项目 frontpage、文档矩阵和 release proof boundary 已一致；真实 release close 仍等待 owner proof。
