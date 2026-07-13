# README 前台与 Proof Boundary 最终审计

本文是 TensorRtSharp4.0 发布候选前台入口的最终一致性审计。目标是把 README、README.zh-CN、docs index、toc、samples、applications 和 release owner 文档放到同一张检查面，确认它们都表达同一个事实：项目已经进入发布候选收口，但真实 release close 仍依赖 owner 在外部环境补齐 proof。

当前必须继续保留：

- `blocked-real-proof-required`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

这些状态不是文案保守，而是 release evidence 的真实边界。只有真实 proof record、真实日志、真实 hash、真实主机 metadata 和 validator 一起通过后，相关 blocker 才能被消除。

## 1. README 前台一致性

README / README.zh-CN 应在第一屏给出三类信息：

| 类型 | 必须出现 | 原因 |
| --- | --- | --- |
| 当前状态 | `blocked-real-proof-required`、`performsPublish=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false` | 防止把 frozen evidence 写成发布放行 |
| 快速入口 | final audit map、public story pack、owner proof backlog、non-substitute proof、article index、frontpage checklist、owner action sequence | 让用户和 owner 能直接进入正确文档 |
| 样例和工具边界 | TensorRtExec、YoloVision、build-only、parse-only、sidecar-only、ProjectReference、blocked-by-cuda-driver | 防止把工具报告或环境 blocker 写成 proof |

README 可以强调项目已经具备完整的 bridge、wrapper、samples、applications、runtime package strategy 和 release evidence automation，但不能暗示五个真实 blocker 已经消失。

## 2. Docs index / toc 一致性

`docs/index.md` 和 `docs/toc.yml` 应包含以下前台与发布边界文章：

- `release-final-audit-map.md`
- `release-public-story-pack.md`
- `release-owner-proof-backlog.md`
- `release-proof-non-substitutes.md`
- `release-article-index-and-publishing-order.md`
- `release-readme-frontpage-checklist.md`
- `release-final-owner-action-sequence.md`
- `release-frontpage-and-proof-boundary-final-audit.md`

如果 README 引用了文章，但 index 或 toc 没有入口，用户会在文档站里断链；如果 index/toc 有文章但 README 不提示状态，用户又容易把 release proof 边界读晚。

## 3. Samples / applications 一致性

样例和应用入口必须保持以下边界：

| 入口 | 可以说明 | 不能替代 |
| --- | --- | --- |
| `samples/OnnxToEngine` | 最小 ONNX round-trip、engine build、parser path | 任意外部模型 runtime proof |
| `applications/TensorRtExec` | build/precheck report、normalized command、sidecar、WinForms 入口 | `package-consumer-runtime` |
| `samples/Classification` | 用户自备分类模型的 sample runner | release package consumer proof |
| `samples/YoloVision` | YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom 与 det、cls、seg、obb、pose、sem 的统一样例框架 | `real-model-runtime`，除非真实资产和日志通过 validator |

旧 YOLO 检测样例名不应作为当前入口重新出现。当前统一入口是 `YoloVision`。

## 4. Proof boundary 一致性

所有前台文档都应使用同一套 proof 分层：

| 证据类型 | 当前作用 | 是否能关闭 release blocker |
| --- | --- | --- |
| helper/template/draft/runbook/collection package/input package | owner guidance | 不能 |
| local feed / ProjectReference | 开发或待发布输入验证 | 不能 |
| dependency probe | 依赖诊断 | 不能 |
| build-only / parse-only / sidecar-only | 构建、解析、metadata 和交接证据 | 不能 |
| `blocked-by-cuda-driver` | 当前主机兼容性阻塞 | 不能 |
| `package-consumer-runtime` | 真实 clean consumer runtime proof | validator 通过后才可以 |
| `real-model-runtime` | 真实模型样例 proof | validator 通过后才可以 |
| `post-publish verification` | 真实渠道发布后的 clean consumer 验证 | validator 通过后才可以 |

这套分层需要在 README、docs index、sample README、TensorRtExec README 和 release owner 文档里保持一致。

## 5. Owner 最后缺口

当前仍有五个 release close blocker：

1. owner authorization
2. `package-consumer-runtime`
3. Linux runner proof
4. `real-model-runtime`
5. `post-publish verification`

如果 owner 没有提供真实外部条件，文档工作只能继续提升可读性和执行清晰度，不能删除 blocker。下一步如果进入真实 proof 回填，应优先执行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleAssetManifest.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidenceRecord.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof
```

## 6. 最终审计命令

frontpage 或 proof boundary 修改后，至少执行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StaleReleaseClaims.ps1
dotnet build .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-restore /m:1 /p:UseSharedCompilation=false /nr:false
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~TechnicalArticleRoadmap|FullyQualifiedName~ReleaseCandidateFinalEvidenceFreeze|FullyQualifiedName~CompatibleHostProofExecutionPack|FullyQualifiedName~ReleaseCloseGapDashboard"
```

如果 stale claim audit 出现 finding，应优先修正文档，而不是放宽规则。发布可信度来自一致的边界，而不是把边界词从文档里拿掉。
