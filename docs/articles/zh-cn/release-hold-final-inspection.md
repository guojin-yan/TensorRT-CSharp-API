# Release Hold Final Inspection

本文用于固定 TensorRtSharp4.0 在 release candidate final hold 状态下的最终巡检顺序。它不是新的 proof record，也不是公开发布完成声明；它只定义在没有真实外部 proof 时维护者应如何确认项目仍处于可审计、可等待 owner 执行的状态。

Owner 最短执行面以 `owner-release-execution-package` 的 `oneScreenReleaseHoldChecklist` 为准。巡检时必须确认 README、DocFX index、owner handoff、freeze 和 final hold 文档都能进入这一屏 Release Hold 清单；它仍是 guidance，不是 owner authorization、`package-consumer-runtime`、Linux runner proof、`real-model-runtime` 或 post-publish verification 的真实 proof。

`release-proof-readiness-snapshot` 是同一组 5 个 blocker 的紧凑状态视图。最终巡检时必须确认它仍保持 `blocked-real-proof-required`、`canPublishPublicly=false` 和 `canCloseReleaseIssue=false`，直到真实 proof validators 通过。

最终巡检还必须确认 `release-issue-close-record-validation=blocked-template-only` 没有被误写成可关闭状态。`release-issue-close-record-template.json` 不是 proof；只有 release close preflight、post-publish verification、stale claim audit、release evidence bundle SHA256、rollback plan 和 owner final close decision 都齐全，并通过 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady` 后，owner 才能手工关闭 release issue。

## 当前 Final Hold 状态

当前必须保持：

```text
FreezeState=blocked-real-proof-required
performsPublish=false
canPublishPublicly=false
canCloseReleaseIssue=false
```

仍未完成的真实 release close blockers：

1. owner authorization
2. package-consumer-runtime
3. Linux runner proof
4. real-model-runtime
5. post-publish verification

只要上述任一项没有真实可复验证据，release issue 就不能关闭，公开发布也不能被写成已经完成。

## Owner Backfill Track 复核

巡检时必须确认 front door、artifact 和 owner 文档都指向同一组 final backfill track：

1. `package-consumer-runtime`：只能由真实 `external-runtime-proof-record.json` 加 `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof` 晋级。
2. `linux-runner-proof`：只能由真实 Linux x64 runner record 加 `Test-LinuxRunnerEvidenceRecord.ps1` 晋级。
3. `real-model-runtime`：只能由 Classification / YoloVision 真实模型日志、hash、资产和 `Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog` 晋级。
4. `post-publish verification`：只能由真实发布渠道的 clean consumer record 加 `Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof` 晋级。

`ownerProofFinalBackfillTracks`、owner execution package、input package 和 collection package 都是执行导航。它们必须继续输出 `canCloseReleaseIssue=false`，并明确 local feed、ProjectReference、bridge-only log、`Skipped=True`、mismatched log SHA256、build-only/precheck、sidecar-only、runbook 和 Windows handoff for Linux proof 不能替代真实 proof。

## 最终巡检顺序

### 1. README Front Door

检查：

- `README.md`
- `README.zh-CN.md`

必须包含：

- `blocked-real-proof-required`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- owner authorization
- package-consumer-runtime
- Linux runner proof
- real-model-runtime
- post-publish verification
- `release-candidate-final-hold-owner-waiting.md`
- `release-owner-action-checklist-final-hold.md`
- `release-hold-final-inspection.md`

不能把 release candidate final hold 写成 publicly released、published to NuGet、post-publish verification completed 或 release issue closure-ready。

### 2. DocFX Index 与 Toc

检查：

- `docs/index.md`
- `docs/toc.yml`

必须能从文档首页和目录进入：

- `release-candidate-final-evidence-freeze.md`
- `release-candidate-publication-summary.md`
- `release-candidate-final-hold-owner-waiting.md`
- `release-owner-action-checklist-final-hold.md`
- `release-hold-final-inspection.md`

如果新增 release-facing 文章而没有接入 index/toc，应视为文档发布面不完整。

### 3. Technical Article Roadmap

检查：

- `docs/articles/zh-cn/technical-article-roadmap.md`

必须包含当前 release hold 文章链路：

- Release Candidate Final Evidence Freeze
- Release Candidate 发布总结
- Release Candidate Final Hold 与 Owner 等待状态
- Release Owner Action Checklist Final Hold
- Release Hold Final Inspection

路线图可以继续增长，但新增项不能复用旧编号，不能把等待 owner 的事项写成已完成 proof。

### 4. Evidence Freeze Artifact

检查：

- `artifacts/final-release/release-candidate-final-evidence-freeze.json`
- `artifacts/final-release/release-candidate-final-evidence-freeze.md`
- `artifacts/final-release/release-proof-readiness-snapshot.json`
- `artifacts/final-release/release-proof-readiness-snapshot.md`

必须保持：

- `blocked-real-proof-required`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- blocker count 仍覆盖 owner authorization、package-consumer-runtime、Linux runner proof、real-model-runtime、post-publish verification。

如果 JSON/MD 发生变化，应重新运行 freeze exporter 或对应 quality tests，而不是手动改状态制造通过。

### 5. Stale Claim Audit

检查：

- `artifacts/final-release/stale-release-claims-audit.json`
- `artifacts/final-release/stale-release-claims-audit.md`
- `eng/Test-StaleReleaseClaims.ps1`

每次修改 release-facing 文档后都应运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StaleReleaseClaims.ps1
```

审计通过只能证明没有明显过度发布宣称，不能替代 package-consumer-runtime、real-model-runtime、Linux runner proof 或 post-publish verification。

### 6. Owner Checklist

检查：

- `docs/articles/zh-cn/release-owner-action-checklist-final-hold.md`

该文件必须继续列出 owner 的五个真实动作：

- owner authorization
- package-consumer-runtime
- real-model-runtime
- Linux runner proof
- post-publish verification

owner checklist 是执行清单，不是 proof。只有真实日志、真实输入、真实环境和 validator 结果才能改变 blocker 状态。

### 7. Sample Naming

YOLO family 样例统一命名为：

```text
samples/YoloVision
```

它覆盖 YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom 与 det、cls、seg、obb、pose、sem 的说明边界。不能回退到旧的检测样例目录名，也不能恢复旧的检测样例项目文件名。

`YoloVision` 的文档和 sidecar 可以帮助准备 real-model-runtime，但不能替代真实模型、真实 input、labels、license、SHA256 和 sample run log。

### 8. Proof Validators

真实 proof 到来后才运行对应 validator：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleAssetManifest.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidenceRecord.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof
```

如果没有真实 owner 授权、真实 package consumer、真实 Linux runner、真实模型资产或真实发布后 feed，不要创建伪 proof 文件。

## 不可接受替代

以下材料不能替代真实 proof：

- ProjectReference consumer。
- local feed。
- draft package。
- helper project。
- build-only。
- parse-only。
- dry-run。
- sidecar-only。
- collection bundle。
- input package。
- runbook。
- `blocked-by-cuda-driver`。
- `blocked-by-application-control`。
- `applications/TensorRtExec` 的 ONNX build/precheck report。

`TensorRtExec` 仍然是重要的 ONNX-to-engine CLI / WinForms 工具，但它的 build/precheck report 只属于构建或解析证据，不属于 package-consumer-runtime 或 real-model-runtime proof。

## 最小验证组合

无真实 proof 条件时，release hold 巡检至少执行：

```powershell
Set-Location .
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StaleReleaseClaims.ps1
dotnet build .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-restore /m:1 /p:UseSharedCompilation=false /nr:false
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~TechnicalArticleRoadmap|FullyQualifiedName~ReleaseCandidateFinalEvidenceFreeze|FullyQualifiedName~CompatibleHostProofExecutionPack|FullyQualifiedName~ReleaseCloseGapDashboard"
```

如果这些验证通过，结论仍然只能写为 release candidate final hold，不能写成公开发布完成。
