# 发布前最终审计地图

TensorRtSharp4.0 当前已经具备完整的 release evidence bundle、release close preflight、compatible host proof execution pack 和 release candidate final evidence freeze。最终审计地图的作用不是宣布发布完成，而是把“已经能自动复核的内容”和“必须由 owner 在真实环境补齐的内容”放到同一个检查面。

当前固定状态仍是：

- `freezeState=blocked-real-proof-required`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`
- `package-consumer-runtime` 仍需要真实 clean consumer runtime smoke
- `real-model-runtime` 仍需要 Classification / YoloVision 真实模型、真实输入、license、hash 和 sample-run-evidence
- `post-publish verification` 只能在 owner 完成真实渠道发布后执行

这篇文章可以作为 release issue、README 和对外博客的审计入口，帮助读者理解：项目已经接近发布候选，但不能把 helper、template、draft、runbook、collection package、local feed、ProjectReference、build-only、parse-only、sidecar-only 或 `blocked-by-cuda-driver` 写成 proof。

## 1. 先看冻结快照

第一步读取最终冻结快照：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateFinalEvidenceFreeze.ps1
```

重点检查：

| 字段 | 期望当前值 | 含义 |
| --- | --- | --- |
| `recordKind` | `release-candidate-final-evidence-freeze` | 当前是冻结快照，不是 proof |
| `freezeState` | `blocked-real-proof-required` | 仍缺真实 proof |
| `performsPublish` | `false` | 脚本不发布包 |
| `canPublishPublicly` | `false` | 还没有 owner 授权和真实 proof |
| `canCloseReleaseIssue` | `false` | release issue 不能关闭 |

如果有文档把这些值写成 `true`，应立即视为 stale claim。

## 2. 再看 release close preflight

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseClosePreflight.ps1
```

`release-close-preflight` 是关闭 issue 前的聚合门禁。它不会上传 NuGet，不会创建 GitHub Release，也不会把 dry-run 结果提升为 release proof record。当前它必须继续报告 owner action，直到以下真实证据通过 validator：

- `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof`
- `Test-LinuxRunnerEvidenceRecord.ps1`
- `Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog`
- `Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof`

## 3. 区分五类证据

| 证据层级 | 当前项目可自动生成 | 能否关闭 blocker |
| --- | --- | --- |
| helper / runbook / collection package | 可以 | 不能 |
| build-only / parse-only / sidecar-only | 可以 | 不能 |
| local feed consumer / ProjectReference consumer | 可以 | 不能 |
| package-consumer-runtime proof | 需要兼容主机真实执行 | 通过 validator 后才可以 |
| post-publish verification proof | 需要真实渠道发布后执行 | 通过 validator 后才可以 |

最容易混淆的是 TensorRtExec 与 OnnxToEngine：它们能生成可审计的构建报告、命令归一化和 sidecar，但 build-only 不证明任意外部模型的推理输出正确，也不属于 `package-consumer-runtime`。

## 4. 样例和真实模型审计

样例层面分成两个路径：

1. 低资产依赖样例，例如 `DynamicShape`、`InferenceBindings`、`OnnxToEngine`，主要证明 wrapper、parser、engine build 或最小 round-trip。
2. 外部资产样例，例如 `Classification` 和 `YoloVision`，必须由 owner 提供真实模型、labels、input、license、SHA256 和 runner log。

YoloVision 的支持范围是 `YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom`，任务范围是 `det、cls、seg、obb、pose、sem`。这是一套 support matrix，不是 `YoloVision Passed=True` 的真实日志。只有真实模型、真实输入和 sample-run-evidence record 一起通过，才可以写成 `real-model-runtime`。

## 5. stale claim 审计

发布前必须运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StaleReleaseClaims.ps1
```

重点避免以下类型的越级句式：

- 把 `blocked-by-cuda-driver` 写成 smoke 已通过。
- 把 runtime proof 写成已完成。
- 不要把当前状态写成公开发布就绪。
- 把 post-publish verification 写成已完成。
- 把 `package-consumer-runtime` 写成已通过。
- 把 `real-model-runtime` 写成已通过。
- 把 `build-only` 写成 release proof。
- 把 `parse-only` 写成 implemented TensorRT 行为。
- 把 `sidecar-only` 写成 runtime proof。
- 把 `canPublishPublicly` 写成 true。
- 把 `canCloseReleaseIssue` 写成 true。

如果这些词必须出现在文档里，只能作为“不要这样写”的反例或 validator 规则说明。

## 6. 最终审计顺序

建议 owner 或维护者按以下顺序执行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CompatibleHostProofExecutionPack.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateFinalEvidenceFreeze.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleasePromotionIssueRecord.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseClosePreflight.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StaleReleaseClaims.ps1
```

然后再按真实外部条件补齐：

1. owner authorization
2. package-consumer-runtime
3. linux-runner-proof
4. real-model-runtime
5. post-publish verification

直到这些 blocker 被真实 proof record 和 validator 消除前，对外宣传可以说“发布候选证据链已冻结、剩余 owner action 明确”，不能说“已经公开发布就绪”。
