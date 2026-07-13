# Compatible Host Proof Backfill Package

`compatible-host-proof-backfill-package` 是面向 release owner 的真实环境执行材料。它不替 owner 执行发布，也不生成 proof，而是把 compatible CUDA/TensorRT 主机上需要采集的证据拆成可回填、可验证的清单。

生成脚本是 `eng/Export-CompatibleHostProofBackfillPackage.ps1`，输出：

- `artifacts/final-release/compatible-host-proof-backfill-package.json`
- `artifacts/final-release/compatible-host-proof-backfill-package.md`

该包必须保持 `performsPublish=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`。它是 owner guidance，不是 owner authorization、package-consumer-runtime proof、post-publish verification proof、real-model-runtime proof 或 package push。

## 为什么需要 compatible host 回填包

当前 release close preflight 仍处于 `blocked-real-proof-required`。这不是文档缺口，而是真实运行环境缺口：本仓库可以生成模板、runbook、collection package 和 validator，但不能伪造真实 CUDA/TensorRT 主机上的 runtime smoke。

Compatible host 回填包的目标是把最后一公里拆清楚：

- 发布前：采集 `package-consumer-runtime` external runtime proof。
- Linux：采集 Linux runner proof。
- 样例：采集 Classification/YoloVision 的 `real-model-runtime` proof。
- 发布后：采集 post-publish verification proof。
- 收口：刷新 `Export-ReleaseClosePreflight.ps1`，只有真实 proof 都通过后才能进入 owner close review。

## Owner Proof Final Backfill Tracks

当前 artifact 会把最终 owner proof 回填拆成 4 条 track，每条 track 都有 input JSON、validator command、expected artifacts、required owner inputs、log fields、SHA256 fields 和 promotion blockers：

| Track | Proof class | Input JSON | Validator |
|---|---|---|---|
| `package-consumer-runtime` | `package-consumer-runtime` | `artifacts/final-release/external-runtime-proof-record.json` | `Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -RequireExistingLog -FailOnNotProof` |
| `linux-runner-proof` | `linux-runner-proof` | `artifacts/linux-dry-run/linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22/linux-runner-evidence-record.json` | `Test-LinuxRunnerEvidenceRecord.ps1 -RuntimePackageKey linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22` |
| `real-model-runtime` | `real-model-runtime` | `artifacts/user-acceptance/sample-run-evidence-record.json` | `Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog` |
| `post-publish-verification` | `post-publish-package-consumer-runtime` | `artifacts/final-release/post-publish-verification-record.json` | `Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json -RequireExistingLog -FailOnNotProof` |

## Required Owner Inputs

Owner 必须按 track 回填真实字段，而不是只复制模板：

- `package-consumer-runtime`：runtime package key、managed/runtime nupkg SHA256、clean consumer project identity、host OS/GPU/driver/CUDA/TensorRT/cuDNN metadata、restore/build/probe/smoke command、stdout/stderr summary、smoke log path、smoke log SHA256、`no ProjectReference`。
- `linux-runner-proof`：Linux runtime package key、真实 Linux x64 host metadata、CUDA/TensorRT/cuDNN runtime metadata、runner command、runner log path、runner log SHA256。
- `real-model-runtime`：Classification/YoloVision model、labels、input asset、license、SHA256、TensorRtExec build report、evidence sidecar、sample runner log path、sample runner log SHA256、`proofClassification=real-model-runtime`。
- `post-publish-verification`：真实 channel URL/source、downloaded managed/runtime nupkg SHA256、timestamped SHA256 source notes、clean consumer root、PackageReference-only consumer、restore/build/native asset/dependency probe/runtime smoke logs、所有 log SHA256。

## Expected Artifacts

完成回填后，owner 至少应能产出并验证以下 artifact：

- `artifacts/final-release/external-runtime-proof-record.json`
- `artifacts/final-release/external-runtime-proof-validation.json`
- `artifacts/linux-dry-run/linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22/linux-runner-evidence-record.json`
- `artifacts/linux-dry-run/linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22/linux-runner-evidence-validation.json`
- `samples/assets/classification-assets.json`
- `samples/assets/yolovision-assets.json`
- `artifacts/user-acceptance/sample-run-evidence-record.json`
- `artifacts/user-acceptance/sample-run-evidence-record-validation.json`
- `artifacts/final-release/post-publish-verification-record.json`
- `artifacts/final-release/post-publish-verification-validation.json`

## Promotion Blockers

以下材料必须保持为非 proof，不能关闭 release issue：

- bridge-only package consumer log
- bridge-only wrapper surface
- `Skipped=True`
- dependency-probe-only
- `WrapperSurfaceEvidenceKind=compile-surface-proof`
- `IsRuntimeExecutionProof=False`
- `ProjectReference`
- mismatched log SHA256
- Parser/ParserRefitter diagnostic snapshots
- copied managed diagnostic snapshot
- Windows handoff for Linux proof
- template-only record
- input package
- build-only / parse-only / sidecar-only
- missing model/input/license hashes
- missing execution steps

## Package Consumer Runtime

`package-consumer-runtime` 只能来自干净 consumer 的真实 runtime smoke。它不能由以下材料替代：

- local feed
- ProjectReference
- DependencyProbe
- build-only
- parse-only
- sidecar-only
- dependency-probe-only
- blocked-by-cuda-driver
- template、draft、runbook 或 collection package

owner 在兼容主机上需要回填 managed/runtime nupkg SHA256、runtimePackageKey、consumer project identity、host metadata、restore/build/probe/smoke command、stdout/stderr summary 和 smoke log SHA256，然后运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -RequireExistingLog -FailOnNotProof
```

## Linux Runner Proof

Linux runner proof 需要目标 runtime key、Linux host metadata、CUDA/TensorRT/cuDNN 版本、命令日志和 validator 输出。Windows handoff、template-only record 或 `blocked-by-cuda-driver` 不能写成 Linux proof。

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidenceRecord.ps1 -RuntimePackageKey linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22
```

## Real Model Runtime

Classification 和 YoloVision 的真实样例运行用于证明 `real-model-runtime`。它们不能替代 release proof record，也不能写成 `package-consumer-runtime`。

YoloVision 当前面向 YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom，并覆盖 det/cls/seg/obb/pose/sem（det、cls、seg、obb、pose、sem）。owner 仍必须提供真实模型资产：

- model path 与 model SHA256。
- labels path 与 labels SHA256。
- input image 或 input tensor 与 SHA256。
- 模型 license 与再分发说明。
- TensorRtExec build report 与 evidence sidecar。
- sample runner command、sample runner log、sampleRunLogSha256。
- stdoutSummary 或 stderrSummary。
- `proofClassification=real-model-runtime`。
- `canPromoteRealModelRuntime=true`。

校验命令：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog
```

## Post-Publish Verification

Post-publish verification 必须在真实渠道发布之后执行。它需要 clean consumer project 从真实 channel 下载 package，并记录 package URL、downloaded nupkg SHA256、restore/build/probe/smoke logs、stdout/stderr summary 和 SHA256 匹配结果。

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json -RequireExistingLog -FailOnNotProof
```

`local feed`、`ProjectReference`、draft、helper scan 或 collection package 都不能关闭 release issue。clean consumer scan 只是组成部分，不是完整 post-publish proof。

## 收口规则

完成 compatible host 回填后，owner 需要重新运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StaleReleaseClaims.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseClosePreflight.ps1
```

只有当 release close preflight 不再是 `blocked-real-proof-required`，且 owner authorization、package-consumer-runtime、post-publish verification、Linux runner proof 和必要 real-model-runtime proof 都成立时，才能进入 release issue close review。

在那之前，该包必须保持 owner-action-required 和 `canCloseReleaseIssue=false`。
