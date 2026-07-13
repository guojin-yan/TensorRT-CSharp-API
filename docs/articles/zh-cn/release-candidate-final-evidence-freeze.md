# Release Candidate Final Evidence Freeze

`Export-ReleaseCandidateFinalEvidenceFreeze.ps1` 用于生成发布候选最终证据冻结快照。它把当前 release close 证据链、剩余 blocker、validator commands、不可替代 proof 类型和 owner action 聚合到一个最终检查面板中。

固定边界：

- `recordKind=release-candidate-final-evidence-freeze`
- `freezeState=blocked-real-proof-required`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

它不执行真实发布、不上传 NuGet.org、不上传 GitHub Packages、不创建 GitHub Release asset、不关闭 release issue。它只能证明“当前证据链已冻结并仍有真实 proof blocker”，不能把任何 helper、template、draft、runbook 或 collection package 晋级为 proof。

Owner 最短执行面以 `owner-release-execution-package` 的 `oneScreenReleaseHoldChecklist` 为准；final evidence freeze 会引用或镜像该一屏 Release Hold 清单，但它仍然只是 owner guidance，不是 owner authorization、`package-consumer-runtime`、Linux runner proof、`real-model-runtime` 或 `post-publish verification` 的真实 proof。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateFinalEvidenceFreeze.ps1
```

输出：

- `artifacts/final-release/release-candidate-final-evidence-freeze.json`
- `artifacts/final-release/release-candidate-final-evidence-freeze.md`

## 冻结的 5 个 Blocker

1. `owner-authorization`
   - 需要真实 owner approval、发布渠道选择、NVIDIA redistribution disposition 和手动命令 materialization。

2. `package-consumer-runtime`
   - 需要 compatible CUDA/TensorRT host 上 clean package consumer runtime smoke。
   - 必须通过 `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof`。

3. `linux-runner-proof`
   - 需要真实 Linux x64 runner evidence。
   - 必须通过 `Test-LinuxRunnerEvidenceRecord.ps1`。

4. `real-model-runtime`
   - 需要真实 Classification/YoloVision 模型资产、hash、license、TensorRtExec sidecar、sample runner log。
   - 必须通过 `Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog`。
   - YoloVision 范围固定为 YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom，以及 det/cls/seg/obb/pose/sem（det、cls、seg、obb、pose、sem）。

5. `post-publish verification`
   - 需要真实发布渠道 package、下载 hash、clean consumer restore/build/probe/smoke 日志。
   - 必须通过 `Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof`。

## 不可替代材料

以下内容可以辅助 owner 执行，但不能被写成 proof：

- helper
- template
- draft
- runbook
- collection package
- input package
- local feed
- `ProjectReference`
- DependencyProbe
- dependency-probe-only
- `blocked-by-cuda-driver`
- build-only
- parse-only
- sidecar-only
- Windows handoff for Linux proof
- owner-action-required without validator pass

## 与其它产物的关系

- `release-evidence-bundle` 是证据聚合。
- `release-promotion-issue-record` 是 promotion issue 视图。
- `release-close-preflight` 是 close 前门禁。
- `release-close-gap-dashboard` 是 blocker dashboard。
- `compatible-host-proof-execution-pack` 是 owner 一站式执行入口。
- `owner-release-execution-package` 是一屏 Release Hold 清单入口。
- `release-candidate-final-evidence-freeze` 是最终冻结快照。

只有真实 proof record、真实日志、真实 hash、真实主机 metadata 和对应 validator 一起通过后，相关 blocker 才能消失。freeze 本身永远不是 proof。
