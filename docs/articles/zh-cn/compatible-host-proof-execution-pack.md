# Compatible Host Proof Execution Pack

`Export-CompatibleHostProofExecutionPack.ps1` 是 release close 前的 owner 执行入口聚合器。它把 `release-close-gap-dashboard` 中的 5 个 blocker 汇总成一份可执行、可验证、不可误晋级的命令包。

固定边界：

- `recordKind=compatible-host-proof-execution-pack`
- `packageState=owner-action-required`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

它不执行真实发布、不上传 NuGet.org、不上传 GitHub Packages、不创建 GitHub Release asset、不关闭 release issue。它只是把已有分散执行包、validator 和真实输入要求集中到一个 owner 可复制的路线图中。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CompatibleHostProofExecutionPack.ps1
```

输出：

- `artifacts/final-release/compatible-host-proof-execution-pack.json`
- `artifacts/final-release/compatible-host-proof-execution-pack.md`

## 覆盖的 5 个 Blocker

1. `owner-authorization`
   - owner 必须明确发布渠道、授权状态、NVIDIA redistribution disposition 和手动命令 materialization。
   - execution pack 只记录 placeholder，不运行 `dotnet nuget push`。

2. `package-consumer-runtime`
   - 必须在 compatible CUDA/TensorRT host 上运行 clean package consumer restore/build/smoke。
   - 必须生成真实 `external-runtime-proof-record.json`。
   - 必须通过 `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof`。
   - bridge-only package consumer log、bridge-only wrapper surface、`Skipped=True`、`dependency-probe-only`、`WrapperSurfaceEvidenceKind=compile-surface-proof`、`IsRuntimeExecutionProof=False`、ONNX Parser diagnostic snapshot、ONNX ParserRefitter diagnostic snapshot 和 copied managed diagnostic snapshot 都不能替代 clean package consumer runtime proof。

3. `linux-runner-proof`
   - 必须来自真实 Linux x64 runner 主机。
   - 必须通过 `Test-LinuxRunnerEvidenceRecord.ps1`。
   - Windows handoff、template-only、dry-run-only 不能替代。

4. `real-model-runtime`
   - 必须提供 Classification/YoloVision 真实模型、labels、输入资产、license、SHA256、TensorRtExec sidecar、sample runner log。
   - 必须通过 `Test-SampleAssetManifest.ps1` 和 `Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog`。
   - YoloVision 范围固定为 YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom，以及 det/cls/seg/obb/pose/sem（det、cls、seg、obb、pose、sem）。

5. `post-publish verification`
   - 只能在 owner-approved real publication 之后执行。
   - 必须来自真实渠道 package、下载后的 nupkg SHA256、clean consumer restore/build/probe/smoke 日志。
   - 必须通过 `Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof`。

## 不能替代 Proof 的材料

以下材料可以帮助 owner 执行，但不能被写成 proof：

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
- `Skipped=True`
- `blocked-by-cuda-driver`
- bridge-only package consumer log
- bridge-only wrapper surface
- `WrapperSurfaceEvidenceKind=compile-surface-proof`
- `IsRuntimeExecutionProof=False`
- Parser/ParserRefitter diagnostic snapshots
- copied managed diagnostic snapshot
- build-only
- parse-only
- sidecar-only
- mismatched log SHA256
- Windows handoff for Linux proof
- owner-action-required without validator pass

## 与其它产物的关系

- `release-close-gap-dashboard` 说明当前还有哪些 blocker。
- `compatible-host-proof-execution-pack` 把这些 blocker 变成 owner 执行顺序。
- `compatible-host-proof-backfill-package`、`compatible-host-runtime-proof-collection-bundle` 和 `external-runtime-proof-collection-package` 是 package-consumer-runtime 的执行材料。
- `real-model-and-package-proof-input-package` 是 Classification/YoloVision 与 package proof 的真实输入清单。
- `post-publish-verification-collection-package` 只能在真实发布后使用。
- local feed、draft、helper scan、bridge-only package consumer log、dependency-probe-only log、collection package 和 input package 都不能关闭 release issue。

只有真实 proof record、真实日志、真实 hash、真实主机 metadata 和对应 validator 一起通过后，相关 blocker 才能被晋级。execution pack 本身永远不是 proof。
