# Release Close Gap Dashboard

`Export-ReleaseCloseGapDashboard.ps1` 用来把 release close 前剩余的真实 proof blocker 摊开成 owner dashboard。它读取 `release-close-preflight`、`release-evidence-bundle`、`release-promotion-issue-record` 和 `real-model-and-package-proof-input-package`，并生成每个 blocker 的 proof class、真实输入、validator、owner action 和不可替代材料。

该 dashboard 是 owner guidance，不是 proof。它不发布包、不上传 release asset、不批准公开发布、不关闭 release issue。

当前 owner 最短执行面以 `owner-release-execution-package` 的 `oneScreenReleaseHoldChecklist` 为准；dashboard 只把同一组 blocker 摊开成 owner action，不会把一屏 Release Hold 清单、runbook、collection package 或 input package 晋级为 proof。

固定边界：

- `recordKind=release-close-gap-dashboard`
- `dashboardState=blocked-real-proof-required`
- `performsPublish=false`
- `canPublishPublicly=false`
- `canCloseReleaseIssue=false`

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCloseGapDashboard.ps1
```

输出：

- `artifacts/final-release/release-close-gap-dashboard.json`
- `artifacts/final-release/release-close-gap-dashboard.md`

## Dashboard 覆盖的 Gap

当前 dashboard 聚合 5 类剩余 proof gap：

- `owner-authorization`：需要真实 owner approval、发布渠道选择和手动命令 materialization。
- `package-consumer-runtime`：需要 compatible host 上 clean package consumer runtime smoke。
- `linux-runner-proof`：需要真实 Linux x64 runner 产物。
- `real-model-runtime`：需要 Classification/YoloVision 真实模型、labels、输入资产、license、SHA256 和 sample runner log。
- `post-publish verification`：需要真实渠道 package、下载 hash、clean consumer restore/build/probe/smoke 日志和 validator。

## Proof 边界

`package-consumer-runtime` 只能由 `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof` 验证后的真实 clean consumer runtime smoke 产生。`blocked-by-cuda-driver` 是环境阻塞，不是 smoke passed。

Bridge-only package consumer 的 wrapper surface 仍然只是 `WrapperSurfaceEvidenceKind=compile-surface-proof`，`IsRuntimeExecutionProof=False`。`Skipped=True`、`dependency-probe-only`、ONNX Parser diagnostic snapshot、ONNX ParserRefitter diagnostic snapshot、Parser/ParserRefitter copied managed diagnostic snapshot、bridge-only wrapper surface 和 package layout evidence 都不能替代 clean package consumer runtime proof。

`real-model-runtime` 只能由真实模型资产和 sample-run-evidence 产生。Classification/YoloVision 样例 proof 只能晋级 real-model-runtime，不能替代 package-consumer-runtime。YoloVision 范围仍是 YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom，以及 det/cls/seg/obb/pose/sem（det、cls、seg、obb、pose、sem）。

`post-publish verification` 只能来自真实发布渠道、下载后的 nupkg SHA256、clean consumer 根目录、native assets listing、dependency probe log、runtime smoke log，以及 `Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof`。

## 不可替代材料

以下材料可以帮助 owner 执行，但不能写成 proof：

- template
- draft
- runbook
- collection package
- input package
- local feed
- `ProjectReference`
- helper
- build-only
- parse-only
- sidecar-only
- DependencyProbe
- dependency-probe-only
- `Skipped=True`
- `blocked-by-cuda-driver`
- bridge-only package consumer log
- bridge-only wrapper surface
- `WrapperSurfaceEvidenceKind=compile-surface-proof`
- `IsRuntimeExecutionProof=False`
- mismatched log SHA256
- Parser/ParserRefitter diagnostic snapshots
- copied managed diagnostic snapshot
- Windows handoff for Linux proof
- owner-action-required without validator pass

## 与其它产物的关系

- `real-model-and-package-proof-input-package` 提供真实输入 checklist。
- `release-evidence-bundle` 聚合当前证据状态。
- `release-promotion-issue-record` 汇总 owner promotion issue 视图。
- `release-close-preflight` 是 close 前的机器门禁。
- `release-close-gap-dashboard` 把以上内容展开为 owner 可以逐项执行的 gap 清单。

只有真实 proof record、真实日志、真实 hash、真实主机 metadata 和 validator 一起通过后，相关 blocker 才能从 dashboard 中消失。
