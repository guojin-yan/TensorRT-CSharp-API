# YoloVision 真实资产 Owner 回填包

## 文章定位

`samples/assets/yolovision-real-asset-owner-backfill-pack.json` 是给 release owner 使用的真实资产回填合同。上一轮的 `yolovision-article-case-pack.json` 解决“文章怎么写、命令怎么给、输出怎么解释”，本包进一步要求 owner 把模型来源、许可证、SHA256、YoloVision 离线 preflight report、TensorRtExec build-only report、YoloVision run log、stdout/stderr 摘要和 owner review 填完整。

它仍然是 `owner-action-required` 模板，不是 runtime proof。只有真实字段被 owner 回填，并通过 `eng/Test-YoloVisionRealAssetOwnerBackfillPack.ps1 -Strict` 等校验后，才可能作为 `real-model-runtime` 候选进入后续 release proof 流程。

## 覆盖范围

本包覆盖六个高价值 YOLOv8n case：

- detection：`yolov8n-det`
- segmentation：`yolov8n-seg`
- pose：`yolov8n-pose`
- oriented bounding box：`yolov8n-obb`
- classification：`yolov8n-cls`
- semantic segmentation：`yolov8n-sem`

每个 case 都保留对应文章路径、ONNX 导出命令、YoloVision preflight 命令、TensorRtExec build-only 命令和 YoloVision run 命令。这样 owner 不需要重复翻找文章，只要按 JSON 字段补真实证据即可。

## Owner 必填字段

每个 case 至少需要回填：

- `model.sourceUrl`
- `model.license`
- `model.sha256`
- `model.onnxSha256`
- `labels.license`
- `labels.classCount`
- `labels.sha256`
- `input.imageLicense`
- `input.imageSha256`
- `input.preprocessedTensorSha256`
- `input.preprocessContract`
- `yoloVisionPreflight.reportSha256`
- `yoloVisionPreflight.schemaVersion`（必须是 `yolovision-preflight.v1`）
- `yoloVisionPreflight.proofClassification`（必须是 `precheck`）
- `yoloVisionPreflight.execution.*`（必须全部为 `false`）
- `yoloVisionPreflight.boundary.*`（必须保持不可晋级）
- `tensorRtExec.reportSha256`
- `tensorRtExec.engineSha256`
- `yoloVision.runLogSha256`
- `yoloVision.stdoutSummary`
- `yoloVision.stderrSummary`
- `yoloVision.outputJsonSha256`
- `ownerReview.reviewer`
- `ownerReview.reviewedAtUtc`
- `ownerReview.notes`

SHA256 必须是 64 位十六进制字符串。模板中的 `owner-required`、`owner-required-or-no-stderr` 只表示字段位置，不表示真实证据。

## 执行顺序

1. 下载或导出模型，记录模型来源和许可证。
2. 计算权重、ONNX、labels、输入图片和预处理 tensor 的 SHA256。
3. 先执行每个 case 的 YoloVision `--preflight` 命令，保存 `yolovision-preflight.v1` report 和 SHA256；确认 `proofClassification=precheck` 且没有 runtime 执行。
4. 执行 TensorRtExec build-only 命令并保存 report、engine 和对应 SHA256。
5. 执行 YoloVision run 命令，确保日志中出现 `YoloVision Passed=True`。
6. 保存 stdout/stderr 摘要、run log、output JSON 和 SHA256。
7. 由 owner review 输出结果是否符合模型任务语义。
8. 运行 `eng/Test-YoloVisionRealAssetOwnerBackfillPack.ps1 -Strict`。

## 校验器

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionRealAssetOwnerBackfillPack.ps1 -Strict
```

默认输出：

- `artifacts/yolovision/yolovision-real-asset-owner-backfill-pack-validation.json`

模板状态下，校验器必须保持：

- `validationState=owner-action-required`
- `canPromoteRealModelRuntime=false`
- `canPromotePackageConsumerRuntime=false`
- `canPublishPublicly=false`

如果 owner 误把模板字段标为可 promote，或者把 build-only report、YoloVision matrix、screenshot、sidecar-only report 当成 proof，校验器必须阻止。

## Proof Boundary

本包、六篇文章、YoloVision preflight report、TensorRtExec report、YoloVision matrix、OnnxToEngine report、sidecar-only report、screenshot、template、dry-run、build-only、local feed、ProjectReference、direct `.nupkg`、readonly diagnostics 都不是 runtime proof。

`real-model-runtime` 只属于真实模型、真实输入、真实 YoloVision run log、完整 SHA256、host metadata 和 owner review 都齐全后的候选状态。`package-consumer-runtime`、public package proof 和 post-publish verification proof 仍然属于独立 release close lane，不能由本包替代。

## 与 TensorRtExec Profile 场景的关系

本包把 TensorRtExec profile shape 固定到具体任务：

- classification：`1x3x224x224`
- detection / segmentation / pose：`1x3x640x640`
- OBB：`1x3x1024x1024`

这些 shape profile 能帮助 build/report 可诊断，但它们仍是 build evidence。只有 YoloVision 真实 run log 与 owner review 能进入 real-model-runtime 候选。
