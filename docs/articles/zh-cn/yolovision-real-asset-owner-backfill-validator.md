# YoloVision 真实资产 Owner Backfill Validator

YoloVision 的目标不是只跑通一个 YOLOv8n demo，而是把 YOLOv5、v6、v7、v8、v9、v10、v11、v26 以及 det、cls、seg、obb、pose、sem 等任务统一到一套可复用样例里。为了让这条线可以发布、可以复现、可以继续扩展，真实模型资产必须经过 owner backfill validator，而不能只依赖模板文件或口头说明。

本文说明 `eng/Test-YoloVisionRealAssetCandidate.ps1` 的定位、输入字段、晋级条件和禁止替代项。它面向后续补真实模型、补运行截图、补公众号/博客案例文章的维护者。

## 为什么需要这个 validator

YoloVision 的候选资产模板已经从 YOLOv8 detection 与 segmentation 扩展到六条路径：

- `samples/assets/yolovision-yolov8-det-candidate.template.json`
- `samples/assets/yolovision-yolov8-seg-candidate.template.json`
- `samples/assets/yolovision-yolov8-pose-candidate.template.json`
- `samples/assets/yolovision-yolov8-obb-candidate.template.json`
- `samples/assets/yolovision-yolov8-cls-candidate.template.json`
- `samples/assets/yolovision-yolov8-sem-candidate.template.json`

这些模板可以描述模型来源、许可证、输入图片、预处理 tensor、TensorRtExec build 命令和 YoloVision run 命令，但模板本身仍然是 `owner-action-required`。也就是说，它们能告诉维护者该补什么，不能证明真实模型已经在目标机器上跑通。

validator 的第一职责是保护这个边界：

- template 可以通过结构校验。
- template 不能晋级 real-model-runtime。
- YoloVision 候选资产永远不能晋级 package-consumer-runtime。
- 真实 owner 回填必须提供 hash、stdout/stderr 摘要和 `YoloVision Passed=True` 证据。

## 推荐执行方式

在仓库根目录执行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionRealAssetCandidate.ps1 -Strict
```

默认会读取 YOLOv8 det/seg 两个候选模板，并输出：

```text
artifacts/yolovision/yolovision-real-asset-candidate-validation.json
```

如果 owner 后续复制出自己的真实资产记录，可以显式传入：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionRealAssetCandidate.ps1 `
  -InputPath .\samples\assets\owner-yolov8-det-candidate.json `
  -Strict
```

## 最低字段要求

validator 会检查以下结构约束：

- `sampleName=YoloVision`
- `family=YOLOv8`
- `task=det|seg|pose|obb|cls|sem`
- `runtimeProofState=owner-action-required|owner-backfilled|validated-real-model-runtime`
- `canPromotePackageConsumerRuntime=false`
- `proofChecklist.requiredHashes` 必须列出 `modelSha256`、`labelsSha256`、`imageSha256`、`preprocessedTensorSha256`、`runLogSha256`
- `proofChecklist.requiredEvidenceLines` 必须包含 `YoloVision Passed=True`

任务专属 metadata 也必须完整：

- pose：`keypointCount`、`keypointLayout`、`keypointScoreField`
- obb：`angleUnit`、`rotatedBoxLayout`、`coordinateSpace`
- cls：`topK`、`classScoreField`、`labelsRequired`
- sem：`semanticMapShape`、`classMapLayout`、`paletteRequired`

当记录仍是模板时，hash 字段可以保留 `owner-required`。一旦 owner 想把记录晋级为真实候选，模型、labels、测试图片、预处理 tensor 和运行日志都必须填写真实 64 位 SHA256。

## 真实晋级条件

只有同时满足下列条件，记录才允许进入 `validated-real-model-runtime`：

1. `runtimeProofState` 不再是 `owner-action-required`。
2. `proofClassification` 不再是 `template-only`。
3. `canPromoteRealModelRuntime=true`。
4. model / labels / image / preprocessed tensor / run log 都有真实 64 位 SHA256。
5. stdout 和 stderr 摘要不再是 placeholder。
6. evidence lines 中保留 `YoloVision Passed=True`。
7. `canPromotePackageConsumerRuntime=false` 继续保持为 false。

这条规则刻意区分两类证据：YoloVision 真实模型运行可以证明样例路径对某个模型资产可用；package-consumer-runtime 只能来自干净外部 consumer 使用公开包的验证，不能由样例资产模板或本仓库直接引用证明。

## 不能替代的证据

以下内容不能作为真实模型 proof：

- 仅有模板 JSON。
- 仅有 TensorRtExec build-only report。
- 仅有本地 `.nupkg`、local feed、ProjectReference 或 direct `.nupkg`。
- 仅有 README、截图或文章草稿。
- 仅有模型下载链接但没有 SHA256 和运行日志。
- 仅有 `Passed=True` 字符串但缺少 stdout/stderr 摘要与 run log hash。

这些内容可以作为教程或准备材料，但不能关闭 owner-action-required。

## 配图建议

- 一张“候选模板 -> owner backfill -> real-model-runtime -> package-consumer-runtime”的证据梯度图。
- 一张 YoloVision det/seg 运行日志截图，标出 owner 提供的成功标记日志行。
- 一张 JSON 字段标注图，突出 64 位 SHA256 和 stdout/stderr 摘要。

## 下一步

下一批工作应当把 validator 从 YOLOv8 六任务扩展到 YOLOv5、v6、v7、v9、v10、v11、v26。每增加一类真实模型资产，都要同时补样例命令、输出 metadata、validator 测试和文章中的 proof boundary。
