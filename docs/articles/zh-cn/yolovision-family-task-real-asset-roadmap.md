# YoloVision 跨家族真实资产路线图

`applications/YoloVision` 的目标是成为统一的 YOLO-family 示例入口：同一个 runner 覆盖 YOLOv5/v6/v7/v8/v9/v10/v11/v26/custom，并覆盖 det、cls、seg、obb、pose、sem 等任务。

本路线图对应 `samples/assets/yolovision-family-task-real-asset-roadmap.json`。它是案例素材和 owner backfill 工作表，不是 `real-model-runtime` proof，不是 `package-consumer-runtime` proof，也不是 post-publish proof。

## 总体证据链

每个真实案例至少需要四段证据：

1. 模型来源：source URL、license、export command、opset、ONNX SHA256。
2. 输入来源：labels、class count、input image 或 fp32 tensor、preprocess contract、SHA256。
3. 构建报告：`applications/TensorRtExec` build-only report、engine path、report SHA256。
4. 运行证据：`applications/YoloVision` run command、stdout/stderr summary、run log SHA256、`YoloVision Passed=True`、输出 tensor roles、阈值和 owner review。

`TensorRtExec` report、YoloVision matrix、文章、截图、sidecar-only report 都只能作为辅助材料。没有真实模型、真实输入、日志、hash 和 validator 输出时，不能写成 real runtime proof。

## 家族路线图

| Family | Priority | Tasks | Recommended assets | Article angle |
| --- | --- | --- | --- | --- |
| YOLOv5 | tutorial-candidate | det, seg, cls | `yolov5s`, `yolov5s-seg` | detection migration, segmentation metadata |
| YOLOv6 | tutorial-candidate | det | `yolov6n`, `yolov6s` | detector export and objectness/output layout |
| YOLOv7 | tutorial-candidate | det, pose | `yolov7-tiny`, pose variant | detection export, pose keypoint metadata |
| YOLOv8 | ready-template-batch | det, cls, seg, obb, pose, sem | `yolov8n`, `yolov8n-seg`, `yolov8n-pose`, `yolov8n-obb`, `yolov8n-cls` | all-task tutorial series |
| YOLOv9 | article-planning | det, seg | owner-approved YOLOv9 export | export compatibility and segmentation metadata |
| YOLOv10 | managed-end-to-end-decoder-ready | det | YOLOv10n/s | `[1,N,6]` xyxy/score/classId decode, no second NMS, owner runtime proof pending |
| YOLOv11 | article-planning | det, cls, seg, obb, pose | YOLO11 task variants | output roles and task matrix |
| YOLOv26 | future-family-planning | det, cls, seg, obb, pose, sem | owner-provided concrete model | future family onboarding checklist |
| custom | owner-template | det, cls, seg, obb, pose, sem | owner-provided private/public model | custom YOLO onboarding |

## Command Pattern

构建阶段使用 `TensorRtExec`：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\<model>.onnx `
  --saveEngine .\models\<model>.plan `
  --minShapes images:1x3x640x640 `
  --optShapes images:1x3x640x640 `
  --maxShapes images:4x3x640x640 `
  --fp16 `
  --buildOnly `
  --exportReport .\models\<model>-build-report.json
```

运行阶段使用 `YoloVision`：

```powershell
dotnet run --project .\applications\YoloVision -- `
  --model .\models\<model>.onnx `
  --labels .\models\labels.txt `
  --input-data .\models\<model>-fp32.bin `
  --input-shape 1x3x640x640 `
  --family v8 `
  --task det `
  --layout auto `
  --has-objectness auto `
  --nms-mode class-aware
```

多输出任务必须显式记录 output role map。例如 segmentation、pose、OBB：

```powershell
dotnet run --project .\applications\YoloVision -- `
  --model .\models\yolov8n-seg.onnx `
  --labels .\models\coco.names `
  --input-data .\models\yolov8n-seg-fp32.bin `
  --input-shape 1x3x640x640 `
  --family v8 `
  --task seg `
  --output-role-map boxes:det,proto:mask-prototypes `
  --mask-coefficient-count 32
```

## 文章规划建议

这张路线图可以直接拆成一组公众号/博客文章，但文章不能替代 proof：

- YOLOv5 detection 迁移到 TensorRT C# runner。
- YOLOv6 detector 输出布局和 objectness 解释。
- YOLOv7 pose keypoint metadata 实战。
- YOLOv8n detection 下载、导出、TensorRtExec 构建、YoloVision 运行。
- YOLOv8n-seg mask prototype 与 mask coefficient 后处理。
- YOLOv8n-pose keypoint 输出解析。
- YOLOv8n-obb angle unit 和 rotated box 输出。
- YOLOv8n-cls classification 输出和 labels 校验。
- YOLOv9 / YOLOv11 输出差异与 owner evidence checklist。
- YOLOv10 官方模型到 TensorRT：检查 `[1,N,6]` 输出、专用 decoder 与 no-second-NMS 边界。
- custom YOLO 模型接入模板。

每篇文章都应该包含：模型来源、license 提醒、导出命令、TensorRtExec 构建命令、YoloVision 运行命令、输出解释、常见问题、proof boundary。

## Forbidden Substitutes

以下内容必须保持非 proof：

- support matrix
- tutorial article
- template-only record
- sidecar-only report
- TensorRtExec build-only report
- YoloVision matrix
- screenshot
- skipped run
- blocked-by-cuda-driver
- local feed
- ProjectReference
- direct `.nupkg`

## Owner Action

Owner 回填真实资产后，应运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleAssetManifest.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OnnxEngineBuildEvidenceSidecar.ps1
```

只有真实模型、labels、input、build report、run log、SHA256、stdout/stderr summary、expected evidence lines 和 owner review 全部通过，才能考虑 `real-model-runtime` promotion。`package-consumer-runtime` 仍属于 release proof records，不属于 YoloVision 样例本身。
