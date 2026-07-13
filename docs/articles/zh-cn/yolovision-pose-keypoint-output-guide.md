# YoloVision YOLOv8n Pose Keypoint 输出指南

## 适用读者

本文适合接入人体姿态、工业关键点或自定义骨架模型的开发者，也适合需要审核 keypoint label map、输入输出和可视化结果是否可复现的维护者。

读者应理解 detection、letterbox 和 NMS 基础，并能够核对模型的 keypoint count、字段顺序和坐标空间。

## 解决问题

本文解决：

1. box、score、class 与 keypoint 字段如何从输出中解析。
2. keypoint count、`x,y,confidence` 或自定义 layout 如何显式描述。
3. 坐标如何从模型输入空间还原到原图。
4. skeleton map、overlay 和输出 JSON 如何进行人工与自动验证。
5. 如何区分 build-only、真实 pose runtime 和 package-consumer proof。

## 背景与场景

本文面向希望在 `samples/YoloVision` 中接入 YOLOv8n-pose 的开发者。Pose 模型通常把 box、score、class 和 keypoint 放在同一组候选里，后处理需要同时处理检测框、关键点坐标、关键点置信度和 letterbox 坐标还原。

本文是一篇可发布的技术文章草稿，也是一份 owner 真实资产回填指南。它不会把文章、模板、matrix、screenshot 或 build-only report 晋级为 runtime proof。

## 适用场景

当你需要验证人体关键点、工业关键点或自定义骨架模型时，可以复用本文结构。对于 YOLOv8n-pose，常见 keypoint count 是 17，每个 keypoint 通常记录 `x,y,confidence`。如果自定义模型包含 visibility 或额外字段，必须在 metadata 中明确写出。

文章建议展示一张原图、一张 keypoint overlay、一段输出 JSON 和一段 keypoint label map。公开前要确认图片和模型许可证。

## 操作路径

1. 获取模型、keypoint label map、输入图片并记录许可证与 SHA256。
2. 固定导出工具版本、opset、dynamic shape 和 imgsz。
3. 确认输出 tensor、candidate layout、keypoint count 与字段顺序。
4. 使用 TensorRtExec 构建 engine 并保留 build-only report。
5. 生成真实 NCHW 输入 tensor并保存 SHA256。
6. 使用 YoloVision 声明 `pose-keypoints` output role 或单 tensor field offset。
7. 执行 detection、NMS、keypoint decode 和 letterbox 坐标还原。
8. 保存输出 JSON、overlay、run log、host metadata 和 owner review。

## 模型与许可证

owner 需要记录：

- 模型权重来源、许可证和 SHA256。
- ONNX 导出命令、opset、dynamic shape 和 SHA256。
- labels 或 keypoint label map 来源与 SHA256。
- 输入图片来源、许可证、SHA256。
- 输出 skeleton 约定是否可公开。

不要把 COCO keypoint 顺序当成所有 pose 模型的默认事实。自定义模型必须提供 keypoint name、index、连接关系和 score 字段说明。

## 导出 ONNX

```powershell
yolo export model=.\models\yolov8n-pose.pt format=onnx opset=12 dynamic=True simplify=True imgsz=640
```

导出后记录：

- 输入 tensor 名和 shape。
- 输出 tensor 名、shape 和 layout。
- keypoint count，例如 17。
- keypoint layout，例如 `x,y,confidence`。
- box 与 keypoint 是否同 tensor。
- 是否包含 graph-side NMS。

## TensorRtExec Build-Only

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\yolov8n-pose.onnx `
  --saveEngine .\models\yolov8n-pose.plan `
  --minShapes images:1x3x640x640 `
  --optShapes images:1x3x640x640 `
  --maxShapes images:4x3x640x640 `
  --fp16 `
  --buildOnly `
  --exportReport .\models\yolov8n-pose-build-report.json
```

这一步用于确认 engine build 和 output binding。它不是 pose runtime proof，因为它没有证明关键点 decode、坐标还原和可视化输出正确。

## YoloVision 运行

```powershell
dotnet run --project .\samples\YoloVision -- `
  --model .\models\yolov8n-pose.onnx `
  --labels .\models\coco.names `
  --input-data .\models\yolov8n-pose-fp32.bin `
  --input-shape 1x3x640x640 `
  --family v8 `
  --task pose `
  --output-role-map boxes:det,keypoints:pose-keypoints `
  --pose-keypoint-count 17
```

真实日志至少应包含：

- `Profile Family=v8 Task=pose`
- `InputSource=external`
- `Poses=...`
- `Postprocess Task=pose`
- owner 提供的 YoloVision 成功标记日志行

如果 keypoint 输出来自单 tensor，也可以在 metadata 中说明 keypoint field offset，而不是伪造第二个 output role。

## Keypoint 输出解释

建议输出 JSON 包含：

- `personBox`
- `personScore`
- `keypointCount`
- `keypointLayout`
- `keypoints[index].name`
- `keypoints[index].x`
- `keypoints[index].y`
- `keypoints[index].score`
- `keypoints[index].visibility`
- `letterboxScale`
- `letterboxPadX`
- `letterboxPadY`

坐标还原应明确从模型输入空间回到原图空间。可视化 overlay 只能作为人工复核材料，不能替代 run log 和 hash。

## 代码与文件入口

- `samples/YoloVision/YoloVisionRuntimePipeline.cs`：pose output role 路由。
- `samples/YoloVision/YoloVisionPoseDecoder.cs`：box 与 keypoint 字段解析。
- `samples/YoloVision/YoloVisionCoordinateMapper.cs`：letterbox 坐标还原。
- `samples/YoloVision/YoloVisionNms.cs`：候选过滤与保留索引。
- `samples/YoloVision/yolovision-task-output-contract.json`：pose 输出契约。
- `samples/YoloVision/Program.cs`：`--task pose`、`--pose-keypoint-count` 和 role map 参数。
- `eng/Test-YoloVisionRealAssetCandidate.ps1`：真实资产与证据字段验证。

## 图示建议

建议准备：

1. Netron 中 pose 输出 tensor 与 shape。
2. keypoint index/name/skeleton map 表格。
3. 原图、letterbox 图和最终 keypoint overlay。
4. 单个 detection 的 box、score 和 17 个 keypoint JSON。
5. 坐标从输入空间还原到原图空间的示意图。

overlay 只能辅助人工检查；runtime proof 仍需要原始 run log、结构化 JSON、SHA256 和 validator。

## 常见问题

如果关键点整体偏移，先检查 letterbox；如果框正确但关键点错位，检查 keypoint field offset；如果只有少数点异常，检查 keypoint label map；如果所有 score 都很低，检查输入归一化、RGB/BGR 和模型输出是否已经 sigmoid。

## 边界说明（Proof Boundary）

本文、keypoint overlay、TensorRtExec report、OnnxToEngine report、YoloVision matrix、readonly diagnostics、template、dry-run、build-only、sidecar-only report、screenshot、local feed、ProjectReference、direct `.nupkg` 都不是 runtime proof。

真实 owner proof 必须包含模型、labels/keypoint map、图片、preprocessed tensor、run log、stdout/stderr summary、SHA256、host metadata、许可证说明和 owner review。`package-consumer-runtime` proof 是另一条 release lane，不能由样例文章替代。

## Owner Backfill Checklist

- 填写 keypoint count、keypoint layout 和 skeleton map。
- 保存模型、labels/keypoint map、输入图片和 preprocessed tensor SHA256。
- 保存 TensorRtExec build-only report。
- 保存 YoloVision run log 和 `YoloVision Passed=True`。
- 记录至少一个关键点输出 JSON 片段，便于人工复查。
- 标注是否允许公开图片和 overlay。

## 下一步

完成 YOLOv8n-pose 后，应增加自定义 keypoint count、不同 skeleton map、单 tensor 与多 tensor 输出布局测试，并覆盖 YOLOv11 等新版本。每个模型都要记录真实 field offset，不能假定所有模型都是 COCO 17 点。

下一阶段再使用仓库外 clean package consumer 重复真实模型运行，补充 package-consumer-runtime；文章和 source-tree 样例本身不能关闭发布 proof blocker。
