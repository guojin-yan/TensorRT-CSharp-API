# YoloVision YOLOv8n OBB Angle 输出指南

## 适用读者

本文适合遥感、文档、工业检测等旋转框任务的开发者，也适合需要审核 angle unit、angle range、坐标方向和 rotated NMS 是否一致的模型维护者。

读者应理解普通 detection 的 box/NMS，并能够使用四点坐标验证不同工具之间的角度约定。

## 解决问题

本文解决：

1. rotated box 的 `cx,cy,w,h,angle` 如何解释。
2. degree、radian 和归一化角度如何避免混用。
3. angle range、顺逆时针方向、宽高交换与四点坐标如何验证。
4. rotated NMS 和普通 NMS 的边界。
5. 如何建立 OBB 真实模型运行证据而不误用 build-only 或截图。

## 背景与场景

本文面向需要在 `samples/YoloVision` 中接入 YOLOv8n-obb 或其他 rotated bounding box 模型的开发者。OBB 比普通 detection 多了 angle 字段和几何约定，一旦角度单位、范围或坐标顺序没有写清楚，输出很容易“看起来有框但方向全错”。

本文给出模型来源、ONNX 导出、TensorRtExec build-only、YoloVision 运行、angle metadata 和 proof boundary 的完整文章结构。它是发布文章和真实资产回填模板，不是 runtime proof。

当前仓库已经完成一个受审计的 source-tree `real-model-runtime` 案例：官方 YOLOv8n-obb 单输出为 `output0:[1,20,21504]`，channel 19 是 radians angle。托管路径支持内嵌或独立 angle，并使用 Ultralytics 兼容的 probabilistic-IoU rotated Fast-NMS。该案例的 430,080 个 raw values 全部通过 ONNX Runtime 对照，40 个旋转框的最小几何 rotated IoU 为 `0.997781`；这不自动证明其他 exporter 的 angle/width-height 合同。

## 适用场景

OBB 常用于遥感、文档、工业检测和需要旋转框的视觉任务。YOLOv8n-obb 常见输入尺寸为 `1x3x1024x1024`，labels 可能来自 DOTA 或自定义数据集。owner 必须确认 labels 和模型许可证是否允许公开。

文章建议配一张 rotated box overlay，并同时展示 axis-aligned box 与 rotated box 的差异，帮助读者理解 angle 字段。

## 操作路径

1. 获取模型、labels 和输入图片，记录来源、许可证与 SHA256。
2. 固定导出工具、opset、dynamic shape 和 imgsz。
3. 确认 box/angle tensor、field offset、angle unit 和 range。
4. 使用 TensorRtExec 构建 engine 并保存 build-only report。
5. 生成真实输入 tensor，记录 letterbox 与坐标空间。
6. 使用 YoloVision 显式声明 OBB output role 或 angle field offset。
7. 执行 score filtering、rotated box decode 和 rotated NMS。
8. 输出中心点角度格式与四点坐标格式，保存 JSON、overlay、日志和 SHA256。

## 模型与许可证

owner 需要记录：

- 模型权重来源和许可证。
- ONNX 文件 SHA256。
- labels 文件 SHA256。
- 输入图片来源、许可证和 SHA256。
- angle 单位、范围和方向约定。
- rotated NMS 策略。

不要仅写“angle”。至少要说明它是 degree、radian 还是归一化值，范围是 `[-90, 90)`、`[0, 180)`、`[-pi/2, pi/2)` 还是模型自定义范围。

## 导出 ONNX

```powershell
yolo export model=.\models\yolov8n-obb.pt format=onnx opset=12 dynamic=True simplify=True imgsz=1024
```

导出后记录：

- input tensor name。
- detection output tensor name。
- angle output tensor name 或 angle field offset。
- rotated box layout，例如 `cx,cy,w,h,angle,score,class`。
- angle unit 和 coordinate space。

如果 angle 和 box 在同一输出中，使用 `--aux-channel-start <offset>` 和 `--aux-layout` 明确 field offset；如果分开输出，则用 `--output-role-map boxes:det,angles:obb-angle` 显式声明。

## TensorRtExec Build-Only

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\yolov8n-obb.onnx `
  --saveEngine .\models\yolov8n-obb.plan `
  --minShapes images:1x3x1024x1024 `
  --optShapes images:1x3x1024x1024 `
  --maxShapes images:2x3x1024x1024 `
  --fp16 `
  --buildOnly `
  --exportReport .\models\yolov8n-obb-build-report.json
```

build-only report 可以证明 engine 构建参数和输出 binding，但不能证明 angle decode 或 rotated NMS 正确。

## YoloVision 离线 Preflight

先检查 OBB 的输出角色和输入资产，再进行真实运行：

```powershell
dotnet run --project .\samples\YoloVision -- --model .\models\yolov8n-obb.onnx --labels .\models\dota.names --input-data .\models\yolov8n-obb-fp32.bin --input-shape 1x3x1024x1024 --family v8 --task obb --class-count 15 --layout channels-first --aux-channel-start 19 --aux-layout channels-first --angle-radians --preflight --preflight-report .\models\yolov8n-obb-preflight.json
```

预检报告必须保持 `yolovision-preflight.v1`/`precheck` 边界，且不执行 TensorRT、parser、engine build 或 inference；它不能替代 angle metadata 和真实 OBB 运行证据。

## YoloVision 运行

```powershell
dotnet run --project .\samples\YoloVision -- `
  --model .\models\yolov8n-obb.onnx `
  --labels .\models\dota.names `
  --input-data .\models\yolov8n-obb-fp32.bin `
  --input-shape 1x3x1024x1024 `
  --family v8 `
  --task obb `
  --class-count 15 --layout channels-first `
  --aux-channel-start 19 --aux-layout channels-first --angle-radians
```

真实日志至少应包含：

- `Profile Family=v8 Task=obb`
- `InputSource=external`
- `OrientedBoxes=...`
- `Postprocess Task=obb`
- owner 提供的 YoloVision 成功标记日志行

如果只输出普通 `Detections=...` 而没有 OBB 字段，说明 metadata 或 decode path 还没有完成，不能晋级为 OBB runtime proof。

## Angle 输出解释

建议输出 JSON 包含：

- `rotatedBoxLayout`
- `angleUnit`
- `angleRange`
- `coordinateSpace`
- `centerX`
- `centerY`
- `width`
- `height`
- `angle`
- `corners`
- `rotatedNmsMode`
- `className`
- `score`

可视化时建议同时输出四点坐标和中心点加角度两种格式。这样即使不同工具对 angle 方向约定不同，owner 也可以通过四点坐标复核。

## 可复用资产目录与完整验证

建议为 YOLOv8n-obb 建立独立的 E 盘 case workspace，把遥感/工业图片、DOTA 或自定义 labels、预处理 tensor、engine、报告和日志分开保存，避免模型和临时包落到系统盘：

..\downloads\cases\yolov8n-obb\models
..\downloads\cases\yolov8n-obb\labels
..\downloads\cases\yolov8n-obb\images
..\downloads\cases\yolov8n-obb\tensors
..\downloads\cases\yolov8n-obb\engines
..\downloads\cases\yolov8n-obb\reports
..\downloads\cases\yolov8n-obb\logs

从 `samples/assets/yolovision-yolov8-obb-candidate.template.json` 开始回填 `model.sourceUrl`、`model.downloadUrl`、`model.license`、`model.sha256`、`labels.sha256`、`input.imageSha256`、`input.preprocessedTensorSha256`，以及 `outputMetadata.outputRoleMap`、`outputMetadata.boxFormat`、`outputMetadata.rotatedBoxLayout`、`outputMetadata.angleUnit`、`outputMetadata.coordinateSpace` 和 `outputMetadata.postprocessMetadata.angleRange`。`rotatedNmsMode`、顺逆时针方向、宽高交换规则和四点 corner 顺序也要和导出说明、运行日志一起记录。

模型、labels、原图、预处理 tensor、engine、build report、preflight report、output JSON、overlay SVG 和 run log 分别计算 SHA256：

```powershell
Get-FileHash -Algorithm SHA256 ..\downloads\cases\yolov8n-obb\models\yolov8n-obb.onnx
Get-FileHash -Algorithm SHA256 ..\downloads\cases\yolov8n-obb\labels\dota.names
Get-FileHash -Algorithm SHA256 ..\downloads\cases\yolov8n-obb\images\airplane.ppm
Get-FileHash -Algorithm SHA256 ..\downloads\cases\yolov8n-obb\tensors\airplane-fp32.bin
```

先只做预处理，确认 1024 输入、RGB、NCHW 和 letterbox 记录一致：

```powershell
dotnet run --project .\samples\YoloVision -- --preprocess-only --image ..\downloads\cases\yolov8n-obb\images\airplane.ppm --preprocessed-output ..\downloads\cases\yolov8n-obb\tensors\airplane-fp32.bin --input-shape 1x3x1024x1024 --tensor-layout NCHW --color-order RGB --resize letterbox
```

运行时保留显式 OBB angle role、单位和输出产物：

```powershell
dotnet run --project .\samples\YoloVision -- --model ..\downloads\cases\yolov8n-obb\models\yolov8n-obb.onnx --labels ..\downloads\cases\yolov8n-obb\labels\dota.names --input-data ..\downloads\cases\yolov8n-obb\tensors\airplane-fp32.bin --input-shape 1x3x1024x1024 --family v8 --task obb --class-count 15 --layout channels-first --aux-channel-start 19 --aux-layout channels-first --angle-radians --nms-mode class-aware --output-json ..\downloads\cases\yolov8n-obb\reports\yolov8n-obb-output.json --visualization-svg ..\downloads\cases\yolov8n-obb\reports\yolov8n-obb-output.svg
```

当前 `yolovision-output.v1` 的 OBB prediction 至少记录 `center.x`、`center.y`、`size.width`、`size.height`、`angle`、`angleUnit`、`angleRange`、`classId`、`className` 和 `score`。程序输出的 `angleUnit` 是 `radian`，`angleRange` 仍是 `owner-record-required`；`corners`、旋转方向和 source-image 坐标反变换仍需 owner 依据模型文档或 golden output 复核。rotated NMS 已由 probabilistic IoU 测试与官方案例覆盖，但不能从一张 SVG 截图推断其他模型也兼容。输出 JSON 还应关联 model/image/tensor/run log hash。

建议按以下顺序验证：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionOutputReport.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionRealAssetCandidate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1
```

只有 owner 回填真实模型、labels、输入 hash、angle unit/range、coordinate space、rotated NMS 说明、`YoloVision Passed=True`、stdout/stderr summary、run log hash 并通过 validator 后，才能形成 `real-model-runtime` 候选。TensorRtExec build-only、preflight、rotated overlay、截图、local feed 和 direct `.nupkg` 仍不是 `package-consumer-runtime` proof。

## 代码与文件入口

- `samples/YoloVision/YoloSampleRunner.cs`：内嵌/独立 angle 路由、严格 channel 合同与 `SourceIndex` 绑定。
- `samples/YoloVision/YoloObbDecoder.cs`：angle 转换、probabilistic IoU 和 rotated Fast-NMS。
- `samples/YoloVision/YoloVisionOutputReport.cs`：中心点、尺寸、radian 与 proof boundary 输出。
- `samples/YoloVision/yolovision-task-output-contract.json`：OBB 输出角色契约。
- `samples/YoloVision/Program.cs`：`--task obb`、`--obb-angle-output` 与 role map 参数。
- `eng/Test-YoloVisionRealAssetCandidate.ps1`：模型、输入、输出和证据 hash 验证。

## 图示建议

建议至少包含：

1. 模型输出 tensor 与 angle field 的 Netron 截图。
2. degree/radian 与不同 angle range 的对照图。
3. axis-aligned box 与 rotated box 的叠加对比。
4. 中心点角度格式和四点坐标 JSON。
5. rotated NMS 前后候选数量与 overlay。

截图不能替代实际 angle metadata、输出 JSON、run log、SHA256 和 validator。

## 常见问题

如果框的位置正确但角度不对，检查 angle unit 和 range；如果角度正确但框尺寸错，检查宽高是否交换；如果 NMS 后框过多，检查 rotated NMS 是否真的启用；如果可视化和 JSON 不一致，优先信任 JSON 并修复 overlay 坐标转换。

## 边界说明（Proof Boundary）

本文、rotated box overlay、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics、sidecar-only report、template、dry-run、build-only、screenshot、local feed、ProjectReference、direct `.nupkg` 都不是 runtime proof。

OBB runtime proof 需要真实模型、labels、图片、preprocessed tensor、run log、输出 JSON、SHA256、host metadata、许可证说明和 owner review。公开包或 post-publish proof 仍属于独立 release close lane。

## Owner Backfill Checklist

- 填写 angle unit、angle range、coordinate space 和 rotated box layout。
- 保存模型、labels、图片、preprocessed tensor 和 run log SHA256。
- 保存 TensorRtExec build-only report，但不要把 report 当 proof。
- 保存 YoloVision 运行日志和 owner 提供的成功标记日志行：`YoloVision Passed=True`。
- 提供至少一个 rotated box JSON 片段和可公开 overlay，或说明不能公开。
- 由 owner 明确 review angle 方向和四点坐标是否可信。

## 下一步

完成 YOLOv8n-obb 后，应覆盖 DOTA 与自定义数据集、degree/radian、不同 angle range、宽高交换和 rotated NMS 实现，并增加四点坐标 golden output。不同 YOLO 版本必须记录实际 layout，不能只复用 YOLOv8 的字段假设。

随后在 clean package consumer 中重复真实 OBB 输入运行；source-tree `real-model-runtime`、YoloVision matrix 和 TensorRtExec report 都不能替代 package-consumer-runtime 或 post-publish proof。
