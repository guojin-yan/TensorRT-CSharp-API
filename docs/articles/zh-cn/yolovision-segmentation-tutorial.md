# YoloVision Segmentation 多输出实战教程

YOLO instance segmentation（`--task seg`）在 detection 的 box/class/score 之外，还需要让每个保留框携带 mask coefficients，并与独立的 prototype tensor 组合。真正容易出错的不是 engine 能否生成，而是输出 role、channel layout、NMS 后的 source index、sigmoid、阈值、letterbox crop 和原图 resize-back 是否使用同一份模型契约。

本文绑定 `samples/YoloVision` 当前真实实现，从 E 盘资产准备、TensorRtExec build-only、YoloVision preflight/runtime、managed mask compose、JSON/SVG 到 owner evidence 验证形成一条完整路径。

## Seg 与 Sem 不同

| 项目 | Instance segmentation `seg` | Semantic segmentation `sem` |
| --- | --- | --- |
| 结果单位 | 每个 detection 一张实例 mask | 每个像素一个类别 |
| 典型输出 | detection rows + prototype tensor | dense class logits map |
| box/NMS | 使用 | 不使用 |
| 关键 metadata | class count、coefficient count、prototype shape、aux layout | class count、NCHW/NHWC、argmax、palette |
| YoloVision 结果 | `YoloVisionResult.Segmentations` | `YoloVisionResult.SemanticMap` |

不要把 `sem` 的逐像素 class map 当作 prototype，也不要让 `seg` 绕过 detection/NMS 后直接生成无归属的 mask。

## 当前实现边界

当前 managed runtime path 已完成：

1. 把实际 TensorRT outputs 复制为 pointer-free `YoloRuntimeOutputTensor`。
2. 将 detection output 解析为 box/class/score，并保留每个 detection 的 `SourceIndex`。
3. 根据 `maskCoefficientCount`、class count、objectness 和 layout 从原 detection row 读取 coefficients。
4. 接受 `[P,H,W]` 或 `[1,P,H,W]` prototype tensor。
5. 执行 coefficient × prototype 的线性组合和数值稳定 sigmoid。
6. 按 `--mask-threshold` 统计 prototype-grid active pixels。
7. 将概率 mask、active/total pixel count、value kind 和统计范围写入 output JSON。
8. 生成最多 `24x24` 采样的概率 mask SVG 预览。
9. 可选使用精确的 `--image` 预处理 metadata，将 prototype probability bilinear 映射到 source-image shape。
10. 可选按显式 detection coordinate space 执行 box crop，并把 source mask 统计写入嵌套 `spatialTransform`。

默认路径仍不会猜测 letterbox crop、box crop 或 resize-back 规则，SVG 只把 prototype probability grid 映射到 detection box。只有 owner 显式传入 `--mask-spatial-transform`、`--mask-coordinate-space` 和 `--image` 时，runner 才使用本次 `YoloImagePreprocessResult` 做逆变换。该通用映射仍要求 owner 验证 exporter-specific mask alignment；没有这些参数时绝不隐式推断。

## 全链路

```mermaid
flowchart LR
    A["Model + labels + image"] --> B["License + SHA256"]
    B --> C["Preprocess tensor on E drive"]
    C --> D["TensorRtExec build-only"]
    D --> E["YoloVision preflight"]
    E --> F["TensorRT multi-output enqueue"]
    F --> G["Detection rows + SourceIndex"]
    F --> H["Prototype P x H x W"]
    G --> I["Coefficient slice after NMS"]
    H --> J["Linear compose + stable sigmoid"]
    I --> J
    J --> K["Threshold stats + prototype SVG"]
    K --> L["Explicit preprocess inverse"]
    L --> M["Optional half-open box crop"]
    M --> N["Source mask JSON/SVG + owner review"]
```

## E 盘案例目录

```text
..\downloads\cases\yolov8n-seg
  models
  labels
  images
  tensors
  engines
  reports
  logs
  overlays
```

仓库不捆绑模型和图片。owner 获取资产时必须保存：

- 模型主页、直接下载地址、许可证和再分发结论。
- 原始权重与导出 ONNX 的 SHA256。
- export 工具版本、opset、dynamic/static shape 和完整命令。
- labels 来源、许可证、行数和 SHA256。
- 输入图来源、许可证、原图尺寸和 SHA256。
- input layout、RGB/BGR、normalize、resize、padding 与 crop 规则。
- detection/prototype tensor name、shape、dtype、layout 和 coefficient count。

可以从 `samples/assets/yolovision-yolov8-seg-candidate.template.json` 和 `samples/assets/yolovision-article-case-pack.json` 的 `yolov8n-seg` case 开始回填。模板只是 owner action 清单，不是下载许可或 runtime proof。

## 模型获取与导出

以 owner 已审核许可的 YOLOv8 segmentation 权重为例，导出骨架是：

```powershell
yolo export `
  model=..\downloads\cases\yolov8n-seg\models\yolov8n-seg.pt `
  format=onnx `
  opset=17 `
  simplify=True `
  dynamic=False `
  imgsz=640
```

本文不固定权重下载 URL，因为来源和许可证可能变化。导出后用 Netron、ONNX 工具或 TensorRtExec binding report 确认真实 tensor 名；命令里的 `images`、`boxes`、`proto` 只是示例，不能覆盖模型事实。

```powershell
Get-FileHash -Algorithm SHA256 ..\downloads\cases\yolov8n-seg\models\yolov8n-seg.pt
Get-FileHash -Algorithm SHA256 ..\downloads\cases\yolov8n-seg\models\yolov8n-seg.onnx
Get-FileHash -Algorithm SHA256 ..\downloads\cases\yolov8n-seg\labels\coco.names
Get-FileHash -Algorithm SHA256 ..\downloads\cases\yolov8n-seg\images\input.ppm
```

## 输出 Role 合同

`YoloRuntimeOutputRoleResolver` 支持显式 role map 和专用参数：

```text
--output-role-map boxes:det,proto:mask-prototypes
--detection-output boxes
--mask-prototypes-output proto
```

当前 `DecodeSegmentationRuntimeOutputs` 消费 detection + mask-prototypes 两个 role。Mask coefficients 预期嵌在 detection rows 中；虽然 resolver 能识别独立的 `mask-coefficients` role，当前通用 segmentation decoder 尚不消费独立 coefficient tensor。遇到三输出 exporter 时，不能假称它已支持，应在 owner adapter 中合并 coefficients，或新增受测试保护的显式三输出 decode path。

Prototype 只接受：

- `[P,H,W]`
- `[1,P,H,W]`

`P` 必须与 `--mask-coefficient-count` 一致。元素总数必须等于 `P*H*W`，否则受控失败。

## Detection Row 与 Coefficient

对于 coefficients 嵌在 detection rows 的模型，YoloVision 根据以下信息计算起始 channel：

```text
4 box channels + optional objectness + classCount + mask coefficients
```

当 shape 精确匹配时，runner 可以推断 objectness；无法唯一推断时应显式传 `--has-objectness true|false`。`--aux-channel-start` 可覆盖 coefficient 起点，`--aux-layout auto|channels-first|boxes-first` 控制逐框读取方式。

NMS 后不能使用“结果数组下标”重新取 coefficient。`YoloDetection.SourceIndex` 保留原 row index，`DecodeSegmentationOutputs` 用它选择对应 coefficients，从而维持 box 与 mask 的所有权关系。

## Probability 与阈值

`YoloMaskComposer.ComposeLinearMask` 保留原始线性组合 API，返回 `RawLogits`，因此现有调用者的入口没有删除。runtime segmentation 路径使用新增的 `ComposeProbabilityMask`：

```text
logit[x,y] = sum(coeff[p] * prototype[p,x,y])
probability[x,y] = sigmoid(logit[x,y])
```

sigmoid 对正负输入使用分支计算，避免大幅值指数溢出。`--mask-threshold` 取值必须在 `[0,1]`，默认 `0.5`。阈值进入 `YoloMultiOutputMetadata`、preflight、`YoloSegmentationMask`、JSON 和 SVG 预览；无效值受控报错。

## 显式 Source-Image 空间变换

空间变换是 opt-in，不是 segmentation 默认行为：

```text
--mask-spatial-transform
--mask-coordinate-space model-input|normalized
--mask-crop-to-box true|false
--image <source.bmp|source.ppm>
```

`model-input` 表示 detection center/width/height 已经是模型输入像素；`normalized` 表示它们位于 normalized coordinate space，变换前会分别乘以 target width/height。不存在 `auto` 值，因为从外部 tensor 猜测坐标空间会产生看似合理但不可审计的 mask。

对每个 source-image pixel center，变换使用本次预处理的 `PadX/PadY` 和 `ResizedWidth/SourceWidth`、`ResizedHeight/SourceHeight` effective scale 映射回 model input，再映射到 prototype grid 并执行 bilinear sampling。effective scale 从真实取整后的 resized dimensions 推导，避免理想等比 scale 与栅格取整产生偏差。最终 mask shape 固定为 `[sourceHeight,sourceWidth]`。启用 crop 时，box 使用半开栅格边界：

```text
[left, right) x [top, bottom)
```

因此 width 为 2、高度为 2 的整数像素框最多覆盖 `2*2` 个整数 pixel centers，不会把 right/bottom 再多包含一行或一列。box 在输出 JSON 中同时逆变换为 source-image coordinate。

空间变换拒绝以下输入：缺少 `--image`、缺少 coordinate space、非正 resize scale、padding/resized shape 超出 model input、非有限 detection coordinate 或负 box size。外部 `--input-data` 即使数值来自图片，也不能提供可信的 `YoloImagePreprocessResult`，因此不能启用该路径。

## 预处理

先固定输入 tensor：

```powershell
dotnet run --project .\samples\YoloVision -- `
  --preprocess-only `
  --image ..\downloads\cases\yolov8n-seg\images\input.ppm `
  --preprocessed-output ..\downloads\cases\yolov8n-seg\tensors\input-fp32.bin `
  --input-shape 1x3x640x640 `
  --tensor-layout NCHW `
  --color-order RGB `
  --resize letterbox
```

这一步只证明预处理产物可复核。保存 source image hash、preprocessed tensor hash、scale、padX、padY、normalize scale 和 element count。若 exporter 使用不同 resize/crop 规则，应使用 owner-approved pipeline，不能为了复用命令强行改成 letterbox。

## TensorRtExec Build-Only

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx ..\downloads\cases\yolov8n-seg\models\yolov8n-seg.onnx `
  --saveEngine ..\downloads\cases\yolov8n-seg\engines\yolov8n-seg.plan `
  --minShapes images:1x3x640x640 `
  --optShapes images:1x3x640x640 `
  --maxShapes images:4x3x640x640 `
  --fp16 `
  --workspace 1GiB `
  --buildOnly `
  --exportReport ..\downloads\cases\yolov8n-seg\reports\build-report.json
```

`--exportReport` 才是 build report 参数；旧文中的 `--exportProfile` 属于 profile artifact，不应用来替代 build report。build-only 证明 parser/builder/serialization 路径，不证明多输出运行或 mask 正确。

## YoloVision Preflight

```powershell
dotnet run --project .\samples\YoloVision -- `
  --preflight `
  --model ..\downloads\cases\yolov8n-seg\models\yolov8n-seg.onnx `
  --labels ..\downloads\cases\yolov8n-seg\labels\coco.names `
  --image ..\downloads\cases\yolov8n-seg\images\input.ppm `
  --input-shape 1x3x640x640 `
  --family v8 --task seg `
  --class-count 80 `
  --output-role-map boxes:det,proto:mask-prototypes `
  --mask-coefficient-count 32 `
  --mask-threshold 0.5 `
  --mask-spatial-transform `
  --mask-coordinate-space model-input `
  --mask-crop-to-box true `
  --aux-layout boxes-first `
  --preflight-report ..\downloads\cases\yolov8n-seg\reports\preflight.json
```

检查 `yolovision-preflight.v1`、`proofClassification=precheck`、`metadata.maskCoefficientCount=32`、`metadata.maskThreshold=0.5`、`spatialTransform.requested=true`、coordinate space 和 `requiresImagePreprocessMetadata=true`，并确认所有 runtime execution/promotion flag 为 false。preflight 只记录 intent，不生成最终 mask。

## YoloVision Runtime

```powershell
dotnet run --project .\samples\YoloVision -- `
  --model ..\downloads\cases\yolov8n-seg\models\yolov8n-seg.onnx `
  --labels ..\downloads\cases\yolov8n-seg\labels\coco.names `
  --image ..\downloads\cases\yolov8n-seg\images\input.ppm `
  --preprocessed-output ..\downloads\cases\yolov8n-seg\tensors\input-fp32.bin `
  --input-shape 1x3x640x640 `
  --family v8 --task seg `
  --class-count 80 `
  --layout auto --has-objectness auto `
  --nms-mode class-aware --confidence 0.25 --iou-threshold 0.45 `
  --output-role-map boxes:det,proto:mask-prototypes `
  --mask-coefficient-count 32 --mask-threshold 0.5 `
  --mask-spatial-transform `
  --mask-coordinate-space model-input `
  --mask-crop-to-box true `
  --aux-layout boxes-first `
  --output-json ..\downloads\cases\yolov8n-seg\reports\output.json `
  --visualization-svg ..\downloads\cases\yolov8n-seg\overlays\mask-preview.svg
```

待采集的真实日志应包含 `Profile Family=v8 Task=seg`、external input、两个 output tensor、`Segmentations=...`、postprocess summary 和 expected real-log success marker。单输出 diagnostic、synthetic tensor 或缺少 prototype 时不能写成 mask runtime 通过。

## Output JSON 语义

每条 segmentation prediction 现在包含：

- `maskShape`：`[H,W]` prototype mask shape。
- `maskPixelCount`：达到阈值的 active prototype-grid pixel 数。
- `maskTotalPixelCount`：`H*W`。
- `maskThreshold`：本次 decode 使用的概率阈值。
- `maskValueKind`：`probability` 或兼容 raw API 的 `raw-logits`。
- `maskPixelCountScope=prototype-grid-before-crop-resize`：明确统计尚未经过模型特定 crop/resize-back。
- 可选 `spatialTransform`：记录 applied flag、显式 coordinate space、crop/interpolation、source/target/resized shape、padding、scale、最终 mask shape/count/threshold、source box、固定 scope 和 boundary。

原始 prototype-grid 字段不会被 source-image 字段覆盖。两层统计同时存在，便于定位问题发生在 compose 阶段还是 inverse-transform/crop 阶段。`spatialTransform.finalMaskScope` 固定为 `source-image-after-explicit-preprocess-inverse-and-optional-box-crop`。

示例位于 `samples/YoloVision/examples/yolovision-output-seg.example.json`，schema 位于 `samples/YoloVision/yolovision-output.schema.json`。output report 还复制 output tensor shape/value SHA256 和 pointer-free binding metadata，但示例中的 synthetic input 与空 hash 仍不是 runtime proof。

## SVG 预览语义

默认 `YoloVisionVisualizationWriter` 读取真实 prototype mask probability，按阈值选择单元格，并限制为最多 `24x24` 采样以控制 SVG 大小。每个单元格带 `data-mask-cell="true"`，opacity 随 probability 变化。

启用显式空间变换后，SVG 使用 source-image width/height，最多采样 `48x48` 的最终 mask，active cell 带 `data-spatial-mask-cell="true"`，并绘制逆变换后的 source box。它证明显式 preprocess inverse 的结果被消费，但仍是 source-tree diagnostic，不会把输入原图像素嵌入 SVG；正式文章仍需 owner 对照原图和 exporter 参考实现。

## 严格验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionOutputReport.ps1 `
  -InputPath ..\downloads\cases\yolov8n-seg\reports\output.json `
  -OutputPath ..\downloads\cases\yolov8n-seg\reports\output-validation.json `
  -Strict
```

validator 会检查：

- `maskShape` 为两个正维度，且乘积等于 `maskTotalPixelCount`。
- `0 <= maskPixelCount <= maskTotalPixelCount`。
- threshold 位于 `[0,1]`。
- value kind 与 pixel-count scope 合法。
- 可选 spatial transform 必须 `applied=true`，coordinate space 与 bilinear interpolation 合法。
- `finalMaskShape` 乘积必须等于 total，且 active 不超过 total、threshold 位于 `[0,1]`。
- final scope/boundary 必须保持 source-image 与 exporter-validation 语义。
- 存在 spatial transform 时，input 必须是 `preprocessed-image-tensor`，并携带 `image` 与 `letterbox` metadata。
- task/output/boundary/hash 字段没有把报告晋级为 proof。

真实候选还需运行 `eng/Test-YoloVisionRealAssetOwnerProofInput.ps1 -Strict`、owner importer 和 `eng/Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog`。

## 故障排查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| metadata 为 null | 未提供 positive coefficient count | 增加 `--mask-coefficient-count` |
| 找不到 prototype | role/name 推断不匹配 | 显式 `--output-role-map` 或 `--mask-prototypes-output` |
| coefficient range 越界 | class count/objectness/aux start 错 | 核对 row channel count，显式相关参数 |
| prototype shape 拒绝 | 不是 `[P,H,W]` / `[1,P,H,W]` | 转换 exporter output 或新增受控 layout 实现 |
| mask 全黑/全白 | coefficient、prototype、sigmoid 或 threshold 错 | 对比 raw logits/probability，调整 owner-confirmed threshold |
| mask 与框不对应 | NMS 后 row index 丢失 | 确认使用 `YoloDetection.SourceIndex` 路径 |
| SVG 轮廓粗糙 | 有界 `24x24` 或 `48x48` diagnostic sampling | 对 source mask 做原图 overlay 与定量对照 |
| 空间变换拒绝启动 | 缺少 `--image` 或 coordinate space | 使用真实图片预处理链并显式声明坐标空间 |
| box 多一行/列 | 外部参考实现使用闭区间 | 对齐 `[left,right) x [top,bottom)` 半开语义 |
| 原图 mask 偏移 | exporter alignment 与通用 inverse 不一致 | 对比 scale/pad/pixel center，并增加 owner adapter |
| validator 通过但仍不可晋级 | 只有结构化报告或 synthetic input | 补真实 log/hash/host metadata/owner review |

## 代码入口

- `samples/YoloVision/YoloSampleRunner.cs`：runtime role 路由、coefficient slice、prototype shape 与 mask compose。
- `samples/YoloVision/YoloMaskComposer.cs`：raw linear API、稳定 sigmoid 和 probability compose。
- `samples/YoloVision/YoloSegmentationMask.cs`：value kind、threshold、active pixel count。
- `samples/YoloVision/YoloSegmentationSpatialTransform.cs`：显式 coordinate space、bilinear inverse、半开 box crop 和 source mask。
- `samples/YoloVision/YoloRuntimeOutputRoleResolver.cs`：output role、aux metadata 和 `--mask-threshold`。
- `samples/YoloVision/YoloVisionOutputReport.cs`：prototype 与 source-image spatial mask report 字段。
- `samples/YoloVision/YoloVisionVisualizationWriter.cs`：有界 prototype/source-image 概率网格 SVG。
- `eng/Test-YoloVisionOutputReport.ps1`：结构和数值一致性 validator。

## Proof Boundary

以下材料不得替代真实模型证明：build-only、parse-only、preflight、synthetic input、single-output diagnostic、示例 JSON、SVG/screenshot、sidecar-only、TensorRtExec report、OnnxToEngine report、local feed、ProjectReference、direct `.nupkg`、readonly diagnostics 和 `blocked-by-cuda-driver`。

`real-model-runtime` 候选需要 owner-approved 模型/labels/图片、许可证、预处理 tensor、真实两个 output、显式 coordinate space、crop/resize-back 规则、output JSON、run log、全部 SHA256、host metadata 和人工 mask review。通用 spatial transform 仍需验证 exporter-specific alignment。它不是 `package-consumer-runtime`；后者需要仓库外 clean consumer 从目标包来源 restore/build/run。

## 发布前检查清单

- [ ] 模型、labels、图片来源、许可证和再分发结论已审核。
- [ ] 资产全部位于 E 盘 case workspace，未散落到 C 盘。
- [ ] input/detection/prototype tensor 名、shape、dtype 和 layout 已确认。
- [ ] class count、objectness、coefficient count/start/layout 已确认。
- [ ] `P` 与 coefficient count 一致，prototype shape 合法。
- [ ] mask threshold 进入 preflight、runtime、JSON 和 review。
- [ ] active/total pixels 与 shape 数学一致。
- [ ] source-image transform 明确使用 `--image`、coordinate space、crop flag 和半开边界。
- [ ] owner 已对照 exporter 参考实现验证 bilinear、pixel center、letterbox inverse 和最终 mask。
- [ ] build report、preflight、output JSON、SVG、final overlay、run log 和 hash 已归档。
- [ ] output validator 与 sample-run evidence validator 均通过。
- [ ] 没有把本地结果写成公开 package、发布批准或 post-publish proof。
