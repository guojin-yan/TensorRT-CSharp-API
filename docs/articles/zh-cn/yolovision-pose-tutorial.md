# YoloVision Pose 多输出实战教程

Pose 模型在 detection 的 box/class/score 之外，还为每个候选目标输出一组 keypoints。工程上最危险的不是少画一个点，而是 detection 行、NMS 后目标、keypoint 行、坐标空间和 stride 没有使用同一份模型契约。本文绑定 `samples/YoloVision` 的当前实现，从 E 盘资产目录、ONNX build-only、preflight、真实运行、JSON/SVG 到证据回填形成完整路径。

## 当前实现范围

YoloVision 的 managed pose 路径已经支持：

1. 将 TensorRT 输出复制为无指针的 `YoloRuntimeOutputTensor`。
2. 通过显式 output role 区分 detection tensor 与 pose keypoint tensor。
3. 对 detection rows 执行 score filtering 和配置的 NMS。
4. 保留每个 detection 的 `SourceIndex`，用原始候选行选择对应 keypoint row。
5. 接受 `[1,N,K*stride]` 或 `[1,K*stride,N]` 的 rank-3 keypoint tensor。
6. 解析 `x,y` 或 `x,y,score`，生成 `YoloPosePrediction`。
7. 将 box、class、score 和 keypoints 写入 `yolovision-output.v1` JSON。
8. 生成 box 加关键点圆点的有界 SVG 预览。

当前通用路径不推断骨架连接、不做关键点类别重排，也不自动把 letterbox 后的关键点逆变换到原图坐标。SVG 会将绝对值直接当作模型画布坐标，将绝对值不超过 `1.5` 的数按 normalized coordinate 缩放。owner 必须确认 exporter 坐标空间，不能依赖这个显示启发式替代模型合同。

## 输出合同

| 项目 | 当前要求 | 失败风险 |
| --- | --- | --- |
| Detection output | box/class/score rows | 目标筛选与 keypoint 行错位 |
| Keypoint output | 每个候选一行 `K*stride` | tensor role 或 layout 错误 |
| `K` | `--pose-keypoint-count` 或 `--keypoint-count` | 行宽不匹配 |
| stride | `--keypoint-stride`，默认 3，最小 2 | score 列解释错误 |
| layout | `--aux-layout boxes-first|channels-first` | N/K 维交换 |
| coordinate space | exporter/owner 显式记录 | 原图 overlay 偏移 |
| skeleton map | owner 侧记录 | 点位含义无法审核 |

当 stride 为 2 时，decoder 只读取 `x,y`，并将 score 设为 `1.0`。当 stride 大于等于 3 时，第三个值是 score；额外列目前不会写为 visibility 或其他语义。若 exporter 将 visibility 与 score 分列，必须增加显式 adapter，不能仅把 stride 改成 4 就声称语义完整。

## NMS 后仍按原始行取点

Pose 解码先处理 detection tensor，再遍历保留的 detection。`YoloDetection.SourceIndex` 保存原始候选行号，因此 keypoint row 的选择是：

```text
kept detection -> detection.SourceIndex -> keypointRows[SourceIndex]
```

不能使用 NMS 后结果数组的下标。否则第一个保留目标如果来自原始第 37 行，错误实现会读取 keypoint 第 0 行，box 与人体点位会发生静默串线。

## E 盘资产目录

建议使用独立 case workspace，避免模型、engine 和临时 tensor 进入 C 盘：

```text
E:\TensorRtSharpAssets\cases\yolov8n-pose
  models
  labels
  images
  tensors
  engines
  reports
  logs
  overlays
```

仓库不捆绑 pose 模型和输入图。owner 获取资产时至少记录：

- 权重主页、直接来源、许可证和再分发结论。
- 原始权重与导出 ONNX 的 SHA256。
- export 工具版本、opset、dynamic/static shape 和完整命令。
- labels 来源、行数、许可证与 SHA256。
- 输入图来源、尺寸、许可证与 SHA256。
- detection/keypoint tensor 名、shape、dtype、layout、K、stride。
- keypoint 索引含义、坐标空间、score/visibility 语义和 skeleton map。

## 模型导出与哈希

以下命令只是 owner 已审核权重的导出骨架，文件名和 tensor 名必须以实际模型为准：

```powershell
yolo export `
  model=E:\TensorRtSharpAssets\cases\yolov8n-pose\models\yolov8n-pose.pt `
  format=onnx `
  opset=17 `
  simplify=True `
  dynamic=False `
  imgsz=640

Get-FileHash -Algorithm SHA256 E:\TensorRtSharpAssets\cases\yolov8n-pose\models\yolov8n-pose.pt
Get-FileHash -Algorithm SHA256 E:\TensorRtSharpAssets\cases\yolov8n-pose\models\yolov8n-pose.onnx
Get-FileHash -Algorithm SHA256 E:\TensorRtSharpAssets\cases\yolov8n-pose\labels\coco.names
Get-FileHash -Algorithm SHA256 E:\TensorRtSharpAssets\cases\yolov8n-pose\images\input.ppm
```

不要从模型名称推断输出合同。应先用 Netron、ONNX 元数据或 TensorRtExec binding report 确认实际的 `images`、`boxes`、`keypoints` 名称与 shape。

## TensorRtExec build-only

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx E:\TensorRtSharpAssets\cases\yolov8n-pose\models\yolov8n-pose.onnx `
  --saveEngine E:\TensorRtSharpAssets\cases\yolov8n-pose\engines\yolov8n-pose.plan `
  --minShapes images:1x3x640x640 `
  --optShapes images:1x3x640x640 `
  --maxShapes images:4x3x640x640 `
  --fp16 `
  --buildOnly `
  --exportReport E:\TensorRtSharpAssets\cases\yolov8n-pose\reports\build-report.json
```

`--exportReport` 是当前真实参数。build-only report 只能证明构建路径和配置被执行，不能证明 keypoint 行与 box 对齐，更不能替代 `real-model-runtime`。

## 显式 output role

推荐同时记录 role map 和专用参数，真实命令至少保留一种明确声明：

```text
--output-role-map boxes:det,keypoints:pose-keypoints
--detection-output boxes
--pose-keypoints-output keypoints
--pose-keypoint-count 17
--keypoint-stride 3
--aux-layout boxes-first
```

名称启发式能识别 `keypoint`、`kpt`、`pose` 等字符串，但它只是保守 fallback。文章案例和 owner evidence 不应依赖启发式，因为 exporter 重命名后可能得到完全不同的 role。

## 离线 preflight

```powershell
dotnet run --project .\samples\YoloVision -- `
  --model E:\TensorRtSharpAssets\cases\yolov8n-pose\models\yolov8n-pose.onnx `
  --labels E:\TensorRtSharpAssets\cases\yolov8n-pose\labels\coco.names `
  --image E:\TensorRtSharpAssets\cases\yolov8n-pose\images\input.ppm `
  --input-shape 1x3x640x640 `
  --family v8 --task pose `
  --class-count 1 `
  --layout auto --has-objectness auto `
  --output-role-map boxes:det,keypoints:pose-keypoints `
  --pose-keypoint-count 17 --keypoint-stride 3 `
  --aux-layout boxes-first `
  --preflight --strict-preflight `
  --preflight-report E:\TensorRtSharpAssets\cases\yolov8n-pose\reports\preflight.json
```

检查报告中的 `schemaVersion=yolovision-preflight.v1`、`proofClassification=precheck`、`poseKeypointCount=17`、`poseKeypointStride=3`、input source exclusivity 和资产 hash。preflight 不打开 TensorRT，不执行 enqueue，所有 runtime/promotion flag 必须保持 false。

## 真实运行与输出

```powershell
dotnet run --project .\samples\YoloVision -- `
  --model E:\TensorRtSharpAssets\cases\yolov8n-pose\models\yolov8n-pose.onnx `
  --labels E:\TensorRtSharpAssets\cases\yolov8n-pose\labels\coco.names `
  --image E:\TensorRtSharpAssets\cases\yolov8n-pose\images\input.ppm `
  --preprocessed-output E:\TensorRtSharpAssets\cases\yolov8n-pose\tensors\input-fp32.bin `
  --input-shape 1x3x640x640 `
  --family v8 --task pose `
  --class-count 1 `
  --layout auto --has-objectness auto `
  --nms-mode class-aware --confidence 0.25 --iou-threshold 0.45 `
  --output-role-map boxes:det,keypoints:pose-keypoints `
  --pose-keypoint-count 17 --keypoint-stride 3 `
  --aux-layout boxes-first `
  --output-json E:\TensorRtSharpAssets\cases\yolov8n-pose\reports\output.json `
  --visualization-svg E:\TensorRtSharpAssets\cases\yolov8n-pose\overlays\pose-preview.svg `
  *> E:\TensorRtSharpAssets\cases\yolov8n-pose\logs\run.log
```

真实日志至少应包含 `Profile Family=v8 Task=Pose`、两个 output tensor、`Poses=`、binding metadata 和 `YoloVision Passed=True`。具体枚举文本以实际日志为准，evidence pack 中的 expected line 必须与采集结果一致。

## JSON 与 SVG 语义

每条 pose prediction 包含 detection `box`、`classId`、`className`、目标 `score`，以及 keypoint 数组中的 `index/x/y/score`。示例位于 `samples/YoloVision/examples/yolovision-output-pose.example.json`。

SVG 只画 box 和 keypoint 圆点，不包含人体骨架边。它最多处理 50 个 pose，适合 owner review 和文章截图，但不是像素级正确性 proof。若点位仍在 letterbox/model-input 坐标，必须在 owner adapter 中显式逆变换后再制作最终原图 overlay。

## 输出校验

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionOutputReport.ps1 `
  -InputPath E:\TensorRtSharpAssets\cases\yolov8n-pose\reports\output.json `
  -OutputPath E:\TensorRtSharpAssets\cases\yolov8n-pose\reports\output-validation.json `
  -Strict
```

validator 会检查 task、输入、engine、runtime、output summaries、pose box/keypoints 和 proof boundary。它不会判断人体拓扑、关键点索引含义或原图对齐是否正确，这些仍需要 owner 依据模型合同审核。

## 常见失败

| 表现 | 首先检查 |
| --- | --- |
| `Poses=0` | confidence、class count、objectness、detection layout |
| box 正确但点属于另一个人 | `SourceIndex` 是否保留、keypoint row 是否与候选行同序 |
| 点呈转置或规律跳跃 | `--aux-layout` 与 `[N,C]`/`[C,N]` 是否一致 |
| 每个点 score 都为 1 | stride 是否配置为 2，exporter 是否真的没有 score |
| 点整体偏移 | letterbox padding、坐标空间和 resize-back 是否记录 |
| 输出 role 缺失 | tensor 名与 `--output-role-map`/`--pose-keypoints-output` 是否一致 |
| build 成功但无 pose proof | 只有 build-only，没有真实图、run log、JSON 和人工 review |

## 证据回填

一次可晋级候选至少需要 model/labels/image/preprocessed tensor/engine/output JSON/run log 的 SHA256、TensorRtExec build report、preflight、完整命令、host/runtime metadata、stdout/stderr 摘要和 owner review。再运行 `eng/Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog` 校验真实日志。

这条链路最多形成 owner 审核后的 `real-model-runtime` 候选。它不是 `package-consumer-runtime`；后者要求仓库外 clean consumer 从目标包来源 restore/build/run。`blocked-by-cuda-driver`、synthetic input、template、sidecar-only、ProjectReference 或本地包都不能替代真实 pose runtime proof。

## 代码入口

- `samples/YoloVision/YoloRuntimeOutputRoleResolver.cs`：role map、keypoint count/stride 和 layout 参数。
- `samples/YoloVision/YoloSampleRunner.cs`：detection decode、`SourceIndex` 绑定与 keypoint row 路由。
- `samples/YoloVision/YoloPoseDecoder.cs`：`x/y/score` 的纯托管解析。
- `samples/YoloVision/YoloVisionOutputReport.cs`：pose JSON prediction。
- `samples/YoloVision/YoloVisionVisualizationWriter.cs`：box 与关键点 SVG。
- `eng/Test-YoloVisionOutputReport.ps1`：输出结构和 proof boundary 校验。

## 收尾清单

- [ ] 权重、labels、图片来源和许可证已审核，资产仅位于 E 盘 case workspace。
- [ ] detection/keypoint tensor 名、shape、dtype、layout 已确认。
- [ ] K、stride、索引含义、坐标空间、score/visibility 语义已记录。
- [ ] NMS 后通过 `SourceIndex` 绑定原始 keypoint row。
- [ ] build report 使用 `--exportReport`，preflight 与 runtime 命令已归档。
- [ ] output JSON、SVG、最终原图 overlay、run log 和全部 hash 已归档。
- [ ] owner 已检查点位顺序、人体归属、letterbox 逆变换和异常样本。
- [ ] 没有把 build-only、precheck 或本地结果写成 package-consumer-runtime 或公开发布批准。
