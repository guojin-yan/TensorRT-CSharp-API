# YoloVision Pose 单输出内嵌通道与多输出实战教程

Pose 模型在 detection 的 box/class/score 之外，还为每个候选目标输出一组 keypoints。工程上最危险的不是少画一个点，而是 detection 行、NMS 后目标、keypoint 行、坐标空间和 stride 没有使用同一份模型契约。本文绑定 `samples/YoloVision` 的当前实现，从 E 盘资产目录、ONNX build-only、preflight、真实运行、JSON/SVG 到证据回填形成完整路径。

## 当前实现范围

YoloVision 的 managed pose 路径已经支持：

1. 将 TensorRT 输出复制为无指针的 `YoloRuntimeOutputTensor`。
2. 解码官方 YOLOv8n-pose 常见的单输出 `output0:[1,56,8400]`：4 个 box 通道、1 个 person class 通道、17×3 个 keypoint 通道。
3. 继续兼容 detection tensor 与独立 pose keypoint tensor 的旧 API。
4. 对 detection rows 执行 score filtering 和配置的 NMS。
5. 保留每个 detection 的 `SourceIndex`，用原始候选行选择对应 keypoint row。
6. 单输出同时接受 `[1,C,N]` / `[1,N,C]`；独立 keypoint tensor 接受 `[1,N,K*stride]` / `[1,K*stride,N]`。
7. 对内嵌输出要求 detection 前缀与 `K*stride` 精确解释全部通道，额外或错位通道直接失败。
8. 解析 `x,y` 或 `x,y,score`，生成 `YoloPosePrediction`，并写入 JSON/SVG。
9. 对官方 COCO 17 点语义绘制人体骨架；有图片预处理元数据和同尺寸背景时，将 box、关键点和骨架统一逆变换到原图坐标。

COCO 17 点骨架只适用于索引语义与官方 COCO Pose 一致的模型。通用路径不做关键点类别重排；自定义关键点集合仍需调用方提供适配。未提供图片元数据和背景时，SVG 保持模型输入坐标；提供完整 letterbox 合同时才执行原图逆变换。

## 输出合同

| 项目 | 当前要求 | 失败风险 |
| --- | --- | --- |
| 官方单输出 | `[1,56,8400]` / `[1,8400,56]` | detection 前缀与 keypoint slice 错位 |
| 独立 keypoint 输出 | 每个候选一行 `K*stride` | tensor role 或 layout 错误 |
| `K` | `--pose-keypoint-count` 或 `--keypoint-count` | 行宽不匹配 |
| stride | `--keypoint-stride`，默认 3，最小 2 | score 列解释错误 |
| 内嵌起点 | 官方 YOLOv8n-pose 为 `--aux-channel-start 5` | class 与 keypoint 通道混读 |
| layout | `--layout` 与 `--aux-layout` 必须与同一 tensor 一致 | N/C 维交换 |
| coordinate space | exporter/owner 显式记录 | 原图 overlay 偏移 |
| skeleton map | 官方 COCO 17 点使用内置 19 条边；自定义模型需显式适配 | 点位含义或连线错误 |

当 stride 为 2 时，decoder 只读取 `x,y`，并将 score 设为 `1.0`。当 stride 大于等于 3 时，第三个值是 score；额外列目前不会写为 visibility 或其他语义。若 exporter 将 visibility 与 score 分列，必须增加显式 adapter，不能仅把 stride 改成 4 就声称语义完整。

## NMS 后仍按原始行取点

Pose 解码先处理 detection tensor，再遍历保留的 detection。`YoloDetection.SourceIndex` 保存原始候选行号，因此 keypoint row 的选择是：

```text
kept detection -> detection.SourceIndex -> keypointRows[SourceIndex]
```

不能使用 NMS 后结果数组的下标。否则第一个保留目标如果来自原始第 37 行，错误实现会读取 keypoint 第 0 行，box 与人体点位会发生静默串线。

## 已审计官方案例

仓库已记录一个真实 source-tree `real-model-runtime` 案例：

- 权重：Ultralytics `v8.3.0` `yolov8n-pose.pt`，Release asset ID `195719300`。
- 权重 SHA256：`c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212`。
- 来源 commit：`6e43d1e1e5db72afbf686dee6745669bcb124b0a`，许可证 `AGPL-3.0-only`。
- 输入：同 commit 的 `bus.jpg`，本地派生 P6 RGB `bus.ppm`，SHA256 `6cdb4b6728a36516826f9adb9387774a6b5db0a49837d515c9045324e04e8688`。
- ONNX：`images:[1,3,640,640] -> output0:[1,56,8400]`，SHA256 `ed1e8d2d2aeb8a2c66e642a16295a72a2990393e3a3843325537da7e11c8a899`。
- 运行时：TensorRT 10.11.0.33、CUDA 12.9、RTX 3060 Laptop GPU。

获取脚本只在 E 盘工作，不导出、不运行、不发布：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Acquire-YoloV8PoseOfficialAssets.ps1 `
  -PythonPath C:\path\to\python.exe

# 已有资产时严格离线复核
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Acquire-YoloV8PoseOfficialAssets.ps1 `
  -PythonPath C:\path\to\python.exe `
  -Offline
```

manifest 位于 `samples/assets/yolovision-yolov8n-pose-official-assets.json`，轻量运行记录位于 `samples/assets/yolovision-yolov8n-pose-real-model-runtime-evidence.json`。`.pt`、ONNX、图片、engine、reference、tensor、SVG 和日志都不进入仓库。

转换后的 ONNX 统一暂存在仓库外 `E:\GitSpace\TensorRT-CSharp-API-4.0\models\YoloVision\Pose\yolov8n-pose-ultralytics-v8.3.0\yolov8n-pose.onnx`，不上传 GitHub。仓库外三包 `PackageReference` 的完整运行、独立关键点对照与负例见 [YoloVision YOLOv8n Pose 本地包消费教程](yolovision-yolov8n-pose-local-package-consumer-tutorial.md)。

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

不要从模型名称推断输出合同。应先用 ONNX checker、Netron 或 TensorRtExec binding report 确认实际 tensor 名与 shape。官方案例经 ONNX checker 确认为单个 `output0:[1,56,8400]`，不是 `boxes + keypoints` 两个输出。

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

## 单输出与显式 output role

官方单输出不需要伪造第二个 tensor role。主输出自动作为 detection tensor，再由 metadata 声明内嵌 keypoint slice：

```text
--task pose
--class-count 1
--layout channels-first
--has-objectness auto
--keypoint-count 17
--keypoint-stride 3
--aux-channel-start 5
--aux-layout channels-first
```

`5 + 17*3 = 56` 必须精确成立。`--aux-channel-start 6`、多出一个尾部通道、layout 冲突或未知 class count 都会 fail closed。

对于真正返回两个输出的模型，继续使用旧合同：

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
  --keypoint-count 17 --keypoint-stride 3 `
  --aux-channel-start 5 --aux-layout channels-first `
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
  --keypoint-count 17 --keypoint-stride 3 `
  --aux-channel-start 5 --aux-layout channels-first `
  --reference-outputs output0:E:\TensorRtSharpAssets\cases\yolov8n-pose\references\output0.reference.json `
  --reference-abs-tolerance 1.25 --reference-rel-tolerance 0.05 `
  --output-json E:\TensorRtSharpAssets\cases\yolov8n-pose\reports\output.json `
  --visualization-svg E:\TensorRtSharpAssets\cases\yolov8n-pose\overlays\pose-preview.svg `
  *> E:\TensorRtSharpAssets\cases\yolov8n-pose\logs\run.log
```

真实日志至少应包含 `Profile Family=YoloV8 Task=Pose Layout=ChannelsFirst`、`output0:[1,56,8400]`、`ReferenceOutputValidation ... Passed=True`、`Poses=2` 和 `YoloVision Passed=True`。具体数量以实际资产为准，证据记录中的 expected line 必须与采集结果一致。

本次 TensorRT 正例比较了全部 `470400` 个输出值，mismatch 为 `0`，最大绝对误差为 `0.001373291`。独立 Ultralytics/PyTorch CPU 对照读取同一个 C# tensor，两个 person pose 的 box IoU 分别为 `0.999999` 和 `0.999998`，可见关键点最大原图坐标误差分别为 `0.000095` 和 `0.000126` 像素。

`eng/New-YoloVisionReferenceMutation.py` 将 reference 第 0 个值增加 `10000` 后，运行必须非零退出。本次负例得到 `Mismatches=1`、`FirstMismatch=0`、`YoloVision Passed=False`，证明 reference 门禁不是只记录不阻断。

## JSON 与 SVG 语义

每条 pose prediction 包含 detection `box`、`classId`、`className`、目标 `score`，以及 keypoint 数组中的 `index/x/y/score`。示例位于 `samples/YoloVision/examples/yolovision-output-pose.example.json`。

SVG 会绘制 box、可见 keypoint 圆点和 COCO 人体骨架边。它最多处理 50 个 pose；当 `--image`、预处理元数据与同尺寸 `--visualization-background` 同时存在时，全部几何元素映射回原图坐标。该图适合结果复查和文章展示，但不替代 raw tensor 与独立后处理比较。

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
| 点呈转置或规律跳跃 | `--layout`、`--aux-layout` 与 `[N,C]`/`[C,N]` 是否一致 |
| 报告 auxiliary range 无法解释 | `4 + objectness + classCount + K*stride` 是否等于总通道数 |
| 每个点 score 都为 1 | stride 是否配置为 2，exporter 是否真的没有 score |
| 点整体偏移 | letterbox padding、坐标空间和 resize-back 是否记录 |
| 输出 role 缺失 | tensor 名与 `--output-role-map`/`--pose-keypoints-output` 是否一致 |
| build 成功但无 pose proof | 只有 build-only，没有真实图、run log、JSON 和人工 review |

## 证据回填

一次可晋级候选至少需要 model/labels/image/preprocessed tensor/engine/output JSON/run log 的 SHA256、TensorRtExec build report、preflight、完整命令、host/runtime metadata、stdout/stderr 摘要和 owner review。再运行 `eng/Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog` 校验真实日志。

这条链路最多形成 owner 审核后的 `real-model-runtime` 候选。它不是 `package-consumer-runtime`；后者要求仓库外 clean consumer 从目标包来源 restore/build/run。`blocked-by-cuda-driver`、synthetic input、template、sidecar-only、ProjectReference 或本地包都不能替代真实 pose runtime proof。

## 代码入口

- `samples/YoloVision/YoloRuntimeOutputRoleResolver.cs`：role map、keypoint count/stride 和 layout 参数。
- `samples/YoloVision/YoloSampleRunner.cs`：`DecodeEmbeddedPoseOutput`、独立 tensor 兼容路径、`SourceIndex` 绑定与 keypoint row 路由。
- `eng/Invoke-YoloVisionPoseReference.py`：ONNX Runtime 原始 reference、Ultralytics/PyTorch CPU 参考与结果比较。
- `samples/YoloVision/YoloPoseDecoder.cs`：`x/y/score` 的纯托管解析。
- `samples/YoloVision/YoloVisionOutputReport.cs`：pose JSON prediction。
- `samples/YoloVision/YoloVisionVisualizationWriter.cs`：box 与关键点 SVG。
- `eng/Test-YoloVisionOutputReport.ps1`：输出结构和 proof boundary 校验。

## 收尾清单

- [ ] 权重、labels、图片来源和许可证已审核，资产仅位于 E 盘 case workspace。
- [ ] 单输出或双输出合同、tensor 名、shape、dtype、layout 已确认。
- [ ] K、stride、索引含义、坐标空间、score/visibility 语义已记录。
- [ ] 内嵌输出的 detection 前缀、`--aux-channel-start` 与总通道数可以精确闭合。
- [ ] NMS 后通过 `SourceIndex` 绑定原始 keypoint row。
- [ ] build report 使用 `--exportReport`，preflight 与 runtime 命令已归档。
- [ ] output JSON、SVG、最终原图 overlay、run log 和全部 hash 已归档。
- [ ] owner 已检查点位顺序、人体归属、letterbox 逆变换和异常样本。
- [ ] 没有把 build-only、precheck 或本地结果写成 package-consumer-runtime 或公开发布批准。
