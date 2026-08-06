# YoloVision OBB 单输出与多输出实战教程

OBB（Oriented Bounding Box）在普通 detection 的 center/size/class/score 之外增加旋转角。角度单位、范围、宽高规范化、输出 layout 或 NMS 策略只要有一个与 exporter 不一致，就会出现“中心正确但方向错误”的静默结果。本文绑定 `applications/YoloVision` 当前真实实现，给出从 E 盘资产准备到 JSON/SVG 和 owner evidence 的完整操作路径。

## 当前实现范围

YoloVision 的 managed OBB 路径已经支持：

1. 将 TensorRT 输出复制为无指针 `YoloRuntimeOutputTensor`。
2. 同时支持 detection tensor 加独立 angle tensor，以及 detection prefix 后内嵌一个 angle channel 的单输出合同。
3. 对候选执行 score filtering，并使用 probabilistic-IoU rotated NMS；`class-aware` 与 `class-agnostic` 保持原有 CLI 语义。
4. 保留 detection `SourceIndex`，从同一原始候选行取 angle。
5. 接受 `[1,N,1]` / `[1,1,N]` 独立 angle tensor，以及 `[1,C,N]` / `[1,N,C]` 内嵌 angle 输出。
6. 根据 `--angle-degrees` 或 `--angle-radians` 将角度统一为 `AngleRadians`。
7. 输出 center/size/angle/class/score JSON，并生成旋转矩形 SVG。

旋转 NMS 固定为 Ultralytics 8.4.21 兼容的 covariance/probabilistic IoU 与按分数排序 Fast-NMS：每个候选会与全部更高分的同类候选比较，达到阈值即抑制。普通 detection 仍使用轴对齐 NMS，两条路径没有静默改义。通用 OBB 路径不会自动交换 width/height、折叠 angle 周期或生成 corner polygon；其他 exporter 若有这些规则，仍需显式 adapter 和独立 golden reference。

## 角度合同

| 字段 | 必须明确的内容 |
| --- | --- |
| 原始单位 | degree 或 radian |
| 原始范围 | 例如 `[-90,90)`、`[0,180)`、`[-pi/2,pi/2)` |
| 方向 | 顺时针或逆时针 |
| 起始轴 | x 轴、y 轴或 exporter 特定轴 |
| width/height 规则 | 是否强制 `width >= height`，交换后是否补偿角度 |
| NMS | axis-aligned、rotated-IoU 或 graph 内置 |
| 坐标空间 | normalized、model-input pixels 或 source-image pixels |

`YoloObbDecoder` 负责 degree 到 radian 的单位转换，以及 rotated NMS 使用的 probabilistic IoU：

```text
angleRadians = angleInDegrees ? angle * PI / 180 : angle
```

它不会自动折叠角度范围，也不会交换 width/height。output prediction 始终写 `angleUnit=radian`，而 `angleRange` 保持 `owner-record-required`，防止报告把一个已验证模型的范围泛化到所有 exporter。

## SourceIndex 绑定

OBB 与 Pose、Segmentation 使用相同的所有权原则：NMS 后结果不能按新数组下标读取 auxiliary tensor。`YoloDetection.SourceIndex` 保存原始候选行号，正确关系是：

```text
kept detection -> detection.SourceIndex -> angleRows[SourceIndex][0]
```

如果第一个保留目标来自原始第 12 行，angle 也必须取第 12 行。这个绑定受 managed tests 保护，但 owner 仍需确认 exporter 的 detection 与 angle tensor 在候选维上同序。

## E 盘资产目录

```text
..\downloads\cases\yolov8n-obb
  models
  labels
  images
  tensors
  engines
  reports
  logs
  overlays
```

仓库提供 `eng/Acquire-YoloV8ObbOfficialAssets.ps1` 获取已固定身份的官方案例，但权重、ONNX、DOTA labels、图片、tensor、reference 和日志只写入外层 E 盘 `downloads`，不会进入 Git。其他模型仍由 owner 提供。owner 必须保存：

- 模型主页、直接来源、许可证和再分发结论。
- 权重与 ONNX 的 SHA256、export 工具版本和完整命令。
- labels 与输入图的来源、许可证、尺寸和 SHA256。
- detection/angle tensor 名、shape、dtype、layout。
- 角度单位、范围、方向、起始轴和 width/height 规范。
- NMS 所在位置与算法，尤其是否要求 rotated IoU。

## 已验证官方案例

仓库中的轻量资产合同是 `samples/assets/yolovision-yolov8n-obb-official-assets.json`，真实运行记录是 `samples/assets/yolovision-yolov8n-obb-real-model-runtime-evidence.json`。固定案例为 Ultralytics `v8.3.0` `yolov8n-obb.pt` 与 commit-pinned `boats.jpg`：

```text
images:[1,3,1024,1024]
  -> output0:[1,20,21504]
  = 4 box + 15 DOTA class + 1 angle
auxiliaryChannelStart=19
angleUnit=radians
angleRange=[-pi/4,3pi/4]
```

本机 TensorRT 10.11/CUDA 12.9/RTX 3060 结果：

- 430,080 个 TensorRT raw values 在 `abs=4.25, rel=0.05` 的统一比较器下与 ONNX Runtime CPU reference 全部匹配，mismatch 为 0。
- 40 个 ship OBB 与 Ultralytics/PyTorch CPU reference 对照，OpenCV 几何 rotated IoU 最小为 `0.997781`。
- 最大 source-image 坐标误差为 `0.126 px`，最大周期 angle 误差为 `0.000432 rad`，最大 score 误差为 `0.000524`。
- 将 reference 第 0 个值增加 10,000 后，程序退出码为 1，`Mismatches=1`、`FirstMismatch=0`。

`4.25` 是 raw tensor 中坐标通道的 backend 容差，不是最终框精度要求。最终几何仍由独立 rotated IoU、坐标、angle 和 score 阈值分别约束。

## 模型导出与哈希

```powershell
yolo export `
  model=..\downloads\cases\yolov8n-obb\models\yolov8n-obb.pt `
  format=onnx `
  opset=17 `
  simplify=True `
  dynamic=False `
  imgsz=1024

Get-FileHash -Algorithm SHA256 ..\downloads\cases\yolov8n-obb\models\yolov8n-obb.pt
Get-FileHash -Algorithm SHA256 ..\downloads\cases\yolov8n-obb\models\yolov8n-obb.onnx
Get-FileHash -Algorithm SHA256 ..\downloads\cases\yolov8n-obb\labels\dota.names
Get-FileHash -Algorithm SHA256 ..\downloads\cases\yolov8n-obb\images\input.ppm
```

命令只是 owner 已审核模型的骨架。导出后应使用 Netron、ONNX metadata 或 TensorRtExec binding report 核对真实输入名和输出 shape，不能从 `yolov8n-obb` 文件名推断 angle 合同。

## TensorRtExec build-only

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx ..\downloads\cases\yolov8n-obb\models\yolov8n-obb.onnx `
  --saveEngine ..\downloads\cases\yolov8n-obb\engines\yolov8n-obb.plan `
  --minShapes images:1x3x1024x1024 `
  --optShapes images:1x3x1024x1024 `
  --maxShapes images:2x3x1024x1024 `
  --fp16 `
  --buildOnly `
  --exportReport ..\downloads\cases\yolov8n-obb\reports\build-report.json
```

`--exportReport` 是当前真实参数。该报告证明构建流程和配置被执行，但不证明 angle 单位、范围、旋转方向或 rotated NMS 正确。

## 输出合同与单位

官方单输出使用：

```text
--class-count 15
--layout channels-first
--has-objectness auto
--aux-channel-start 19
--aux-layout channels-first
--angle-radians
```

`--aux-channel-start` 会显式创建内嵌 OBB metadata；程序严格要求它等于 box/objectness/class detection prefix，且后面只能剩一个 angle channel。layout 不同、class count 未知、起点错误或多余 channel 都会 fail closed。

旧的独立 angle tensor 路径继续支持：

```text
--output-role-map boxes:det,angles:obb-angle
--detection-output boxes
--obb-angle-output angles
--aux-layout boxes-first
--angle-radians
```

如果 angle tensor 是 degree，改为 `--angle-degrees`。专用参数或 role map 至少要保留一个；推荐 evidence 同时记录 tensor 原名和 role。名称启发式可以识别 `angle`、`theta`、`obb`，但不适合作为 owner 合同。

`YoloRuntimeOutputRoleResolver.CreateMetadata` 在明确声明独立 OBB angle role或提供 `--aux-channel-start` 时创建 OBB metadata。两者都没有时，单输出诊断路径不能冒充完整 OBB 解码。

## 离线 preflight

```powershell
dotnet run --project .\applications\YoloVision -- `
  --model ..\downloads\cases\yolov8n-obb\models\yolov8n-obb.onnx `
  --labels ..\downloads\cases\yolov8n-obb\labels\dota.names `
  --image ..\downloads\cases\yolov8n-obb\images\input.ppm `
  --input-shape 1x3x1024x1024 `
  --family v8 --task obb `
  --layout channels-first --has-objectness auto --class-count 15 `
  --nms-mode class-aware `
  --aux-channel-start 19 --aux-layout channels-first --angle-radians `
  --preflight --strict-preflight `
  --preflight-report ..\downloads\cases\yolov8n-obb\reports\preflight.json
```

检查 `yolovision-preflight.v1`、`proofClassification=precheck`、`obbAngleInDegrees=false`、input source exclusivity、资产 hash 和 owner action。preflight 不打开 TensorRT，不执行 enqueue，也不会验证 rotated geometry。

## 真实运行与输出

```powershell
dotnet run --project .\applications\YoloVision -- `
  --model ..\downloads\cases\yolov8n-obb\models\yolov8n-obb.onnx `
  --labels ..\downloads\cases\yolov8n-obb\labels\dota.names `
  --image ..\downloads\cases\yolov8n-obb\images\input.ppm `
  --preprocessed-output ..\downloads\cases\yolov8n-obb\tensors\input-fp32.bin `
  --input-shape 1x3x1024x1024 `
  --family v8 --task obb `
  --layout channels-first --has-objectness auto --class-count 15 `
  --nms-mode class-aware --confidence 0.25 --iou-threshold 0.45 `
  --top-k 40 --aux-channel-start 19 --aux-layout channels-first --angle-radians `
  --output-json ..\downloads\cases\yolov8n-obb\reports\output.json `
  --visualization-svg ..\downloads\cases\yolov8n-obb\overlays\obb-preview.svg `
  *> ..\downloads\cases\yolov8n-obb\logs\run.log
```

若实际 tensor 使用 degrees，运行命令、preflight 和 owner manifest 必须同时改为 `--angle-degrees`，不能只在文章文字中改单位。

## JSON 与 SVG 语义

每条 OBB prediction 包含：

- `center.x/y` 与 `size.width/height`，来自保留的 detection box。
- `angle`，统一为 radians。
- `angleUnit=radian`。
- `angleRange=owner-record-required`。
- `classId/className/score`。

示例位于 `applications/YoloVision/examples/yolovision-output-obb.example.json`。真实 runtime writer 的规范化字段优先于示例中的说明性值，owner 仍应单独记录 exporter 原始角度合同。

SVG 使用 `AngleRadians * 180 / PI` 旋转矩形，适合快速发现 90 度偏差、宽高颠倒和明显坐标错误。它不是 rotated-IoU 评估，也不能证明原图 resize-back 正确。

## 输出校验

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionOutputReport.ps1 `
  -InputPath ..\downloads\cases\yolov8n-obb\reports\output.json `
  -OutputPath ..\downloads\cases\yolov8n-obb\reports\output-validation.json `
  -Strict
```

validator 会检查 task、center、size、angle、angleUnit、angleRange、输出摘要与 proof boundary。它不会替 owner 判断 clockwise/counter-clockwise、角度周期、宽高交换规则或 rotated NMS 的正确性。

## 人工几何检查

建议至少选择三类目标：接近 0 度、接近周期边界、明显斜放。对每个目标记录：

1. 原图中长轴方向。
2. exporter 原始 angle 与单位。
3. YoloVision 输出 radians。
4. SVG 显示方向。
5. width/height 是否发生交换。
6. 与模型参考实现的 corner coordinates 对比。

仅检查“框覆盖了目标”不够，因为 90 度偏差在近方形目标上很难肉眼发现。

## 常见失败

| 表现 | 首先检查 |
| --- | --- |
| 全部旋转约 57.3 倍 | degree/radian 参数写反 |
| 全部相差 90 度 | 起始轴或 width/height 规范不一致 |
| 中心正确但长短边颠倒 | exporter 是否交换 width/height 并补偿角度 |
| 相邻方向目标被错误抑制 | 核对 probabilistic-IoU 阈值、class-aware 模式和 exporter 的 rotated NMS 定义 |
| 角度属于另一个目标 | `SourceIndex` 或候选行顺序不一致 |
| OBB metadata 为 null | 既没有 `--aux-channel-start`，也没有声明独立 angle output role |
| build 成功但没有 OBB proof | 只有 build-only，没有真实图、run log、JSON、SVG 和人工 review |

## 证据边界

真实 OBB 候选至少需要 model/labels/image/preprocessed tensor/engine/output JSON/run log 的 SHA256、TensorRtExec build report、preflight、完整命令、host/runtime metadata、stdout/stderr 摘要、参考实现对照和 owner review。再用 `eng/Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog` 验证真实日志。

owner 审核通过后最多形成 `real-model-runtime` 候选。它不是 `package-consumer-runtime`；后者要求仓库外 clean consumer 从目标 package source restore/build/run。`blocked-by-cuda-driver`、template、build-only、sidecar-only、synthetic input、ProjectReference 和本地 `.nupkg` 都不能替代真实 OBB runtime proof。

当前官方 YOLOv8n-obb 已完成仓库外三包 `PackageReference` 运行、430,080 值 raw 对照、40 个旋转框独立几何对照和受控负例，详见 [YoloVision YOLOv8n OBB 本地包消费教程](yolovision-yolov8n-obb-local-package-consumer-tutorial.md)。该记录分类为 `local-package-consumer-runtime`，仍不是公共 feed 下载、post-publish、Owner 发布批准或 release proof。转换后的 ONNX 暂存在外层 `..\models`，不上传当前仓库。

## 代码入口

- `applications/YoloVision/YoloRuntimeOutputRoleResolver.cs`：angle role、单位和 auxiliary layout 参数。
- `applications/YoloVision/YoloSampleRunner.cs`：detection decode、`SourceIndex` 绑定与 angle row 路由。
- `applications/YoloVision/YoloObbDecoder.cs`：degree/radian 转换、probabilistic IoU 与 rotated Fast-NMS。
- `applications/YoloVision/YoloVisionOutputReport.cs`：center/size/radian 输出合同。
- `applications/YoloVision/YoloVisionVisualizationWriter.cs`：旋转矩形 SVG。
- `eng/Test-YoloVisionOutputReport.ps1`：输出结构与 proof boundary 校验。

## 收尾清单

- [ ] 模型、labels、图片来源、许可证和再分发结论已审核，资产仅位于 E 盘。
- [ ] detection/angle tensor 名、shape、dtype、候选维顺序和 layout 已确认。
- [ ] 原始 angle 单位、范围、方向、起始轴和 width/height 规则已记录。
- [ ] 已明确 graph、应用层或 adapter 中的 NMS 类型。
- [ ] 内嵌或独立 angle 都通过 `SourceIndex` 绑定原始候选，并使用模型合同匹配的 rotated NMS。
- [ ] build report 使用 `--exportReport`，preflight/runtime 命令已归档。
- [ ] output JSON、SVG、参考 corner 对照、run log 和全部 hash 已归档。
- [ ] 没有把已验证的 probabilistic-IoU 路径泛化成所有 exporter 的 rotated NMS 合同。
- [ ] 没有把 build-only、precheck 或本地结果写成 package-consumer-runtime 或发布批准。
