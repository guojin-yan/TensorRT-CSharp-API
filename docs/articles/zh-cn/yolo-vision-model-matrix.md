# YoloVision 模型系列与任务矩阵

`samples/YoloVision` 是项目统一的 YOLO-family 样例入口，当前定位已经从单一检测样例扩展为覆盖 detection、classification、segmentation、OBB、pose 和 semantic segmentation 的综合样例。它不再使用早期过窄的检测样例语义作为当前项目名。

机器可读任务输出契约位于 `samples/YoloVision/yolovision-task-output-contract.json`。该契约把 `det`、`cls`、`seg`、`obb`、`pose`、`sem` 的输出角色、必填 metadata、TensorRtExec shape profile 建议、文章入口和 proof 边界集中维护，避免 README、asset pack、文章和测试各写一份后发生漂移。它只是 support/planning evidence：不能替代真实模型、真实输入、输出 JSON、日志 SHA256 和 owner review。

本文面向后续技术文章、微信公众号教程和真实案例 proof backfill：它说明每个 YOLO 系列和任务类型应该如何准备模型、导出 ONNX、记录输入输出 metadata、选择后处理路径，并明确当前是否已经具备 runtime proof。

> 证据边界：本文和 `samples/YoloVision/README.md` 是模型矩阵与教程规划，不是 runtime proof。没有真实模型、许可证、输入资产、运行日志、SHA256、stdout/stderr summary 和 owner review 时，只能标记为 `planned` / `documented` / `not-proof`。

## 支持范围总览

| 系列 | 当前矩阵状态 | 主要任务 | 模型来源建议 | ONNX 导出建议 | Runtime proof 状态 |
|---|---|---|---|---|---|
| YOLOv5 | documented | det / cls / seg | Ultralytics YOLOv5 release 或用户自训模型 | 使用官方 export.py 或等价导出，固定 opset、input size、dynamic axes | not-proof，等待 owner 资产 |
| YOLOv6 | documented | det | Meituan YOLOv6 release 或用户自训模型 | 使用官方部署脚本导出 ONNX，记录 decode 方式和 NMS 位置 | not-proof |
| YOLOv7 | documented | det / pose variants | WongKinYiu YOLOv7 release 或用户自训模型 | 导出时记录 end-to-end NMS 是否在图内 | not-proof |
| YOLOv8 | documented | det / cls / seg / obb / pose | Ultralytics YOLOv8 模型或用户自训模型 | `yolo export format=onnx`，记录 imgsz、dynamic、simplify、opset | not-proof |
| YOLOv9 | documented | det / seg variants | YOLOv9 release 或用户自训模型 | 记录 dual branch/head 输出是否已简化 | not-proof |
| YOLOv10 | managed end-to-end decoder ready | det | THU-MIG YOLOv10 或用户自训模型 | `[1,N,6]` 使用 `--layout end2end`；其他输出必须记录真实 metadata | managed smoke；real model not-proof |
| YOLOv11 | documented | det / cls / seg / obb / pose | Ultralytics YOLOv11 模型或用户自训模型 | 与 v8 类似，但必须记录实际导出命令和输出 tensor metadata | not-proof |
| YOLOv26 | documented | det / cls / seg / obb / pose / sem planned | 用户或上游候选模型 | 先以 custom profile 接入，补 family-specific decode notes | not-proof |

## 任务矩阵

| 任务 | Alias | 当前代码路径 | 输入要求 | 输出 metadata | 后处理边界 | Runtime proof 状态 |
|---|---|---|---|---|---|---|
| Detection | `det` | `DecodeEndToEnd`、`YoloEndToEndOutput`、`YoloSampleRunner.DecodeOutput` | `NCHW` float32 tensor，典型 `1x3x640x640` | output shape、layout、class count、objectness/column rule、NMS mode | raw head 使用应用侧 NMS；end-to-end 六列输出禁止二次 NMS | synthetic/managed pipeline ready；real model not-proof |
| Classification | `cls` | `YoloVisionResult` classification path | 分类模型输入 tensor，labels 必须匹配 logits | logits tensor name、class count、top-k rule | 单输出 logits/top-k | managed smoke ready；real model not-proof |
| Segmentation | `seg` | `YoloMaskComposer`、`DecodeSegmentationOutputs` | detection 输入 + mask proto metadata | boxes tensor、mask coefficient count、prototype tensor shape/layout | mask coefficient/prototype compose；crop/resize 由 owner 记录 | managed metadata ready；real model not-proof |
| Oriented Bounding Box | `obb` | `YoloObbDecoder`、`DecodeObbOutputs` | 检测输入 + angle tensor metadata | angle channel/tensor name、degree/radian flag | box + angle conversion，NMS 策略需记录 | managed metadata ready；real model not-proof |
| Pose | `pose` | `YoloPoseDecoder`、`DecodePoseOutputs` | 检测输入 + keypoint metadata | keypoint count、stride、tensor role/layout | keypoint tensor mapping；skeleton 可在文章中说明 | managed metadata ready；real model not-proof |
| Semantic Segmentation | `sem` | `YoloSemanticMap` path | semantic model输入 tensor | semantic tensor name、class count、map width/height | 单输出 semantic map decoder | managed smoke ready；real model not-proof |

## 模型获取与资产记录

真实教程不能只写“下载模型”。每篇模型文章至少记录：

- 模型来源 URL。
- 模型许可证。
- 模型文件名和 SHA256。
- ONNX 导出命令。
- opset、dynamic axes、input size。
- labels 来源、许可证、行数和 SHA256。
- 测试图片来源、许可证和 SHA256。
- 预处理：resize、letterbox、padding、RGB/BGR、normalize、NCHW/NHWC。
- 输出 tensor 名称、shape、dtype、layout。
- 后处理阈值、NMS mode、mask/pose/OBB/semantic metadata。

建议资产清单从以下模板开始：

```text
samples/assets/yolovision-assets.template.json
artifacts/user-acceptance/onnx-engine-build-evidence-sidecar.yolovision.template.json
artifacts/user-acceptance/sample-run-evidence-record.yolovision.template.json
```

## ONNX 导出建议

不同 YOLO 系列的导出命令不同，文章中不要伪造统一命令。建议统一记录为：

```text
family: YOLOv8
task: det
sourceModel: yolov8n.pt
exportCommand: yolo export model=yolov8n.pt format=onnx imgsz=640 opset=17 simplify=True dynamic=False
inputShape: 1x3x640x640
outputLayout: boxes-first 或 channels-first
hasObjectness: auto / true / false
nmsMode: class-aware / class-agnostic
```

对于 YOLOv5/v6/v7/v9/v10/v11/v26，必须保留实际上游导出命令，而不是复制其他系列命令。

## YoloVision 命令骨架

Detection：

```powershell
dotnet run --project .\samples\YoloVision -- `
  --model .\models\yolo-det.onnx `
  --labels .\models\coco.names `
  --input-data .\models\det-fp32.bin `
  --input-shape 1x3x640x640 `
  --family v8 `
  --task det `
  --layout auto `
  --has-objectness auto `
  --nms-mode class-aware
```

Segmentation：

```powershell
dotnet run --project .\samples\YoloVision -- `
  --model .\models\yolo-seg.onnx `
  --labels .\models\coco.names `
  --input-data .\models\seg-fp32.bin `
  --input-shape 1x3x640x640 `
  --family v8 `
  --task seg `
  --output-role-map boxes:det,proto:mask-prototypes `
  --mask-coefficient-count 32
```

Pose：

```powershell
dotnet run --project .\samples\YoloVision -- `
  --model .\models\yolo-pose.onnx `
  --labels .\models\labels.txt `
  --input-data .\models\pose-fp32.bin `
  --input-shape 1x3x640x640 `
  --family v8 `
  --task pose `
  --output-role-map boxes:det,keypoints:pose-keypoints `
  --pose-keypoint-count 17
```

OBB：

```powershell
dotnet run --project .\samples\YoloVision -- `
  --model .\models\yolo-obb.onnx `
  --labels .\models\labels.txt `
  --input-data .\models\obb-fp32.bin `
  --input-shape 1x3x1024x1024 `
  --family v8 `
  --task obb `
  --output-role-map boxes:det,angles:obb-angle `
  --obb-angle-output angles
```

Classification / Semantic：

```powershell
dotnet run --project .\samples\YoloVision -- --model .\models\yolo-cls.onnx --labels .\models\labels.txt --input-data .\models\cls-fp32.bin --input-shape 1x3x224x224 --family custom --task cls --classification-output logits

dotnet run --project .\samples\YoloVision -- --model .\models\yolo-sem.onnx --labels .\models\labels.txt --input-data .\models\sem-fp32.bin --input-shape 1x3x512x512 --family custom --task sem --semantic-output semantic --class-count 21
```

这些命令是可复制教程骨架，不代表仓库已经携带真实模型 proof。

## 真实 proof checklist

一篇可发布的真实模型文章必须附带或引用以下记录：

- `TensorRtExec` build-only report 和 normalized command SHA256。
- owner 提供的 YoloVision 成功标记运行日志。
- stdout/stderr summary。
- model / ONNX / engine / labels / image / preprocessed tensor / run log 的 SHA256。
- 模型与图片许可证。
- host OS / GPU / driver / CUDA / TensorRT / cuDNN。
- `real-model-runtime` 判定结果。
- 明确声明它不是 `package-consumer-runtime`。

## 后续文章拆分建议

本矩阵可以拆成多篇公众号/博客文章：

1. YoloVision 总览：一个 C# TensorRT 项目如何统一 YOLO 系列。
2. YOLOv8 Detection 从模型下载到 TensorRT engine。
3. YOLOv8 Segmentation 的 mask prototype 后处理。
4. YOLOv8 Pose 的 keypoint 输出解释。
5. YOLOv8 OBB 的 angle tensor 与旋转框。
6. YOLOv5/v7/v9 输出布局差异。
7. YOLOv10 NMS-free/end-to-end 六列输出，从官方模型获取、ONNX 检查到 TensorRT 结果。
8. YOLOv11 与 YOLOv26 的接入计划。
9. YoloVision 真实资产 manifest 与 proof backfill。
10. 常见问题：shape、layout、labels、NMS、许可证和 runtime proof。

## 第二批正文门禁

### 适用读者

本文适合准备选择 YOLO family 和任务类型的使用者，也适合计划撰写多篇模型案例文章的维护者。

### 解决问题

YOLO 系列的难点不只是模型数量多，还包括任务类型、输出 layout、NMS 规则、mask/keypoint/angle 后处理和许可证差异。本文解决“先选哪个模型、如何记录资产、哪些还只是计划、哪些可以进入 proof”的问题。

### 核心思路

核心思路是先建立 family/task matrix，再为每个模型填充资产证据。YoloVision matrix 只能说明路线图和能力边界；build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、OnnxToEngine report、readonly diagnostics 都不能替代 runtime proof。

### 操作路径

选择 family 和任务后，记录模型来源、许可证、ONNX 导出命令、labels 和输入图片；用 OnnxToEngine 或 TensorRtExec 生成 build/report evidence；最后用 YoloVision runner 生成真实推理输出，再由 validator 晋级为 runtime proof。

### 边界说明

本文是矩阵和教程规划，不是 public package proof、post-publish proof 或 release close approval。只有真实模型、真实输入、真实兼容主机、输出摘要、hash 和 owner review 都存在时，某一行才能从 planned/documented 晋级为 runtime proof。

### 下一步

下一步把矩阵拆成 detection、classification、segmentation、pose、OBB 和 semantic segmentation 多篇案例文章。
