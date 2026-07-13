# YOLO 全系列配置与后处理指南

`samples/YoloVision` 的目标不是在仓库里内置某个特定 YOLO 权重，而是为 YOLO-family 模型提供一套可审计的配置底座。模型、labels、图片和许可证由用户或发布 owner 选择；样例负责把输入 shape、输出 layout、任务类型和托管后处理边界讲清楚。

当前样例已经覆盖以下托管能力：

- YOLOv5、v6、v7、v8、v9、v10、v11、v26 和 custom family 标记。
- `det`、`cls`、`seg`、`obb`、`pose`、`sem` 任务标记。
- `--list-capabilities` 离线能力矩阵，不依赖 CUDA、TensorRT、ONNX 模型、labels 或图片。
- `[1,84,8400]` channel-first 和 `[1,8400,84]` box-first 输出布局推断。
- objectness 自动判断和显式指定。
- confidence filtering。
- class-aware 和 class-agnostic NMS。
- segmentation mask compose、pose keypoint、OBB angle、semantic map 的托管辅助类型。
- end-to-end NMS 输出校验模型。

## 为什么先做配置底座

YOLO 系列的 ONNX 输出并不完全统一。不同 family、export 脚本、opset、NMS 插入方式和 task 会影响输出 tensor：

- 有的模型输出 `[box + class]`，没有 objectness。
- 有的模型输出 `[box + objectness + class]`。
- 有的模型把 NMS 放在图内或 plugin 内，输出已经是 end-to-end detection。
- segmentation 需要 mask coefficients 和 prototype。
- pose 需要 keypoint count 和 stride。
- OBB 需要 angle channel 和角度单位。
- semantic segmentation 输出常常不是检测框，而是类别图或 logits map。

如果直接把某一种 YOLOv8 detection decode 写死为“YOLO 支持”，项目会很快在其它 family 上失真。因此本阶段先把 family/task/profile/layout/postprocess 元数据补齐，让后续每个真实模型都能按清单接入。

## 基本运行命令

先查看当前 family/task 支持矩阵：

```powershell
dotnet run --project .\samples\YoloVision -- --list-capabilities
```

该命令只输出支持范围，不会加载 TensorRT runtime，也不会把任意外部模型声明为已验证。它适合用于文档、CI smoke、资产清单规划和下一步模型接入前的能力对照。

真实模型运行命令示例：

```powershell
dotnet run --project .\samples\YoloVision -- `
  --model .\models\yolo.onnx `
  --labels .\models\coco.names `
  --input-data .\models\yolo-preprocessed-fp32.bin `
  --input-shape 1x3x640x640 `
  --tensor-rt-line 10 `
  --family v8 `
  --task det `
  --layout auto `
  --has-objectness auto `
  --nms-mode class-aware `
  --confidence 0.25 `
  --iou-threshold 0.45
```

默认样例可以用 synthetic input 证明 TensorRT 管线和托管 decode 路径，但这不是目标检测质量证明。要写成真实 smoke passed，必须提供模型、labels、图片、预处理后 tensor、hash、授权信息和真实输出日志。

## 输入数据边界

`YoloVision` 现在区分三类输入：

- 未传 `--input` / `--input-data`：使用 `zeros`、`ones` 或 `ramp` 合成 tensor，只能证明 pipeline。
- `--input <path>`：读取 raw byte tensor，byte 数量必须等于输入元素数量，并归一化到 `[0,1]`。
- `--input-data <path>`：读取预处理好的 float tensor，支持 `.bin`/`.raw` float32 或文本浮点值，元素数量必须和 `--input-shape` 完全一致。

图片解码、resize、letterbox、通道顺序、归一化和 padding 仍由 owner 的预处理命令负责，并写入 asset manifest。这样可以避免样例把某个 YOLO family 的图像策略硬编码成全系列默认行为。

## 输出布局选择

常见 detection 输出有两类：

- Channel-first：`[1, 84, 8400]`，通道通常是 4 个 box 值加 80 类分数。
- Box-first：`[1, 8400, 84]`，每个候选框一行。

`--layout auto` 会根据 shape 推断；如果模型输出存在歧义，应显式指定。`--has-objectness auto` 会在 class count 可判断时自动选择 objectness 规则，否则应根据模型文档指定。

## NMS 模式

class-aware NMS 只在同一 class 内抑制重叠框，适合多数标准检测输出。class-agnostic NMS 不区分类别，适合某些 end-to-end 或部署策略要求更强全局抑制的模型。

样例中 `YoloPostprocessOptions` 支持 `YoloNmsMode.ClassAware`、`YoloNmsMode.ClassAgnostic` 和 `YoloNmsMode.None`。命令行可用 `--nms-mode class-aware|class-agnostic|none` 选择，也可用 `--no-nms` 临时关闭 NMS 只保留 score filtering。项目质量测试已经覆盖这些模式的托管行为，但真实模型仍要用实际图像验证阈值、类别和框坐标。

## Segmentation、Pose、OBB、Semantic

本阶段对高级任务的策略是“先提供安全辅助结构，再接具体模型”：

- Segmentation：`YoloMaskComposer` 负责从 coefficients/prototypes 组合 mask，但 crop、resize、threshold 规则需要模型元数据确认。
- Pose：`YoloPoseDecoder` 提供 keypoint 快照结构，keypoint count/stride 必须来自模型说明。
- OBB：`YoloObbDecoder` 记录角度和单位，角度通道位置不能猜。
- Semantic：`YoloSemanticMap` 表达语义图尺寸与类别数据，后续可接 argmax 或调色板输出。

这些能力让样例可以逐步支持 det/seg/pose/obb/sem，而不需要每次重写 runner。

## 资产清单

每个真实 YOLO demo 都应填写 `samples/assets/yolovision-assets.template.json`，至少包括：

- 模型名称、family、task、source URL、license、download URL、SHA256、opset、export command。
- labels 来源、license、class count、SHA256。
- 测试图片来源、license、SHA256。
- 输入 tensor name、shape、layout、dtype。
- 输出 tensor name、shape、layout、class count、objectness 规则。
- resize、padding、color order、scale、mean/std。
- NMS location 和 NMS mode。
- segmentation/pose/OBB/semantic 的任务专属元数据。
- 实际 run command 和 evidence lines。

在这些字段未补齐前，`isSmokePassed` 必须保持 `false`。这不是保守主义，而是为了让发布材料经得起复现。

## 推荐接入顺序

1. 先选定一个许可证清晰、导出流程稳定的 detection ONNX。
2. 填写 asset manifest，记录 hash 和模型 I/O。
3. 用 `--buildOnly` 或 TensorRtExec 先验证 ONNX 能构建 engine。
4. 再运行 `samples/YoloVision`，确认输出 layout、objectness、class count。
5. 调整 confidence 和 IoU threshold，记录真实图片输出。
6. 最后再推进 seg、pose、OBB 或 semantic 模型。

这样可以把“构建成功”“管线跑通”“检测质量可接受”拆成三类证据，避免一个日志承担过多语义。
