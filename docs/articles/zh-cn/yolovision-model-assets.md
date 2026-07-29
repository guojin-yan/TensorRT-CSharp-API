# YoloVision Model Assets

`samples/YoloVision` 面向用户提供的 YOLO-family ONNX。仓库不内置 detector、COCO labels 或图片资产。本文记录完整检测 demo 的资产要求和验证边界。

## 需要的文件

- `model.onnx`：单输入 float YOLO-family detector。
- `labels.txt` 或 `coco.names`：类别名顺序必须与模型输出一致。
- `input.*`：一张或多张可再分发测试图片。
- `postprocess.json` 或等价说明：confidence、IoU、NMS 策略、输出 layout、objectness 规则。

## 输出布局

样例支持两类常见输出：

- `[1, 84, 8400]` channel-first。
- `[1, 8400, 84]` box-first。

不同导出工具可能包含或不包含 objectness，也可能将 NMS 放进图内或要求应用侧处理。文档必须写清楚当前模型的输出格式，不能只写“YOLO 通用”。

## 运行命令

```powershell
dotnet run --project .\samples\YoloVision -- `
  --model .\models\yolo.onnx `
  --labels .\models\coco.names `
  --input-shape 1x3x640x640 `
  --tensor-rt-line 10 `
  --layout auto `
  --has-objectness auto `
  --confidence 0.25
```

## 证据要求

完整 demo 至少要保存：

- 模型来源、许可证、SHA256。
- labels 来源和类别数量。
- 测试图片来源和许可证。
- 输入 shape、layout、dtype。
- 输出 shape、layout、objectness 规则。
- 检测输出或 `Detections=0` 的命令日志。

如果使用 synthetic input，输出只能作为 pipeline evidence，不能作为真实检测质量证据。

## 六任务 reference 语义

`samples/assets/cross-task-reference-provenance-contract.json` 不把 YOLO 输出简化为一个通用 hash。`det` 必须固定 box/
score/objectness/NMS/coordinate 规则；`cls` 必须固定 class score、softmax、labels 和 Top-K；`seg` 必须固定 boxes、mask
coefficients、prototypes、composition/crop/resize/inverse transform；`obb` 必须固定 angle unit/range、rotated layout 和 NMS；
`pose` 必须固定 keypoint count/stride/layout/score/skeleton；`sem` 必须固定 class-axis、argmax、map layout、palette、void
class 和 resize/inverse transform。

因此 det/cls/seg/obb/pose/sem 的 reference 不能互相替代，通用 Classification 与 YoloVision `cls` 也不是同一 profile。
即使 raw tensor SHA256 相同，只要 model/input/preprocess/output/labels/task-semantics 任一 fingerprint 不同，就不能跨任务
复用。当前 7 行 readiness matrix 的 ready row 为 0，所有 promotion flags 保持 false；这准确反映 Owner 资产与语义输入
仍缺失，不应通过填模板或借用 MNIST reference 消除。
