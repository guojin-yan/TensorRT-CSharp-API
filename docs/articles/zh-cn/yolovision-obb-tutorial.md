# YoloVision OBB 教程

OBB（Oriented Bounding Box）任务在普通检测框之外还需要角度。本文说明 angle metadata、oriented box 证据回填和常见 layout 风险。

## OBB 与普通检测的差异

普通 det 通常输出 center x/y、width、height、class score。OBB 还需要 angle，且不同模型可能使用 degrees 或 radians。若 angle unit 写错，结果可能看起来“有框但方向全错”。

## 必填 metadata

| 字段 | 说明 |
| --- | --- |
| box layout | center/size 或 corner format |
| angle unit | degrees 或 radians |
| angle range | 例如 [-90,90] 或 [0,180] |
| class count | labels 行数一致 |
| NMS | 是否使用 rotated NMS 或普通 NMS |

## Build-only 与 runner

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\yolo-obb.onnx `
  --saveEngine .\models\yolo-obb.engine `
  --buildOnly `
  --exportProfile .\models\yolo-obb-build-report.json
```

该命令只是 build-only。真实 OBB 输出还需要 YoloVision runner log 和 sample-run-evidence。

## 证据边界

- `TrtexecAlignmentStatus=parse-only` 仍表示高级参数处于 parser/report/GUI 边界。
- `sidecar-only` 只能桥接模型资产和 build report。
- `real-model-runtime` 需要真实模型、真实输入、真实日志和 validator。
- `package-consumer-runtime` 属于 release proof record。
- `blocked-by-cuda-driver` 需要 owner action。

## 常见问题

| 表现 | 可能原因 |
| --- | --- |
| 角度全部旋转 90 度 | angle unit 或 range 错 |
| 框中心正确但宽高颠倒 | layout 与 postprocess 不一致 |
| 类别正确但 NMS 异常 | rotated NMS 策略未明确 |
| build 成功但无输出 proof | 只有 build-only，没有 runner evidence |

## 推荐写法

对外文章可以说“YoloVision 提供 OBB metadata 和 managed postprocess 底座”。不要写成“真实 OBB 模型已经全部通过”，除非 owner 提供模型、license、hash、runner log 和 sample-run-evidence validator 结果。
