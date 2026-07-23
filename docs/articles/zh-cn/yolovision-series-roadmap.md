# YoloVision 系列样例路线图

`samples/YoloVision` 是新的统一 YOLO-family 视觉样例入口，用来替代旧的单一检测命名思路。它面向 YOLOv5/v6/v7/v8/v9/v10/v11/v26/custom，并覆盖 `det`、`cls`、`seg`、`obb`、`pose`、`sem` 等任务。

## 当前定位

YoloVision 的目标不是把一个检测 demo 跑通就结束，而是建立一套可扩展的视觉模型样例框架：

- 统一 family/task/profile/postprocess 元数据。
- 统一模型、labels、输入图片、license、hash 和 run log 记录。
- 统一 TensorRtExec build-only report 与 YoloVision real run evidence 的关系。
- 统一对外文章写法，避免把样例模板误写成 runtime proof。

## Family 与 Task 矩阵

| Family | det | cls | seg | obb | pose | sem |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv5 | planned | planned | planned | planned | planned | planned |
| YOLOv6 | planned | planned | planned | planned | planned | planned |
| YOLOv7 | planned | planned | planned | planned | planned | planned |
| YOLOv8 | documented | documented | documented | documented | documented | planned |
| YOLOv9 | planned | planned | planned | planned | planned | planned |
| YOLOv10 | managed end-to-end decoder | planned | planned | planned | planned | planned |
| YOLOv11 | planned | planned | planned | planned | planned | planned |
| YOLOv26 | planned | planned | planned | planned | planned | planned |
| custom | documented | documented | documented | documented | documented | documented |

状态说明：

- `planned`：路线已收敛，仍需要 owner 资产。
- `documented`：README、矩阵或文章已有入口。
- `managed end-to-end decoder`：`[1,N,6]` xyxy/score/classId 纯托管解码与 smoke 已完成，真实模型 proof 仍需 owner 资产。
- `runnable with user assets`：用户提供真实 ONNX、labels、图片和 metadata 后可运行。
- `proof candidate`：真实日志、hash、host metadata 和 validator 通过后才可能进入 sample-level proof。

## 推荐文章拆分

YoloVision 至少可以支撑以下文章：

1. YoloVision 总览：一个样例覆盖 YOLO 全系列。
2. YOLOv8 Detection：模型下载、导出、engine 构建、YoloVision 推理。
3. YOLOv8 Segmentation：mask prototype 与后处理。
4. YOLOv8 Pose：keypoint 输出解释。
5. YOLOv8 OBB：旋转框 angle 与坐标还原。
6. YOLO Classification 与 semantic segmentation。
7. YOLO 模型资产选择：许可证、SHA256、可再分发边界。
8. YoloVision 真实资产证据链：从矩阵到 owner proof。
9. YOLOv10 End-to-End 输出：官方模型、ONNX、TensorRtExec 与 no-second-NMS decoder。

## 与 TensorRtExec 的关系

推荐用户先用 `applications/TensorRtExec` 或 `samples/OnnxToEngine` 生成 build-only report：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\yolov8n.onnx `
  --saveEngine .\models\yolov8n.plan `
  --minShapes images:1x3x640x640 `
  --optShapes images:1x3x640x640 `
  --maxShapes images:4x3x640x640 `
  --fp16 `
  --exportReport .\models\yolov8n-build-report.json `
  --buildOnly
```

然后再用 YoloVision 做真实样例运行：

```powershell
dotnet run --project .\samples\YoloVision -- `
  --family yolov8 `
  --task det `
  --engine .\models\yolov8n.plan `
  --labels .\models\coco.names `
  --image .\images\bus.jpg `
  --output .\artifacts\yolov8n-det-output.json
```

TensorRtExec report 是 build/report evidence；只有 YoloVision 真实运行日志、输出 JSON、SHA256、host metadata 和 owner review 完整时，才可能进入 `real-model-runtime` 候选。

## 禁止回退

- 旧 detection-only 样例目录已经被 YoloVision 取代；公开入口必须继续使用 YoloVision。
- 不把 detection-only 命名作为公开主入口。
- 不把 build-only、dry-run、template、sidecar、screenshot、local feed、ProjectReference 或 direct `.nupkg` 写成 runtime proof。
- 不用 `package-consumer-runtime` 描述样例自身证据；该分类属于 release proof records。
