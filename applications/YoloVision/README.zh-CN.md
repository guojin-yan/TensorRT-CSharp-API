# YoloVision

[English](README.md) | 简体中文

`YoloVision` 是统一的 YOLO 系列视觉应用，通过 TensorRT 运行用户提供的 float ONNX，覆盖检测 `det`、分类 `cls`、实例分割 `seg`、旋转框 `obb`、关键点 `pose` 和语义分割 `sem`。程序包含图像预处理、多输入绑定、输出角色路由、任务后处理、结构化 JSON、独立参考比较和结果图绘制。

该项目是 `IsPackable=false` 的应用程序，通过 NuGet 使用 4 系列 `JYPPX.TensorRT.CSharp.API` 与项目自有 `JYPPX.OpenCV.CSharp.API`，不发布 `JYPPX.TensorRT.CSharp.API.YoloVision` 示例包。

## 离线能力检查

无需 CUDA、TensorRT、ONNX 或图片即可查看能力矩阵和执行托管解码自检：

```powershell
dotnet run --project .\applications\YoloVision -- --list-capabilities
dotnet run --project .\applications\YoloVision -- --self-test-end2end
```

`--preflight` 用于检查 family、task、模型、标签、输入和输出角色元数据，并生成 `yolovision-preflight.v1`。它不会解析 ONNX、构建 Engine 或执行推理，因此不是 runtime proof。

## 支持范围

| 任务 | 典型输出 | 结果展示 |
| --- | --- | --- |
| `det` | raw head 或 YOLOv10 `[1,N,6]` end-to-end | 原图绘制框、类别和置信度 |
| `cls` | 分类 logits/score | Top-K 标签和分数 |
| `seg` | 检测输出、mask coefficient 和 prototype | 原图叠加实例掩码与框 |
| `obb` | 中心点、宽高、角度和类别 | 原图绘制旋转框 |
| `pose` | 检测框与 keypoints | 原图绘制关键点和骨架 |
| `sem` | NCHW/NHWC 语义 logits | argmax 类别图、直方图和彩色叠加图 |

支持的 family 标签包括 YOLOv5/v6/v7/v8/v9/v10/v11/v26、YOLOX 和 custom。不同家族的 score、objectness、输出布局和 NMS 规则不能只凭文件名猜测，必须通过显式 profile 和输出元数据确定。

## 模型获取与 ONNX 转换

仓库不提交模型、权重、标签和测试图片。获取脚本把固定版本资产与转换后的 ONNX 保存到源码仓库同级 `<workspace-root>/models` 或 `downloads` 工作区，并记录来源、许可证、输入输出契约和 SHA256：

```text
eng/Acquire-YoloV8DetectionOfficialAssets.ps1
eng/Acquire-YoloV8ClassificationOfficialAssets.ps1
eng/Acquire-YoloV8SegOfficialAssets.ps1
eng/Acquire-YoloV8ObbOfficialAssets.ps1
eng/Acquire-YoloV8PoseOfficialAssets.ps1
eng/Acquire-YoloV10OfficialAssets.ps1
eng/Acquire-YoloXOfficialAssets.ps1
eng/Acquire-TorchVisionLrasppOfficialAssets.ps1
```

每篇任务文章必须写出上游项目和许可证、固定版本或提交、权重下载命令、框架到 ONNX 的转换命令、opset、输入输出名称和 Shape、预处理/后处理参数以及文件 SHA256。OBB 和 Seg 可使用用户位于外部数据目录的图片，但文章和命令应使用 `<image>` 或相对路径，不写开发机绝对盘符。

统一获取与转换说明见 [模型资产说明](../../docs/articles/zh-cn/yolovision-model-assets.md) 和 [演示模型获取与 ONNX 转换](../../docs/articles/zh-cn/demo-model-acquisition-and-onnx-conversion.md)。

## 图像预处理

`--image` 支持 JPEG/PNG/BMP/PPM。JPEG/PNG 由 `JYPPX.OpenCV.CSharp.API` 解码，BMP/PPM 具有托管回退。内置路径支持 stretch、letterbox、shorter-side center crop、RGB/BGR、NCHW/NHWC、scale、mean/std，并可通过 `--preprocessed-output` 写出实际 float32 tensor。

可在没有 TensorRT Bridge 时先生成预处理 tensor：

```powershell
dotnet run --project .\applications\YoloVision -- `
  --preprocess-only `
  --image <image> `
  --preprocessed-output .\artifacts\yolovision\input-f32.bin `
  --input-shape 1x3x640x640 `
  --tensor-layout NCHW `
  --color-order RGB `
  --resize letterbox
```

## 真实模型运行

以下命令展示检测任务的参数结构，模型和图片按对应文章准备：

```powershell
dotnet run --project .\applications\YoloVision -- `
  --family v8 `
  --task det `
  --model <model.onnx> `
  --labels <labels.txt> `
  --image <image> `
  --input-shape 1x3x640x640 `
  --tensor-layout NCHW `
  --color-order RGB `
  --resize letterbox `
  --output-json .\artifacts\yolovision\output.json `
  --visualization .\artifacts\yolovision\result.png
```

多输出任务通过 `--output-role-map` 明确 boxes、prototype、keypoint、angle 或 semantic tensor 的职责；自定义多输入模型必须为每个输入提供完整 Shape 和唯一数据源。缺失、重复、未知或数值非有限的输入输出应在解码前失败，不进行猜测式降级。

## 文章与结果图

正式任务文章要从项目和所用库介绍开始，依次给出模型获取、ONNX 转换、环境、完整代码流程、命令、真实 stdout、独立参考、受控负例和结果解释。检测框、实例掩码、旋转框、关键点和语义类别必须绘制回原图；分类任务需要展示 Top-K；程序运行输出还要提供已去除本机绝对路径的真实终端截图。

当前系列入口：

- [六任务总览](../../docs/articles/zh-cn/yolovision-all-task-overview.md)
- [检测完整教程](../../docs/articles/zh-cn/yolovision-detection-real-model-tutorial.md)
- [分类真实模型教程](../../docs/articles/zh-cn/yolovision-classification-real-model-tutorial.md)
- [实例分割真实模型教程](../../docs/articles/zh-cn/yolovision-segmentation-real-model-tutorial.md)
- [OBB 教程](../../docs/articles/zh-cn/yolovision-obb-tutorial.md)
- [Pose 教程](../../docs/articles/zh-cn/yolovision-pose-tutorial.md)
- [语义分割指南](../../docs/articles/zh-cn/yolovision-semantic-segmentation-map-guide.md)

## 证据边界

合成输入、preflight、build-only、输出文件哈希或一张结果图都不能单独证明真实模型正确。`real-model-runtime` 至少需要真实模型、真实图片、标签、全部哈希、准确的预处理/输出契约、TensorRT 日志、独立参考比较、受控负例和渲染结果一致。

源码树运行仍不是 `package-consumer-runtime`、公开包或 post-publish 证明。用户机器缺少兼容 CUDA/TensorRT 时应明确记录环境阻塞，并在发布说明中列出运行要求。

## 关键输出

成功执行应包含任务 profile、输入 tensor 指纹、输出角色、解码数量、参考比较状态、结果 JSON/图片路径和 `YoloVision Passed=True`。完整字段和 schema 见英文 README 及 `applications/YoloVision/*.schema.json`。
