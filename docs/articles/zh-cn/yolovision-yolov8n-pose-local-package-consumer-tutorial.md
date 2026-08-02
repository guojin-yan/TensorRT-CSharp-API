# YoloVision YOLOv8n Pose 本地包消费教程

本文验证官方 YOLOv8n Pose 模型的仓库外 `PackageReference` 路径。消费项目只引用 managed API、YoloVision 与一个 bridge-only 包；CUDA、cuDNN 和 TensorRT 由用户自行安装，模型不进入 Git、NuGet 或 GitHub Release。

## 获取模型

固定资产为 Ultralytics `v8.3.0`：

| 资产 | 来源与固定值 |
| --- | --- |
| 权重 | `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt` |
| revision | `ultralytics-v8.3.0@6e43d1e1e5db72afbf686dee6745669bcb124b0a` |
| 权重 SHA256 | `c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212` |
| 许可证 | `AGPL-3.0-only` |
| 输入图片 | 同 revision 的 `bus.jpg`，SHA256 `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63` |

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Acquire-YoloV8PoseOfficialAssets.ps1 `
  -OutputRoot E:\GitSpace\TensorRT-CSharp-API-4.0\downloads\yolov8n-pose-ultralytics-v8.3.0 `
  -PythonPath C:\Users\<user>\.conda\envs\ultralytics\python.exe
```

已有资产可增加 `-Offline`，只做固定长度与 SHA256 复核。权重、图片和导出模型均未获得本项目公开再分发批准。

## 转换 ONNX

```powershell
yolo export `
  model=E:\GitSpace\TensorRT-CSharp-API-4.0\downloads\yolov8n-pose-ultralytics-v8.3.0\source\yolov8n-pose.pt `
  format=onnx imgsz=640 opset=17 simplify=True dynamic=False batch=1 device=cpu
```

转换后的模型必须暂存在 Git 仓库外：

```text
E:\GitSpace\TensorRT-CSharp-API-4.0\models\YoloVision\Pose\yolov8n-pose-ultralytics-v8.3.0\yolov8n-pose.onnx
```

固定长度为 `13,514,570`，SHA256 为 `ed1e8d2d2aeb8a2c66e642a16295a72a2990393e3a3843325537da7e11c8a899`。静态合同是 `images:[1,3,640,640] -> output0:[1,56,8400]`：4 个 box 通道、1 个 person class 通道和 `17*3` 个内嵌关键点通道，没有独立 objectness。

独立参考由以下脚本生成：

```powershell
& C:\Users\<user>\.conda\envs\ultralytics\python.exe `
  .\eng\Invoke-YoloVisionPoseReference.py `
  --onnx-model E:\GitSpace\TensorRT-CSharp-API-4.0\models\YoloVision\Pose\yolov8n-pose-ultralytics-v8.3.0\yolov8n-pose.onnx `
  --weights E:\GitSpace\TensorRT-CSharp-API-4.0\downloads\yolov8n-pose-ultralytics-v8.3.0\source\yolov8n-pose.pt `
  --input-tensor E:\GitSpace\TensorRT-CSharp-API-4.0\downloads\yolov8n-pose-ultralytics-v8.3.0\runtime\bus-1x3x640x640-rgb-letterbox.fp32.bin `
  --image E:\GitSpace\TensorRT-CSharp-API-4.0\downloads\yolov8n-pose-ultralytics-v8.3.0\source\bus.jpg `
  --output-directory E:\GitSpace\TensorRT-CSharp-API-4.0\downloads\yolov8n-pose-ultralytics-v8.3.0\reference
```

它分别生成 ONNX Runtime CPU 的 `470,400` 值 raw reference，以及 Ultralytics/PyTorch CPU 的 4 个 Pose 后处理参考。

## 构建与运行三个包

先生成 managed API、YoloVision 和对应 `.Bridge` 包，然后执行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Test-YoloVisionPoseLocalPackageConsumer.ps1 `
  -RepositoryRoot $PWD `
  -PackageVersion 4.0.0
```

runner 将选定的三个 nupkg 分别复制到隔离 feed，restore 后比较 NuGet 缓存中的 nupkg SHA256，避免同 ID/版本的旧包污染证据。包中 NVIDIA vendor runtime 条目必须为 0，bridge-only 包只能包含 `jyppxtrtbridge.dll`。

## 验证结果

- C# center-letterbox 预处理生成 `1,228,800` 个 float，SHA256 为 `46a0278967f1230ef8db59b0b8311a3aba3dce45233f1ba68821418ae02a574d`，与权威 tensor 逐字节一致。
- TensorRT 比较 `output0` 全部 `470,400` 值，mismatch 为 `0`。统一比较器使用 `abs=1.25/rel=0.05`；本次最大绝对误差为 `1.392639`，相对容差对坐标量级继续生效。
- 后处理得到 4 个 `person` Pose，每个包含 17 个关键点。
- 独立 PyTorch 对照的最小 box IoU 为 `0.998908`，最大目标分数误差为 `0.000263`，最大可见关键点坐标误差为 `3.927416` 像素，最大关键点分数误差为 `0.005012`。
- raw reference 第 0 个值增加 `10000` 后必须退出 `1`，产生一个 mismatch，`firstMismatchIndex=0`。

紧凑记录位于 `samples/assets/yolovision-yolov8n-pose-local-package-consumer-runtime-evidence.json`。它是 `local-package-consumer-runtime` 工程证据，不是 nuget.org 下载、public-package、post-publish、模型再分发、Owner 发布批准或 release proof。

Pose 的单输出通道解释、NMS 后 `SourceIndex` 对齐和 JSON/SVG 语义见 [YoloVision Pose 单输出内嵌通道与多输出实战教程](yolovision-pose-tutorial.md)。所有演示模型的获取、转换、外层路径与哈希见[演示模型获取、ONNX 转换与本地暂存目录](demo-model-acquisition-and-onnx-conversion.md)。
