# YoloVision YOLOv8n OBB 本地包消费教程

本文验证官方 YOLOv8n OBB 模型的仓库外 `PackageReference` 路径。消费项目只引用 managed API、YoloVision 与一个 bridge-only 包；CUDA、cuDNN 和 TensorRT 由用户自行安装。权重、图片、转换后的 ONNX、输入 tensor、参考输出和运行日志均不进入 Git、NuGet 或 GitHub Release。

## 获取模型

固定资产来自 Ultralytics `v8.3.0`：

| 资产 | 来源与固定值 |
| --- | --- |
| 权重 | `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-obb.pt` |
| source revision | `ultralytics-v8.3.0@6e43d1e1e5db72afbf686dee6745669bcb124b0a` |
| 权重 SHA256 | `fa6e4cd2691f132875c143135affaa66b5d89394ebb1d07d19770a9b6382c1b8` |
| 许可证 | `AGPL-3.0-only` |
| 输入图片 | Ultralytics assets commit `428939c3e501f70ab2a0ded889663efcf5bfbe6c` 的 `boats.jpg` |
| 图片 SHA256 | `8c5ada657cf8110a9f8aaac954c1dd96cde0187315b581276c32b0d1863e756f` |

联网获取并校验：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Acquire-YoloV8ObbOfficialAssets.ps1 `
  -OutputRoot ..\downloads\yolov8n-obb-ultralytics-v8.3.0 `
  -PythonPath python
```

已有资产可增加 `-Offline`，只做固定长度与 SHA256 复核。轻量来源合同位于 `samples/assets/yolovision-yolov8n-obb-official-assets.json`。权重与图片没有获得本项目公开再分发批准。

## 转换 ONNX

在安装了 Ultralytics `8.4.21`、PyTorch `2.10.0`、ONNX `1.15.0` 的本地 Python 环境执行：

```powershell
yolo export `
  model=..\downloads\yolov8n-obb-ultralytics-v8.3.0\source\yolov8n-obb.pt `
  format=onnx imgsz=1024 opset=17 simplify=True dynamic=False batch=1 device=cpu
```

转换后的 ONNX 必须暂存在 Git 仓库外：

```text
..\models\YoloVision\OrientedBoundingBox\yolov8n-obb-ultralytics-v8.3.0\yolov8n-obb.onnx
```

固定长度为 `12,664,838`，SHA256 为 `5f2701ef5326fb5a691999438cfc55a69656323c21ffddebaff8968ab6de2e92`。该目录只用于当前开发暂存，后续迁移到单独治理的 Model Zoo，不上传当前 GitHub 仓库。

静态合同是 `images:[1,3,1024,1024] -> output0:[1,20,21504]`：4 个 box 通道、15 个 DOTA class 通道和 channel 19 的内嵌角度，没有独立 objectness。角度单位为 radians，范围为 `[-pi/4,3pi/4]`；后处理使用 class-aware probabilistic-IoU rotated NMS。

## 生成独立参考

先由 YoloVision 的 center-letterbox 预处理生成固定输入 tensor，再执行：

```powershell
& python `
  .\eng\Invoke-YoloVisionObbReference.py `
  --onnx-model ..\models\YoloVision\OrientedBoundingBox\yolov8n-obb-ultralytics-v8.3.0\yolov8n-obb.onnx `
  --weights ..\downloads\yolov8n-obb-ultralytics-v8.3.0\source\yolov8n-obb.pt `
  --image ..\downloads\yolov8n-obb-ultralytics-v8.3.0\source\boats.jpg `
  --input-tensor ..\downloads\yolov8n-obb-ultralytics-v8.3.0\runtime\boats-1x3x1024x1024-rgb-letterbox.fp32.bin `
  --output-directory ..\downloads\yolov8n-obb-ultralytics-v8.3.0\reference `
  --input-shape 1 3 1024 1024 `
  --output-shape 1 20 21504 `
  --max-detections 40
```

脚本生成 ONNX Runtime CPU 的 `430,080` 值 raw reference，以及 Ultralytics/PyTorch CPU 的 40 个旋转框参考。固定 raw reference SHA256 为 `53f00a488c44227a3e5fbc86b530355acfd0c17cf9fdfc771a6d1f99f72fd65d`；固定 PyTorch 参考 SHA256 为 `97e777e1bcecec7fdb3b15e8d4d1e34468505947522c8881c435b484afc10045`。

## 构建与运行三个包

先生成 managed API、YoloVision 和对应 `.Bridge` 包，然后执行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Test-YoloVisionObbLocalPackageConsumer.ps1 `
  -RepositoryRoot $PWD `
  -PackageVersion 4.0.0
```

runner 将选定的三个 nupkg 分别复制到隔离 feed，restore 后比较 NuGet 缓存中的 nupkg SHA256，防止同 ID/版本的旧包污染证据。包中 NVIDIA vendor runtime 条目必须为 0，bridge-only 包只能包含 `jyppxtrtbridge.dll`。

## 验证结果

- C# center-letterbox 预处理生成 `3,145,728` 个 float，SHA256 为 `c56c027619088bce94f9160a3f602b4ad81fe323001867b1ee456100040fec6e`，与权威 tensor 逐字节一致。
- TensorRT 比较 `output0` 全部 `430,080` 值，mismatch 为 `0`。统一比较器使用 `abs=4.25/rel=0.05`；本次最大绝对误差为 `3.734070`。
- 后处理得到 40 个 `ship` 旋转框，全部使用 radians angle。
- 独立 PyTorch 对照的最小 rotated IoU 为 `0.997750`，最大坐标误差为 `0.122431` 像素，最大周期角度误差为 `0.000420` radians，最大目标分数误差为 `0.000495`。
- raw reference 第 0 个值增加 `10000` 后必须退出 `1`，产生一个 mismatch，`firstMismatchIndex=0`。

紧凑记录位于 `samples/assets/yolovision-yolov8n-obb-local-package-consumer-runtime-evidence.json`。它是 `local-package-consumer-runtime` 工程证据，不是 nuget.org 下载、public-package、post-publish、模型再分发、Owner 发布批准或 release proof。

OBB 的角度合同、内嵌/独立 angle tensor、坐标还原和 rotated NMS 语义见 [YoloVision OBB 实战教程](yolovision-obb-tutorial.md)。所有演示模型的获取、转换、外层路径与哈希见[演示模型获取、ONNX 转换与本地暂存目录](demo-model-acquisition-and-onnx-conversion.md)。
