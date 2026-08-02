# 演示模型获取、ONNX 转换与本地暂存目录

TensorRtSharp4.0 的演示代码不把深度学习模型提交到 GitHub。仓库只保存模型来源、许可证边界、固定版本、获取脚本、转换命令、ONNX 契约、文件长度和 SHA256；实际 ONNX 统一暂存在 Git 仓库外：

```text
E:\GitSpace\TensorRT-CSharp-API-4.0\models
```

这个目录是第一版发布前的本地模型缓存，后续迁移到独立 Model Zoo。不得把该目录复制进 `TensorRtSharp4.0`，不得把模型塞进 managed/native NuGet 包，也不得把 CUDA、cuDNN、TensorRT 或 NVRTC 一并打包。用户自行安装 GPU 依赖。

机器可读清单是 `samples/assets/demo-model-inventory.json`。清单中的 `trackedByGit=false`、`uploadsModelFiles=false` 和 `publishesModelFiles=false` 是强制边界，不是临时注释。

各获取/导出步骤完成后，执行统一同步和哈希校验：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Sync-DemoOnnxModels.ps1
```

脚本读取清单，把 `downloads` 或用户 TensorRT sample data 中的 ONNX 复制到外层 `models`，逐项检查长度与 SHA256，并写
`artifacts/demo-models/inventory-validation.json`。只检查而不复制时使用 `-VerifyOnly`；脚本不访问上传或发布接口。

## 目录布局

```text
models/
  Classification/
    resnet18-torchvision-v0.25.0/
  OnnxToEngine/
    MNIST/nvidia-tensorrt-10.11/
  YoloVision/
    Classification/yolov8n-cls-ultralytics-v8.3.0/
    Detection/yolov8n-ultralytics-v8.3.0/
    Detection/yolov10n-thu-mig-v1.1/
    Detection/yolox-s-megvii-v0.1.1rc0/
    InstanceSegmentation/yolov8n-seg-ultralytics-v8.3.0/
    OrientedBoundingBox/yolov8n-obb-ultralytics-v8.3.0/
    Pose/yolov8n-pose-ultralytics-v8.3.0/
    SemanticSegmentation/lraspp-mobilenet-v3-large-torchvision-v0.25.0/
```

## 逐模型可复现合同

以下字段逐项对应 `demo-model-inventory.json`。获取脚本负责下载或定位固定资产；转换命令说明如何得到 ONNX。
`upstream-onnx-no-local-conversion` 表示上游已经发布 ONNX，此时“转换方式”就是不重复转换，而是获取固定发布资产并校验
SHA256。命令里的 `<downloads>`、`<models>` 和 `<artifacts>` 是本机目录占位符。

### `classification-resnet18-imagenet1k-v1`

- 获取 URL：<https://download.pytorch.org/models/resnet18-f37072fd.pth>
- 固定 revision：`torchvision-v0.25.0@8ac84ee75afb1c327902156b5336f56ad63b7e2f`
- 获取脚本：`eng/Acquire-TorchVisionResNet18OfficialAssets.ps1`
- 转换类型：`local-weight-export`
- 转换命令：`python eng/Export-ClassificationResNet18Onnx.py --weights <downloads>/resnet18-f37072fd.pth --onnx <models>/resnet18-imagenet1k-v1.onnx --labels <models>/imagenet1k.names --report <models>/resnet18-onnx-export.json`
- 工具链：`PyTorch 2.10.0+cpu; torchvision 0.25.0+cpu; opset 17`
- ONNX 暂存：`models/Classification/resnet18-torchvision-v0.25.0/resnet18-imagenet1k-v1.onnx`
- SHA256：`ead3558569edd88aa73a4eb46acbe6c38dee113933234547f04a0f6e48169903`

### `onnxtoengine-nvidia-mnist-opset8`

- 获取 URL：<https://github.com/onnx/models/tree/main/validated/vision/classification/mnist>
- 固定 revision：`TensorRT-10.11.0.33-sample-data`
- 获取/验证脚本：`eng/Test-TensorRtExecMnistOnnxRuntimeReference.ps1`
- 转换类型：`upstream-onnx-no-local-conversion`
- 转换说明：`Copy mnist.onnx from the user-installed TensorRT data/mnist directory; no local framework-to-ONNX conversion is performed.`
- 工具链：`upstream ONNX opset 8`
- ONNX 暂存：`models/OnnxToEngine/MNIST/nvidia-tensorrt-10.11/mnist.onnx`
- SHA256：`2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf`

### `yolovision-yolov8n-detection-v8.3.0`

- 获取 URL：<https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt>
- 固定 revision：`ultralytics-v8.3.0@6e43d1e1e5db72afbf686dee6745669bcb124b0a`
- 获取脚本：`eng/Acquire-YoloV8DetectionOfficialAssets.ps1`
- 转换类型：`local-weight-export`
- 转换命令：`yolo export model=yolov8n.pt format=onnx imgsz=640 opset=17 simplify=True dynamic=False batch=1 device=cpu`
- 工具链：`Ultralytics 8.4.21; opset 17`
- ONNX 暂存：`models/YoloVision/Detection/yolov8n-ultralytics-v8.3.0/yolov8n.onnx`
- SHA256：`db28a49ffbb0425f39ae56252e7e0b43d06b357416c7da58872e285560b4221e`

### `yolovision-yolov10n-detection-v1.1`

- 获取 URL：<https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.onnx>
- 固定 revision：`v1.1@799ff3be47d21173bcf29b351820d4b8e955e0fe`
- 获取脚本：`eng/Acquire-YoloV10OfficialAssets.ps1`
- 转换类型：`upstream-onnx-no-local-conversion`
- 转换说明：`Use the hash-pinned official v1.1 ONNX release asset; no local conversion is required.`
- 工具链：`upstream release ONNX`
- ONNX 暂存：`models/YoloVision/Detection/yolov10n-thu-mig-v1.1/yolov10n.onnx`
- SHA256：`7025ea1913f9a259cf8a8465ed608e10610d1bb376db2e0348b13e3bd286e0d3`

### `yolovision-yolox-s-detection-0.1.1rc0`

- 获取 URL：<https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx>
- 固定 revision：`0.1.1rc0@e1052df71842031413f6030723c3607b839c80ce`
- 获取脚本：`eng/Acquire-YoloXOfficialAssets.ps1`
- 转换类型：`upstream-onnx-no-local-conversion`
- 转换说明：`Use the official ONNX release asset; upstream reproduction command: python3 tools/export_onnx.py --output-name yolox_s.onnx -n yolox-s -c yolox_s.pth`
- 工具链：`upstream YOLOX exporter; opset 11 release graph`
- ONNX 暂存：`models/YoloVision/Detection/yolox-s-megvii-v0.1.1rc0/yolox_s.onnx`
- SHA256：`c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`

### `yolovision-yolov8n-classification-v8.3.0`

- 获取 URL：<https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-cls.pt>
- 固定 revision：`ultralytics-v8.3.0@6e43d1e1e5db72afbf686dee6745669bcb124b0a`
- 获取脚本：`eng/Acquire-YoloV8ClassificationOfficialAssets.ps1`
- 转换类型：`local-weight-export`
- 转换命令：`yolo export model=yolov8n-cls.pt format=onnx imgsz=224 opset=17 simplify=True dynamic=False batch=1 device=cpu`
- 工具链：`Ultralytics 8.4.21; opset 17`
- ONNX 暂存：`models/YoloVision/Classification/yolov8n-cls-ultralytics-v8.3.0/yolov8n-cls.onnx`
- 本地包消费：[YoloVision YOLOv8n 分类本地包消费教程](yolovision-yolov8n-cls-local-package-consumer-tutorial.md)
- SHA256：`630c022a99885d59f633ab5a614738f8a49be7f361e340fd3ff89b8c19b0768f`

### `yolovision-yolov8n-instance-segmentation-v8.3.0`

- 获取 URL：<https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt>
- 固定 revision：`ultralytics-v8.3.0@6e43d1e1e5db72afbf686dee6745669bcb124b0a`
- 获取脚本：`eng/Acquire-YoloV8SegOfficialAssets.ps1`
- 转换类型：`local-weight-export`
- 转换命令：`yolo export model=yolov8n-seg.pt format=onnx imgsz=640 opset=17 simplify=True dynamic=False batch=1 device=cpu`
- 工具链：`Ultralytics 8.4.21; opset 17`
- ONNX 暂存：`models/YoloVision/InstanceSegmentation/yolov8n-seg-ultralytics-v8.3.0/yolov8n-seg.onnx`
- SHA256：`08b5c61368d4ddec5e647522fc55a93c42a9e0c581770aae48b87bba65a9b21d`

### `yolovision-yolov8n-pose-v8.3.0`

- 获取 URL：<https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt>
- 固定 revision：`ultralytics-v8.3.0@6e43d1e1e5db72afbf686dee6745669bcb124b0a`
- 获取脚本：`eng/Acquire-YoloV8PoseOfficialAssets.ps1`
- 转换类型：`local-weight-export`
- 转换命令：`yolo export model=yolov8n-pose.pt format=onnx imgsz=640 opset=17 simplify=True dynamic=False batch=1 device=cpu`
- 工具链：`Ultralytics 8.4.21; opset 17`
- ONNX 暂存：`models/YoloVision/Pose/yolov8n-pose-ultralytics-v8.3.0/yolov8n-pose.onnx`
- SHA256：`ed1e8d2d2aeb8a2c66e642a16295a72a2990393e3a3843325537da7e11c8a899`
- 本地包消费：[YoloVision YOLOv8n Pose 本地包消费教程](yolovision-yolov8n-pose-local-package-consumer-tutorial.md)

### `yolovision-yolov8n-obb-v8.3.0`

- 获取 URL：<https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-obb.pt>
- 固定 revision：`ultralytics-v8.3.0@6e43d1e1e5db72afbf686dee6745669bcb124b0a`
- 获取脚本：`eng/Acquire-YoloV8ObbOfficialAssets.ps1`
- 转换类型：`local-weight-export`
- 转换命令：`yolo export model=yolov8n-obb.pt format=onnx imgsz=1024 opset=17 simplify=True dynamic=False batch=1 device=cpu`
- 工具链：`Ultralytics 8.4.21; opset 17`
- ONNX 暂存：`models/YoloVision/OrientedBoundingBox/yolov8n-obb-ultralytics-v8.3.0/yolov8n-obb.onnx`
- SHA256：`5f2701ef5326fb5a691999438cfc55a69656323c21ffddebaff8968ab6de2e92`

### `yolovision-lraspp-mobilenet-v3-large-v0.25.0`

- 获取 URL：<https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth>
- 固定 revision：`torchvision-v0.25.0@8ac84ee75afb1c327902156b5336f56ad63b7e2f`
- 获取脚本：`eng/Acquire-TorchVisionLrasppOfficialAssets.ps1`
- 转换类型：`local-weight-export`
- 转换命令：`python eng/Invoke-YoloVisionSemanticReference.py --weights <models>/lraspp_mobilenet_v3_large-d234d4ea.pth --image <downloads>/dog.jpg --onnx <models>/lraspp-mobilenet-v3-large-320.onnx --output-directory <artifacts> --export-onnx`
- 工具链：`PyTorch 2.10.0+cpu; torchvision 0.25.0+cpu; opset 17`
- ONNX 暂存：`models/YoloVision/SemanticSegmentation/lraspp-mobilenet-v3-large-torchvision-v0.25.0/lraspp-mobilenet-v3-large-320.onnx`
- 本地包消费：[YoloVision LRASPP 语义分割本地包消费教程](yolovision-lraspp-semantic-local-package-consumer-tutorial.md)
- SHA256：`3cb94e561bdefe606ed7d1a2c4d0296409bec066f3a39a9fe9dabd72b23728f8`

## Classification：TorchVision ResNet18

清单 ID：`classification-resnet18-imagenet1k-v1`。

- 权重：TorchVision `ResNet18_Weights.IMAGENET1K_V1`。
- 固定源码：torchvision `v0.25.0`，commit `8ac84ee75afb1c327902156b5336f56ad63b7e2f`。
- 获取地址：`https://download.pytorch.org/models/resnet18-f37072fd.pth`。
- 权重 SHA256：`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。
- 转换：PyTorch `2.10.0+cpu`、torchvision `0.25.0+cpu`、opset 17，固定 `images:[1,3,224,224] -> logits:[1,1000]`。

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Acquire-TorchVisionResNet18OfficialAssets.ps1 `
  -AllowDownload `
  -ExportOnnx `
  -PythonPath C:\Users\guoji\.conda\envs\ultralytics\python.exe
```

脚本调用 `eng/Export-ClassificationResNet18Onnx.py`，同时生成 `imagenet1k.names` 和导出报告。当前 ONNX 长度是 `46,748,553` bytes，SHA256 是 `ead3558569edd88aa73a4eb46acbe6c38dee113933234547f04a0f6e48169903`。

`eng/Invoke-ClassificationResNet18Reference.py` 使用精确 C# 预处理 tensor 生成 ONNX Runtime raw logits 与任务级 Softmax
reference，并与 PyTorch CPU 交叉比较。本机 TensorRT 10.11 已同时完成 1000 个 raw logits 和 1000 个 probabilities 的
全量比较，两层 mismatch 均为 0，`outputValidated=true`，Top-1 为 `Samoyed`（`0.8799871`）。单值篡改负例得到 exit code
1、mismatch 1、first mismatch 0。证据见 `samples/assets/classification-resnet18-real-model-runtime-evidence.json`；它是
source-tree `real-model-runtime`，不是 package consumer、公开包、再分发授权或发布后证明。

## OnnxToEngine：NVIDIA MNIST

清单 ID：`onnxtoengine-nvidia-mnist-opset8`。

该模型由 TensorRT 的 `data/mnist` 样例数据直接提供，目录内 README 将它标注为 ONNX Model Zoo 的 opset 8 模型。它已经是 ONNX，因此没有本地框架到 ONNX 的转换步骤；只从用户安装的 TensorRT SDK 样例目录复制并核对 hash：

```powershell
$source = Join-Path $env:TENSORRT_PATH 'data\mnist\mnist.onnx'
$target = 'E:\GitSpace\TensorRT-CSharp-API-4.0\models\OnnxToEngine\MNIST\nvidia-tensorrt-10.11\mnist.onnx'
New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
Copy-Item -LiteralPath $source -Destination $target
Get-FileHash -Algorithm SHA256 -LiteralPath $target
```

项目当前固定副本为 `26,454` bytes，SHA256 `2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf`。若用户安装包不包含 `data`，可以从 NVIDIA TensorRT sample data 或 ONNX Model Zoo 获取，但必须先核对同一 hash，不能把名字相同但图契约不同的文件混用。

当前 `OnnxToEngine --mnist` 已用外层 `models` 中的同一 ONNX 和 TensorRT `7.pgm` 完成 TensorRT 10.11 源树实跑：
预测 digit 7、置信度 `0.99999285`，10 个 logits 对独立 ONNX Runtime 1.23.2 CPU reference 为 mismatch 0，最大绝对
误差 `0.000006`。只把 `expectedDigit` 改为 6 的受控负例完成 enqueue 后返回 exit code 2、
`State=mnist-output-mismatch`、`OutputMatch=False`。小型证据见
`samples/assets/onnxtoengine-mnist-real-model-runtime-evidence.json`；模型、engine 和运行日志均不上传。

## YOLOv8n Detection

清单 ID：`yolovision-yolov8n-detection-v8.3.0`。

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Acquire-YoloV8DetectionOfficialAssets.ps1 -AllowDownload
yolo export model=yolov8n.pt format=onnx imgsz=640 opset=17 simplify=True dynamic=False batch=1 device=cpu
```

权重来自 Ultralytics assets `v8.3.0`，源码固定到 commit `6e43d1e1e5db72afbf686dee6745669bcb124b0a`。ONNX 为 `12,851,047` bytes，SHA256 `db28a49ffbb0425f39ae56252e7e0b43d06b357416c7da58872e285560b4221e`。完整步骤见 `yolovision-detection-yolov8n-download-export-run.md`。

## YOLOv10n Detection

清单 ID：`yolovision-yolov10n-detection-v1.1`。

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Acquire-YoloV10OfficialAssets.ps1 -AllowDownload
```

该用例直接使用 THU-MIG YOLOv10 `v1.1` 发布的 `yolov10n.onnx`，不再做本地转换。固定 revision 是 `799ff3be47d21173bcf29b351820d4b8e955e0fe`；文件为 `9,386,466` bytes，SHA256 `7025ea1913f9a259cf8a8465ed608e10610d1bb376db2e0348b13e3bd286e0d3`。

## YOLOX-S Detection

清单 ID：`yolovision-yolox-s-detection-0.1.1rc0`。

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Acquire-YoloXOfficialAssets.ps1 -AllowDownload
```

运行用例直接使用 Megvii YOLOX `0.1.1rc0` 官方 ONNX 发布资产，不需要再次转换。需要从权重复现时，上游命令是：

```powershell
python3 tools/export_onnx.py --output-name yolox_s.onnx -n yolox-s -c yolox_s.pth
```

固定 revision 是 `e1052df71842031413f6030723c3607b839c80ce`；文件为 `35,858,002` bytes，SHA256 `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`。

## YOLOv8n Classification

清单 ID：`yolovision-yolov8n-classification-v8.3.0`。

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Acquire-YoloV8ClassificationOfficialAssets.ps1 -AllowDownload
yolo export model=yolov8n-cls.pt format=onnx imgsz=224 opset=17 simplify=True dynamic=False batch=1 device=cpu
```

ONNX 为 `10,911,331` bytes，SHA256 `630c022a99885d59f633ab5a614738f8a49be7f361e340fd3ff89b8c19b0768f`。模型输出已包含 Softmax，不能再把它当 raw logits 重复做 Softmax。

## YOLOv8n Instance Segmentation

清单 ID：`yolovision-yolov8n-instance-segmentation-v8.3.0`。

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Acquire-YoloV8SegOfficialAssets.ps1 -AllowDownload
yolo export model=yolov8n-seg.pt format=onnx imgsz=640 opset=17 simplify=True dynamic=False batch=1 device=cpu
```

ONNX 为 `13,873,432` bytes，SHA256 `08b5c61368d4ddec5e647522fc55a93c42a9e0c581770aae48b87bba65a9b21d`。模型有 detection rows 与 mask prototypes 两个输出，不能只按普通 detection 图处理。

## YOLOv8n Pose

清单 ID：`yolovision-yolov8n-pose-v8.3.0`。

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Acquire-YoloV8PoseOfficialAssets.ps1 -AllowDownload
yolo export model=yolov8n-pose.pt format=onnx imgsz=640 opset=17 simplify=True dynamic=False batch=1 device=cpu
```

ONNX 为 `13,514,570` bytes，SHA256 `ed1e8d2d2aeb8a2c66e642a16295a72a2990393e3a3843325537da7e11c8a899`；输出包含 17 个关键点，每个关键点 3 个值。

## YOLOv8n OBB

清单 ID：`yolovision-yolov8n-obb-v8.3.0`。

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Acquire-YoloV8ObbOfficialAssets.ps1 -AllowDownload
yolo export model=yolov8n-obb.pt format=onnx imgsz=1024 opset=17 simplify=True dynamic=False batch=1 device=cpu
```

ONNX 为 `12,664,838` bytes，SHA256 `5f2701ef5326fb5a691999438cfc55a69656323c21ffddebaff8968ab6de2e92`；输入是 `1024x1024`，角度通道使用弧度。

## LRASPP Semantic Segmentation

清单 ID：`yolovision-lraspp-mobilenet-v3-large-v0.25.0`。

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Acquire-TorchVisionLrasppOfficialAssets.ps1 `
  -AllowDownload `
  -ExportOnnx `
  -PythonPath C:\Users\guoji\.conda\envs\ultralytics\python.exe
```

权重来自 TorchVision LRASPP MobileNetV3 Large，固定到 torchvision `v0.25.0`。转换脚本 `eng/Invoke-YoloVisionSemanticReference.py` 导出 opset 17、`images:[1,3,320,320] -> semantic:[1,21,320,320]`。ONNX 为 `12,879,801` bytes，SHA256 `3cb94e561bdefe606ed7d1a2c4d0296409bec066f3a39a9fe9dabd72b23728f8`。

## 真实运行证据索引

机器清单的 10 个模型均显式链接到已经存在的小型 source-tree runtime 证据；这些记录不包含模型文件，也不等同于
package consumer、公开包、post-publish 或再分发授权。

| 清单 ID | 真实运行证据 |
| --- | --- |
| `classification-resnet18-imagenet1k-v1` | `samples/assets/classification-resnet18-real-model-runtime-evidence.json` |
| `onnxtoengine-nvidia-mnist-opset8` | `samples/assets/onnxtoengine-mnist-real-model-runtime-evidence.json` |
| `yolovision-yolov8n-detection-v8.3.0` | `samples/assets/yolovision-yolov8n-det-real-model-runtime-evidence.json` |
| `yolovision-yolov10n-detection-v1.1` | `artifacts/interface-coverage/yolov10-official-runtime-proof-closure.json` |
| `yolovision-yolox-s-detection-0.1.1rc0` | `artifacts/interface-coverage/yolox-official-runtime-proof-closure.json` |
| `yolovision-yolov8n-classification-v8.3.0` | `samples/assets/yolovision-yolov8n-cls-real-model-runtime-evidence.json` |
| `yolovision-yolov8n-instance-segmentation-v8.3.0` | `samples/assets/yolovision-yolov8n-seg-real-model-runtime-evidence.json` |
| `yolovision-yolov8n-pose-v8.3.0` | `samples/assets/yolovision-yolov8n-pose-real-model-runtime-evidence.json` |
| `yolovision-yolov8n-obb-v8.3.0` | `samples/assets/yolovision-yolov8n-obb-real-model-runtime-evidence.json` |
| `yolovision-lraspp-mobilenet-v3-large-v0.25.0` | `samples/assets/yolovision-torchvision-lraspp-real-model-runtime-evidence.json` |

## 不需要外部深度学习模型的样例

`DynamicShape`、`InferenceBindings`、`MultiStream`、CUDA RTC 与多数接口 smoke 在代码中创建最小网络或只读取 metadata，不依赖下载的深度学习模型。`OnnxToEngine` 的默认 identity 图也是进程内生成的教学图，不属于需要进入 Model Zoo 的真实模型。自定义 `classifier.onnx`、`image-and-metadata.onnx`、`model.onnx` 等路径是用户替换占位符，不是项目声称已经提供的模型。

## 发布边界

1. GitHub 只上传源码、C# managed 包和按 TensorRT/CUDA 版本编译的 bridge-only 包。
2. 所有模型权重、ONNX、TensorRT engine、输入图片和大体积 reference 留在外层工作区。
3. 模型 SHA256 只证明本地文件身份，不等于获得公开再分发授权。
4. CUDA、cuDNN、TensorRT 与 NVRTC 始终由用户自行安装。
5. 独立 Model Zoo 建立前，不创建任何含模型二进制的 Release asset 或 NuGet 包。
