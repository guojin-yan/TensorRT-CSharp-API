# YoloVision YOLOv8n 分类本地包消费教程

本文验证官方 YOLOv8n 分类模型的仓库外 `PackageReference` 路径。消费项目只引用 managed API、YoloVision 与一个 bridge-only 包；CUDA、cuDNN 和 TensorRT 由用户自行安装，模型不进入 Git、NuGet 或 GitHub Release。

## 获取模型

固定资产为 Ultralytics `v8.3.0`：

| 资产 | 来源与固定值 |
| --- | --- |
| 权重 | `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-cls.pt` |
| revision | `ultralytics-v8.3.0@6e43d1e1e5db72afbf686dee6745669bcb124b0a` |
| 权重 SHA256 | `11fa19f2aea79bc960d680a13f82f22105982b325eb9e17a4a5e1a9f8245980a` |
| 许可证 | `AGPL-3.0-only` |
| 输入图片 | Ultralytics `bus.jpg`，SHA256 `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63` |

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Acquire-YoloV8ClassificationOfficialAssets.ps1 `
  -AssetRoot E:\GitSpace\TensorRT-CSharp-API-4.0\downloads\yolov8n-cls-ultralytics-v8.3.0 `
  -PythonPath C:\Users\<user>\.conda\envs\ultralytics\python.exe
```

权重、图片和导出模型均未获得本项目公开再分发批准。

## 转换 ONNX

导出命令固定为：

```powershell
yolo export `
  model=E:\GitSpace\TensorRT-CSharp-API-4.0\downloads\yolov8n-cls-ultralytics-v8.3.0\source\yolov8n-cls.pt `
  format=onnx imgsz=224 opset=17 simplify=True dynamic=False batch=1 device=cpu
```

转换后的模型必须暂存在 Git 仓库外：

```text
E:\GitSpace\TensorRT-CSharp-API-4.0\models\YoloVision\Classification\yolov8n-cls-ultralytics-v8.3.0\yolov8n-cls.onnx
```

固定 ONNX 长度为 `10,911,331`，SHA256 为 `630c022a99885d59f633ab5a614738f8a49be7f361e340fd3ff89b8c19b0768f`。静态图合同是 `images:[1,3,224,224] -> output0:[1,1000]`，最后一个 ONNX 节点必须是 `Softmax`。

使用独立参考脚本生成 labels、权威输入 tensor、1,000 值 reference 和受控负例：

```powershell
& C:\Users\<user>\.conda\envs\ultralytics\python.exe `
  .\eng\Invoke-YoloVisionClassificationReference.py `
  --weights E:\GitSpace\TensorRT-CSharp-API-4.0\downloads\yolov8n-cls-ultralytics-v8.3.0\source\yolov8n-cls.pt `
  --imagenet-yaml E:\GitSpace\TensorRT-CSharp-API-4.0\downloads\yolov8n-cls-ultralytics-v8.3.0\source\ImageNet.yaml `
  --image E:\GitSpace\TensorRT-CSharp-API-4.0\downloads\yolov8n-cls-ultralytics-v8.3.0\source\bus.jpg `
  --onnx E:\GitSpace\TensorRT-CSharp-API-4.0\models\YoloVision\Classification\yolov8n-cls-ultralytics-v8.3.0\yolov8n-cls.onnx `
  --output-directory E:\GitSpace\TensorRT-CSharp-API-4.0\downloads\yolov8n-cls-ultralytics-v8.3.0\reports\independent-reference
```

## 构建与运行三个包

先按[LRASPP 语义分割本地包消费教程](yolovision-lraspp-semantic-local-package-consumer-tutorial.md)中的命令生成 managed API、YoloVision 和对应 `.Bridge` 包，然后执行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Test-YoloVisionClassificationLocalPackageConsumer.ps1 `
  -RepositoryRoot $PWD `
  -PackageVersion 4.0.0
```

runner 不直接使用三个可能包含同 ID/版本重复文件的大目录。它先把选定的三个 nupkg 分别复制到隔离 feed，restore 后再比较 NuGet 缓存中的 nupkg SHA256，确保实际消费包与证据包完全相同。

运行合同为权威 `1x3x224x224` RGB/NCHW center-crop tensor、`classificationScoreMode=probabilities`、`ApplyNms=False`、`NmsMode=None`、`TopK=5`。严格结果为：

- `output0` 全部 `1,000` 个概率与 ONNX Runtime reference 在 `abs=0.001`、`rel=0.001` 下零 mismatch。
- Top-5 顺序固定为 `minibus / police_van / trolleybus / golfcart / jinrikisha`。
- reference 索引 0 增加 `0.125` 后必须退出 `1`，产生一个 mismatch，`firstMismatchIndex=0`。
- 三个包内 NVIDIA vendor runtime 条目必须为 0，bridge 包只能含 `jyppxtrtbridge.dll`。

紧凑记录位于 `samples/assets/yolovision-yolov8n-cls-local-package-consumer-runtime-evidence.json`。它是 `local-package-consumer-runtime` 工程证据，不是 nuget.org 下载、public-package、post-publish、模型再分发、Owner 发布批准或 release proof。

所有演示模型的获取、转换、外层路径与哈希见[演示模型获取、ONNX 转换与本地暂存目录](demo-model-acquisition-and-onnx-conversion.md)。
