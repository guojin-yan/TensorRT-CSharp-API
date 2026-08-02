# YoloVision LRASPP 语义分割本地包消费教程

本文验证第一版发布前最接近真实用户的语义分割安装路径：消费项目位于源码仓库外，只通过 `PackageReference` 引用 managed API、YoloVision 和一个 bridge-only 包；CUDA、cuDNN 与 TensorRT 由用户自行安装。模型与运行资产不进入 Git、NuGet 或 GitHub Release。

## 模型来源与许可证

本例固定使用 torchvision `v0.25.0` 的 LRASPP MobileNetV3 Large：

| 项目 | 固定值 |
| --- | --- |
| 权重 | `lraspp_mobilenet_v3_large-d234d4ea.pth` |
| 获取 URL | `https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth` |
| torchvision revision | `v0.25.0@8ac84ee75afb1c327902156b5336f56ad63b7e2f` |
| 权重 SHA256 | `d234d4eae9d55d5f76de18b77cf0dc62c66fe5c5482758209d00f950c92bb280` |
| 许可证 | `BSD-3-Clause` |
| 输入图片 | PyTorch Hub `dog.jpg@c7895df70c7767403e36f82786d6b611b7984557` |

执行固定资产获取脚本：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Acquire-TorchVisionLrasppOfficialAssets.ps1 `
  -AllowDownload `
  -PythonPath C:\Users\<user>\.conda\envs\ultralytics\python.exe
```

获取不等于允许本项目公开再分发。当前权重与图片的 `publicRedistributionOwnerApproval` 均为 `false`。

## 转换 ONNX

所有演示 ONNX 统一暂存到 Git 仓库外的：

```text
E:\GitSpace\TensorRT-CSharp-API-4.0\models
```

LRASPP 的固定目录为：

```text
models\YoloVision\SemanticSegmentation\lraspp-mobilenet-v3-large-torchvision-v0.25.0
```

使用源码仓库内的结构化导出脚本完成转换和 ONNX Runtime 参考生成：

```powershell
$workspace = "E:\GitSpace\TensorRT-CSharp-API-4.0"
$modelRoot = "$workspace\models\YoloVision\SemanticSegmentation\lraspp-mobilenet-v3-large-torchvision-v0.25.0"
$assetRoot = "$workspace\downloads\lraspp-mobilenet-v3-large-torchvision-v0.25.0\source"
$referenceRoot = "$workspace\TensorRtSharp4.0\artifacts\yolovision\semantic-lraspp-reference"

& C:\Users\<user>\.conda\envs\ultralytics\python.exe `
  .\eng\Invoke-YoloVisionSemanticReference.py `
  --weights "$modelRoot\lraspp_mobilenet_v3_large-d234d4ea.pth" `
  --image "$assetRoot\dog.jpg" `
  --onnx "$modelRoot\lraspp-mobilenet-v3-large-320.onnx" `
  --output-directory $referenceRoot `
  --export-onnx
```

脚本通过 wrapper 只导出 `model(images)["out"]`，opset 固定为 17，静态合同必须是：

```text
images:[1,3,320,320] -> semantic:[1,21,320,320]
```

转换后的 ONNX 长度为 `12,879,801`，SHA256 为 `3cb94e561bdefe606ed7d1a2c4d0296409bec066f3a39a9fe9dabd72b23728f8`。它只保存在外层 `models`，为后续独立 Model Zoo 暂存；不能执行 `git add`，也不能塞进任何 NuGet 包。

## 构建本地三包

```powershell
dotnet pack .\pack\JYPPX.TensorRT.CSharp.API\JYPPX.TensorRT.CSharp.API.csproj `
  -c Release -o .\artifacts\managed `
  -p:JYPPXPackageVersion=4.0.0 -p:UseSharedCompilation=false

dotnet pack .\samples\YoloVision\YoloVision.csproj `
  -c Release -o .\artifacts\yolovision-nupkg `
  -p:JYPPXPackageVersion=4.0.0 -p:UseSharedCompilation=false

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 `
  -SourceRuntimeKey win-x64-trt10.11-cuda12.9-cudnn9.22 `
  -Version 4.0.0 -SplitPackageRole bridge `
  -SkipManagedPack -SkipConsumerValidation
```

第三个包只能是 `.Bridge`。脚本会拒绝包含 `nvinfer`、`nvonnxparser`、`cudart`、`cudnn`、`nvrtc` 等 NVIDIA vendor runtime 的包集合。

## 运行 clean consumer

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Test-YoloVisionSemanticLocalPackageConsumer.ps1 `
  -RepositoryRoot $PWD `
  -PackageVersion 4.0.0
```

runner 在外层 E 盘创建临时项目，并生成只有三个 `PackageReference` 的 `NuGet.config` 和项目文件。它不会设置 `JYPPX_NATIVE_BRIDGE_PATH`，bridge 必须由 NuGet 的 `runtimes/win-x64/native` 资产复制；TensorRT、CUDA、cuDNN 路径来自用户安装目录。运行结束后临时工作区必须被删除。

实际模型参数为 `RGB + NCHW + stretch 320x320 + scale 1/255 + ImageNet mean/std`，并关闭 TF32。验证要求：

- `semantic` 的 `2,150,400` 个 logits 与 ONNX Runtime reference 在 `abs=1e-4`、`rel=1e-4` 下零 mismatch。
- 完整 `320x320` 类别索引共 `102,400` 个像素，与 ONNX Runtime 的 int32 little-endian 参考哈希完全一致。
- 直方图只能包含 `65,193` 个 background 像素与 `37,207` 个 dog 像素。
- 原始 reference 索引 0 增加 `10` 后，消费者必须退出 `1`，报告一个 mismatch，`firstMismatchIndex=0`。
- 类别图首字节篡改且 manifest SHA256 不变时，严格校验器必须退出 `1` 并报告 `artifact-sha256`。

本机 TensorRT 10.11 结果记录在：

- `samples/assets/yolovision-lraspp-semantic-local-package-consumer-runtime-evidence.json`
- `eng/Test-YoloVisionSemanticMapArtifact.ps1`
- `eng/Export-YoloVisionSemanticLocalPackageConsumerEvidence.ps1`

## 证据边界

该记录是 `local-package-consumer-runtime` 工程证据，证明本地生成的三个包可以由仓库外项目 restore、build 和运行。它不是 nuget.org 下载证明，不是公开包证明，不是模型再分发授权，不是 post-publish 证明，也不授权创建 tag、Release 或推送包。

模型的统一获取、转换与外层暂存清单见[演示模型获取、ONNX 转换与本地暂存目录](demo-model-acquisition-and-onnx-conversion.md)，语义张量、布局、argmax 和 artifact 合同见[LRASPP 语义分割图指南](yolovision-semantic-segmentation-map-guide.md)。
