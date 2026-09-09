# 图像分类案例

[English](README.md) | 简体中文

该案例把用户提供的 float ONNX 分类模型交给 TensorRT，完成图像解码、resize/crop、归一化、推理、Top-K、结构化 JSON 和标注结果图输出。项目本身是 `IsPackable=false` 的演示程序，通过 NuGet 使用 4 系列 `JYPPX.TensorRT.CSharp.API` 和项目自有 `JYPPX.OpenCV.CSharp.API`，不会发布 Classification 扩展包。

## 基准模型

可复现基准为 TorchVision ResNet18 `IMAGENET1K_V1`。获取脚本会下载固定权重、标签和许可证，并导出 opset 17 的 `images:[1,3,224,224] -> logits:[1,1000]` ONNX：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Acquire-TorchVisionResNet18OfficialAssets.ps1 `
  -AllowDownload `
  -ExportOnnx `
  -PythonPath python
```

模型、权重、标签和输入图片保存在源码仓库同级的 `<workspace-root>/models/Classification/resnet18-torchvision-v0.25.0`，不上传 GitHub。来源、许可证、转换工具、输入输出契约和 SHA256 记录在 `samples/assets/classification-resnet18-official-assets.json`，完整转换说明见 [演示模型获取与 ONNX 转换](../../../docs/articles/zh-cn/demo-model-acquisition-and-onnx-conversion.md)。

## 安装边界

外部消费者使用当前 4 系列预览包，不需要在示例代码中写死版本：

```powershell
dotnet add package JYPPX.TensorRT.CSharp.API --version "4.0.0"
dotnet add package JYPPX.OpenCV.CSharp.API --prerelease
dotnet add package JYPPX.OpenCV.runtime.win-x64 --prerelease
dotnet add package JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge --version "4.0.0"
```

最后一个 Bridge ID 只是 Windows x64、TensorRT 10.11、CUDA 12.9 和 cuDNN 9.22 的示例。必须根据目标平台选择对应 Bridge；CUDA、cuDNN 和 TensorRT 由用户自行安装。

## 真实图片运行

```powershell
dotnet run --project .\samples\ComputerVision\01.Classification -- `
  --model ..\models\Classification\resnet18-torchvision-v0.25.0\resnet18-imagenet1k-v1.onnx `
  --labels ..\models\Classification\resnet18-torchvision-v0.25.0\imagenet1k.names `
  --image ..\models\Classification\resnet18-torchvision-v0.25.0\input.jpg `
  --preprocessed-output .\artifacts\classification\input-f32.bin `
  --input-shape 1x3x224x224 `
  --image-resize shorter-side-center-crop `
  --resize-shorter-side 256 `
  --tensor-layout NCHW `
  --color-order RGB `
  --scale 0.0039215689 `
  --mean 0.485,0.456,0.406 `
  --std 0.229,0.224,0.225 `
  --score-transform softmax `
  --top-k 5 `
  --output-json .\artifacts\classification\output.json `
  --visualization .\artifacts\classification\result.svg
```

`--image` 通过 OpenCV 包读取 JPEG/PNG，并通过托管路径读取 BMP/PPM。`--input-data` 用于已经完成预处理的 float32 tensor，`--input` 则读取逐元素字节并归一化到 `[0,1]`，三种输入语义不能混用。

## 输出与验证

`--output-json` 写出输入和预处理指纹、原始/变换后输出、Top-K、运行摘要和证据边界。`--reference-output` 可校验任务级 logits/softmax 结果，`--reference-outputs` 可逐 Tensor 校验原始运行时输出；元数据或数值不匹配会以非零退出码结束。

一篇完整演示必须同时展示：模型获取和转换过程、真实运行命令、终端输出截图、Top-K 解释以及覆盖在原图上的分类结果。当前真实 ResNet18 工作流和两张执行结果图见 [真实分类模型完整实战](../../../docs/articles/zh-cn/classification-real-asset-walkthrough.md)。合成输入只能证明管线可运行，不能证明真实图像分类准确率。
