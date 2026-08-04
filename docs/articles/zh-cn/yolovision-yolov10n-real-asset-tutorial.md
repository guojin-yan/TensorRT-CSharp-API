# 使用 TensorRtSharp4.0 在 C# 中运行 YOLOv10n：从官方模型到本地三包实机验证

本文以 THU-MIG 官方 YOLOv10n v1.1 ONNX 为例，完整演示模型获取、ONNX 转换与合同确认、图片预处理、TensorRT 推理、end-to-end 输出解码、原图结果绘制，以及 managed、YoloVision、bridge-only 三个本地候选包的隔离消费验证。

本次验证使用真实模型、CC0 图片和 TensorRT 10.11 实机执行。CUDA、cuDNN、TensorRT 由使用者按版本安装，项目包不携带 NVIDIA 厂商运行库；ONNX、engine 和预处理 tensor 暂存在仓库外层 `models` 或工作目录，不上传 GitHub。

## 1. 项目、功能与依赖库

TensorRtSharp4.0 的顶层托管命名空间是 `JYPPX.TensorRtSharp` 和 `JYPPX.CudaSharp`。本案例使用四层能力：

| 层次 | 项目或包 | 本案例职责 |
| --- | --- | --- |
| TensorRT 托管接口 | `JYPPX.TensorRtSharp` | 解析 ONNX、创建 engine/context、绑定 tensor、enqueue 并读取输出 |
| CUDA 托管接口 | `JYPPX.CudaSharp` | 管理 CUDA context、stream 和 device memory 生命周期 |
| 视觉任务接口 | `JYPPX.TensorRT.CSharp.API.YoloVision` | 图片预处理、模型配置、检测解码、JSON 报告和 SVG 可视化 |
| 本地包消费者 | `samples/YoloVision.PackageConsumer` | 只通过三个 `PackageReference` 调用 YoloVision，不引用源码项目 |

YOLOv10n v1.1 的关键点不是模型文件名，而是输出合同。本例的 `output0` 为 `[1,300,6]`，每行依次是：

```text
x1, y1, x2, y2, score, classId
```

模型图内已经完成候选筛选和 NMS，因此应用端必须使用 `EndToEndNms` 布局，并保持 `ApplyNms=false`、`NmsMode=None`。如果实际 ONNX 输出不是六列合同，就不能套用本文配置。

## 2. 模型与输入图片

### 2.1 官方模型

模型来自 [THU-MIG YOLOv10 v1.1 Release](https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.onnx)：

| 项目 | 固定值 |
| --- | --- |
| 上游 revision | `799ff3be47d21173bcf29b351820d4b8e955e0fe` |
| 许可证 | `AGPL-3.0-only` |
| 文件 | `yolov10n.onnx` |
| 长度 | `9,386,466` bytes |
| SHA256 | `7025ea1913f9a259cf8a8465ed608e10610d1bb376db2e0348b13e3bd286e0d3` |

项目不再把模型或 NVIDIA 运行库打入 GitHub/Release 包。模型暂存位置为：

```text
models/YoloVision/Detection/yolov10n-thu-mig-v1.1/yolov10n.onnx
```

这里的 `models` 是仓库外层模型工作区，后续可迁移到独立 Model Zoo。机器可读来源和许可证记录位于 `samples/assets/yolovision-yolov10-official-assets.json`。

### 2.2 CC0 输入图片

实机输入为 Wikimedia Commons 的 `Liverpool Street Bus station 2025`，许可证为 [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)。运行使用 1280×961 PPM，SHA256 为：

```text
80715af66669147b049fec9386152bd45505f4e494a65079fdc55404d4589b8e
```

模型因 AGPL 再分发尚未获得项目所有者批准而不提交；结果图使用可公开再分发的 CC0 输入。

## 3. 获取模型

仓库提供哈希固定的获取脚本。输出目录应位于仓库外层：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File ./eng/Acquire-YoloV10OfficialAssets.ps1 `
  -OutputRoot <downloads-root>/yolov10-agpl
```

脚本会下载官方 ONNX 与许可证、校验长度和 SHA256，并生成本地获取报告。将校验通过的 ONNX 暂存到上一节的 `models/YoloVision/Detection/...` 目录即可；不要把模型复制进 Git 仓库。

## 4. 从 checkpoint 转换 ONNX

本文实机数据使用官方已发布 ONNX，因此复现本次哈希不需要再次转换。如果业务模型来自 checkpoint，应固定官方源码 revision、Python、PyTorch、导出器版本、opset 和输入尺寸，再执行上游导出：

```powershell
git clone https://github.com/THU-MIG/yolov10.git <work-root>/yolov10
git -C <work-root>/yolov10 checkout 799ff3be47d21173bcf29b351820d4b8e955e0fe
python -m pip install -r <work-root>/yolov10/requirements.txt
```

```python
from ultralytics import YOLOv10

model = YOLOv10("<model-root>/yolov10n.pt")
model.export(format="onnx", imgsz=640, opset=13, simplify=True)
```

转换后必须先检查实际 ONNX：

```text
images:[1,3,640,640]
output0:[1,300,6]
```

重新导出的文件不保证与 v1.1 Release 文件同哈希，也可能导出 raw head。若输出为 `[1,84,8400]`、`[1,8400,84]` 或其他布局，应根据真实 tensor metadata 选择 decoder，不能仅凭“YOLOv10”名称推断。

## 5. 准备本机运行环境

本案例验证组合为 TensorRT `10.11.0`、CUDA Toolkit `12.9`、Windows x64 和 .NET 8。cuDNN、TensorRT、CUDA 由用户安装；bridge-only 包只包含本项目编译的 `jyppxtrtbridge.dll`。

构建候选包时使用：

```powershell
dotnet pack ./pack/JYPPX.TensorRT.CSharp.API/JYPPX.TensorRT.CSharp.API.csproj `
  -c Release -o ./artifacts/managed -p:JYPPXPackageVersion=4.0.0

dotnet pack ./samples/YoloVision/YoloVision.csproj `
  -c Release -o ./artifacts/yolovision-nupkg -p:JYPPXPackageVersion=4.0.0

pwsh -NoProfile -ExecutionPolicy Bypass `
  -File ./eng/Invoke-LocalSplitRuntimePackage.ps1 `
  -SourceRuntimeKey win-x64-trt10.11-cuda12.9-cudnn9.22 `
  -Version 4.0.0 -SplitPackageRole bridge `
  -SkipManagedPack -SkipBaseRuntimeBuild -SkipConsumerValidation
```

这些命令只生成本地候选包，不执行 `push`、不创建 tag、Release 或 GitHub Package。

## 6. 创建隔离的三包消费者

`samples/YoloVision.PackageConsumer` 的项目文件模板只引用三个包：

```xml
<ItemGroup>
  <PackageReference Include="JYPPX.TensorRT.CSharp.API" Version="4.0.0" />
  <PackageReference Include="JYPPX.TensorRT.CSharp.API.YoloVision" Version="4.0.0" />
  <PackageReference
    Include="JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge"
    Version="4.0.0" />
</ItemGroup>
```

验证脚本会为每个包创建只含一个 `.nupkg` 的隔离文件源，生成 `<clear />` NuGet 配置，再在仓库外工作目录执行 restore/build/run。它还会检查：

1. restore graph 中 `ProjectReference` 数量为 0。
2. 恢复到缓存的三个 `.nupkg` 与选中包 SHA256 一致。
3. bridge 由 NuGet 复制到输出目录，且未设置 `JYPPX_NATIVE_BRIDGE_PATH` 绕过包布局。
4. 三个包内 NVIDIA 厂商运行库数量为 0，bridge native 文件数量为 1。
5. restore、build、runtime 退出码均为 0。

## 7. 图片预处理与输出解码

输入预处理合同如下：

| 项目 | 值 |
| --- | --- |
| 原图 | `1280×961` |
| 目标尺寸 | `640×640` |
| resize | `640×480` |
| padding | `x=0, y=80`，居中 |
| color/layout | `RGB / NCHW` |
| scale | `1/255` |
| fill | `114` |

生成的 float32 tensor 共 `1,228,800` 个值、`4,915,200` bytes，SHA256 为：

```text
050935ebf471ec32ab4327d9f5643f0fe1a203289088895205e732e448a8d225
```

程序入口通过 `YoloImagePreprocessor` 生成 tensor，随后调用 TensorRT 托管接口并交给 end-to-end decoder：

```csharp
YoloImagePreprocessResult imagePreprocess = YoloImagePreprocessor.Preprocess(
    imagePath,
    tensorPath,
    profile.InputShape,
    profile.Preprocess);

OnnxSampleMultiOutputResult runtime =
    TensorRtOnnxSample.RunSingleFloatInputOutputs(options);

YoloVisionResult result = YoloSampleRunner.DecodeOutput(
    runtime.PrimaryOutput.Values,
    runtime.PrimaryOutput.Shape.Values,
    profile);
```

`YoloDetectionDecoder.DecodeEndToEnd` 会检查 rank、六列长度、有限坐标、`x2>x1`、`y2>y1`、score 范围、整数 class id 和 class count。检测框再按 `scale=0.5`、`padY=80` 反变换回原图坐标。

## 8. 执行本地包实机验证

准备好本机 TensorRT、CUDA、cuDNN 根目录后执行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File ./eng/Test-YoloVisionLocalPackageConsumer.ps1 `
  -Scenario yolov10-detection `
  -RuntimePackageKey win-x64-trt10.11-cuda12.9-cudnn9.22 `
  -TensorRtRoot <TensorRT-root> `
  -CudaRoot <CUDA-root> `
  -CudnnRoot <cuDNN-root>
```

脚本内部传给 YoloVision 的核心参数为：

```text
--input-shape 1x3x640x640
--input-name images
--output-name output0
--family v10
--task det
--layout end2end
--class-count 80
--confidence 0.25
--top-k 100
```

## 9. 实际运行结果

本次运行环境为 RTX 3060 Laptop GPU、驱动 `576.02`、TensorRT `10.11.0`、CUDA Toolkit `12.9`、.NET SDK `10.0.301`。结果不是模板或 dry-run：真实执行了图片预处理、TensorRT enqueue、输出读取和托管解码。

![YOLOv10n 本地三包消费原图检测结果](../../images/yolovision-yolov10n-local-package-consumer-annotated-cc0.jpg)

上图由同次 `yolovision-output.json` 中的 6 个检测框和 letterbox 参数反投影到 CC0 原图生成。编号与顶部图例对应，避免右侧密集 person 框的文字互相遮挡。

![YOLOv10n 本地三包消费实际终端窗口](../../images/yolovision-yolov10n-local-package-consumer-terminal.png)

实际结果为：

| 类别 | 数量 | 最高置信度 |
| --- | ---: | ---: |
| bus | 1 | `0.950415` |
| person | 5 | `0.804006` |

关键运行事实：

```text
ProjectReference=False
BridgeTensorRt=10.11.0 BridgeCuda=12.9
Input=images:[1,3,640,640] Output=output0:[1,300,6]
Layout=EndToEndNms Nms=False NmsMode=None
Execution ElapsedMs=5.596
Postprocess Detections=6
YoloVision Passed=True
```

输出 JSON SHA256 为 `369878a520f0256000c57892f12bc72f97d8a3c2e4e6a94ed3c1fa774a35173e`，脱敏文本和机器证据分别位于：

```text
samples/assets/yolovision-yolov10n-local-package-consumer-tensorrt10.11.txt
samples/assets/yolovision-yolov10n-local-package-consumer-runtime-evidence.json
```

## 10. 证据边界与复查

这次运行证明的是当前源码对应候选包的 `local-package-consumer-runtime`：三个本地包可被隔离 restore/build，bridge 能由 NuGet 布局加载，官方 YOLOv10n 能完成真实 TensorRT 推理，并生成可复查的检测结果。

本次没有可用的独立 raw tensor reference，因此不声称跨框架 raw output 或后处理逐值一致。固定输出 shape、预处理 tensor SHA256、类别分布和结果图均已校验，但它们不能替代独立参考。

它也不证明公开 NuGet 下载、GitHub Package、post-publish、Owner 发布验收或 AGPL 模型公开再分发授权。本次没有创建 tag、Release，也没有发布任何包。正式发布前仍应在目标机器确认：

1. TensorRT、CUDA、cuDNN 与 bridge-only 包版本组合一致。
2. 实际 ONNX 的输入输出名称、shape 和六列顺序符合合同。
3. `ProjectReference=False`，恢复包哈希与选中候选包一致。
4. `ApplyNms=false`，不对 end-to-end 结果执行第二次 NMS。
5. bus 框覆盖车辆主体，5 个 person 框位于站台右侧，坐标无整体偏移。
6. 进程退出码为 0，最终标记为 `YoloVision Passed=True`。
