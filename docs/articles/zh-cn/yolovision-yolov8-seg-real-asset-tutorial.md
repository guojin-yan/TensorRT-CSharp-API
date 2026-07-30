# YOLOv8n Segmentation 真实多输出运行教程

本文记录 `samples/YoloVision` 已执行的 YOLOv8n segmentation 真实案例。它覆盖官方资产获取、ONNX 导出、
TensorRT 多输出构建与加载、ONNX Runtime tensor reference、Ultralytics/PyTorch 独立后处理比较、source-image mask
工件和受控失败验证。该结果可归类为 source-tree `real-model-runtime`，但不是 package-consumer、公开包、
post-publish、Owner 发布验收或模型再分发许可。

## 固定资产

仓库内的轻量记录：

```text
samples/assets/yolovision-yolov8n-seg-official-assets.json
samples/assets/yolovision-yolov8n-seg-real-model-runtime-evidence.json
```

本次资产身份：

| 资产 | 来源与身份 | SHA256 |
| --- | --- | --- |
| `yolov8n-seg.pt` | Ultralytics assets `v8.3.0`, Release asset `195720083` | `a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c` |
| Ultralytics source | tag commit `6e43d1e1e5db72afbf686dee6745669bcb124b0a` | 由 acquisition manifest 固定 |
| AGPL license | 同一 source commit 的 `LICENSE` | `0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0` |
| `dog.ppm` | Apache-2.0 YOLOX case | `6cb94c9cd0781412598fe179246b09041af4303d388a5ba3c55f760dff11ec2c` |
| `coco.names` | Apache-2.0 YOLOX case | `4d4aaea7bee6be2f675d9b53a9195ca36dfe6429f7479f29155da522a6c85930` |

旧 GitHub Release API 没有为该模型返回服务端 digest，因此 acquisition manifest 明确记录“官方 asset ID +
首次下载后仓库 pin”，没有把本地 SHA256 伪称为上游 checksum。模型、图片、engine、reference 和 mask 二进制
都保留在 E 盘工作区，不提交、不打包、不发布。

```powershell
.\eng\Acquire-YoloV8SegOfficialAssets.ps1 `
  -DestinationRoot E:\GitSpace\TensorRT-CSharp-API-4.0\downloads\yolov8n-seg-ultralytics-v8.3.0
```

`-Offline` 可在已有资产上只做长度、SHA256、许可证和来源复核。脚本不导出 ONNX、不运行 TensorRT，也不发布资产。

## ONNX 合同

本次使用 Ultralytics `8.4.21` 和 Torch `2.10` CPU 导出静态 FP32 ONNX：

```powershell
yolo export model=E:\GitSpace\TensorRT-CSharp-API-4.0\downloads\yolov8n-seg-ultralytics-v8.3.0\source\yolov8n-seg.pt format=onnx imgsz=640 opset=17 simplify=True dynamic=False
```

导出文件 SHA256：

```text
08b5c61368d4ddec5e647522fc55a93c42a9e0c581770aae48b87bba65a9b21d
```

实际 tensor 合同：

| Tensor | Shape | 角色 |
| --- | --- | --- |
| `images` | `[1,3,640,640]` | NCHW RGB float32 letterbox 输入 |
| `output0` | `[1,116,8400]` | 4 box + 80 class + 32 mask coefficients |
| `output1` | `[1,32,160,160]` | 32 个 mask prototypes |

这里显式使用 `output0:det,output1:mask-prototypes`。不能仅凭名称猜测 output role 后再把结果晋级为 proof。

## 预处理

YoloVision 从 `768x576` PPM 生成 `640x640` NCHW RGB float32 tensor：等比缩放到 `640x480`，上下各填充
80 像素，填充值 `114`，数值乘以 `1/255`。生成 tensor 共 `1228800` 个元素，SHA256 为：

```text
3a37e91ca77118ae168b367583faea65f61613ba71a38413f6adf794d88d0488
```

必须让 mask resize-back 使用同一份 preprocess metadata。外部 tensor 本身不携带 source-image 尺寸、pad 或 scale，
因此 `--mask-spatial-transform` 强制要求同时提供 `--image`。

## TensorRtExec 构建和加载

本次使用 TensorRT `10.11.0.33`、CUDA Toolkit `12.9`、RTX 3060 Laptop GPU 和 FP32。构建结果包含 3 个
bindings、255 layers、1 profile，保存 engine SHA256 为：

```text
5173601e56a872e74490e32eea0c82069b0d39759d1abb992300106c18931bfa
```

加载 engine 后用 `--referenceOutputs` 同时比较两个 output。`abs=0.02 / rel=0.02` 的 TensorRtExec
load-engine 运行中，两份 reference 均通过。reference 来源是独立 ONNX Runtime CPU Execution Provider：

| Tensor | Reference SHA256 | 比较值数 | Mismatch | Max abs |
| --- | --- | ---: | ---: | ---: |
| `output0` | `a607dd1d80dbb85a1a77f5354c7669e357c2caf8d0a9cd93328397d1858bde5f` | 974400 | 0 | 1.2499084 |
| `output1` | `3cc8387483187bc5d86ef8e82943e215caadefbbf41a8523d9ffc4c6b040f8d5` | 819200 | 0 | 0.010861397 |

不同 TensorRT tactic 可能改变逐值误差。最终 YoloVision 运行采用 `abs=0.02 / rel=0.03`，不是隐藏误差：报告仍保留
每个 tensor 的 mismatch、first mismatch、max abs/max rel、NaN 和 Infinity policy。

## YoloVision 运行

以下命令展示本次路径的关键参数；路径可替换为自己的 E 盘 case 目录：

```powershell
dotnet .\samples\YoloVision\bin\Release\net8.0\YoloVision.dll `
  --model E:\...\source\yolov8n-seg.onnx `
  --labels E:\...\yolox-apache\derived\coco.names `
  --image E:\...\yolox-apache\derived\dog.ppm `
  --preprocessed-output E:\...\runtime\dog-1x3x640x640-rgb-letterbox.fp32.bin `
  --input-shape 1x3x640x640 `
  --tensor-rt-line 10 `
  --family v8 --task seg `
  --output-role-map output0:det,output1:mask-prototypes `
  --mask-coefficient-count 32 `
  --confidence 0.25 --iou-threshold 0.45 --mask-threshold 0.5 `
  --mask-spatial-transform --mask-coordinate-space model-input --mask-crop-to-box true `
  --reference-outputs output0:E:\...\reference\output0.reference.json,output1:E:\...\reference\output1.reference.json `
  --reference-abs-tolerance 0.02 --reference-rel-tolerance 0.03 `
  --output-json E:\...\runtime\yolovision-yolov8n-seg-output.json `
  --segmentation-mask-output-directory E:\...\runtime\segmentation-masks `
  --visualization E:\...\runtime\yolovision-yolov8n-seg-output.svg
```

最终日志同时满足：

```text
BindingMetadata Index=1 Name=output0 Mode=Output DataType=Float Shape=[1, 116, 8400]
BindingMetadata Index=2 Name=output1 Mode=Output DataType=Float Shape=[1, 32, 160, 160]
ReferenceOutputValidation Requested=True Completed=True Passed=True Tensors=2
Postprocess Task=Segmentation;Detections=4;Classifications=0;Segmentations=4
OutputValidated=True
YoloVision Passed=True
```

四个实例为 dog、bicycle、truck 和 car。输出 JSON、SVG、run log 和 mask manifest 均在 sample-run evidence 中记录
SHA256，但文件本体不进入 Git。

## Mask 工件

`--segmentation-mask-output-directory` 为每个 prediction 写出三种数据：

1. `prototype-grid-probability`：`160x160` float32 little-endian。
2. `source-image-probability`：按同一 letterbox metadata 映射回 `576x768` 的 float32 little-endian。
3. `source-image-thresholded`：阈值化后的 `576x768` uint8，值只允许 `0/1`。

`segmentation-mask-artifacts.manifest.json` 保存 shape、element/byte count、SHA256、active pixel count、threshold、
class、score、source index、box 和 proof boundary。manifest 自身 SHA256 为：

```text
75237b8f28d5659dbd79b0fdc6259d267c1b5482bb1bab1201efe68ae61e58c5
```

二进制工件只为独立比较提供确定输入，单独存在时 `isRuntimeProof=false`。

## 独立后处理比较

仓库脚本使用 OpenCV 读取同一 PPM，并通过 Ultralytics/PyTorch CPU、`rect=False` 和 `retina_masks=True` 生成
source-image mask reference：

```powershell
C:\Users\guoji\.conda\envs\ultralytics\python.exe `
  .\eng\Invoke-YoloVisionSegmentationReference.py `
  --model E:\...\source\yolov8n-seg.pt `
  --image E:\...\yolox-apache\derived\dog.ppm `
  --output-directory E:\...\independent-postprocess `
  --actual-manifest E:\...\runtime\segmentation-masks\segmentation-mask-artifacts.manifest.json
```

比较门槛为 box coordinate error `<=1.0`、score error `<=0.01`、box IoU `>=0.995`、mask IoU `>=0.99`。
最终结果：

| 类别 | Box IoU | Mask IoU | 结果 |
| --- | ---: | ---: | --- |
| dog | 0.999666 | 0.995842 | passed |
| bicycle | 0.999345 | 0.995746 | passed |
| truck | 0.998472 | 0.996669 | passed |
| car | 0.998696 | 0.991141 | passed |

数值比较未达到门槛时脚本退出 `2`，不会把不完整比较写成 passed。manifest、文件长度、SHA256、`0/1` 值域或
active pixel count 损坏时同样以非零退出码 fail closed；这些完整性错误不会伪装成普通 IoU mismatch。

## 受控失败

为了证明 raw tensor reference 比较会 fail closed，本次只把 `output0` reference 的索引 `0` 增加 `10000`，随后加载
同一份已保存 engine。结果为：

```text
ReferenceOutputValidation Requested=True Completed=True Passed=False Tensors=2
ReferenceOutputTensor Tensor=output0 Passed=False Compared=974400 Mismatches=1 FirstMismatch=0 MaxAbs=9999.999
TensorRtExec State=load-engine-reference-validation-failed Success=False
exit code: 2
```

该负例验证的是数值 mismatch，不是 JSON 解析失败或缺依赖。

另一个完整性负例只修改一个 source thresholded mask 字节并保持 manifest SHA256 不变。独立脚本在计算 IoU 前即以
`Thresholded mask SHA256 does not match the manifest.` 拒绝该工件，证明比较链不会默默接受被篡改的 mask。

## 验证与边界

本地可执行：

```powershell
.\eng\Test-SampleRunEvidenceRecord.ps1 `
  -InputPath .\samples\assets\yolovision-yolov8n-seg-real-model-runtime-evidence.json `
  -RequireExistingLog -FailOnNotProof

dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug `
  --filter "FullyQualifiedName~YoloVision"
```

当前可声明：官方模型和 Apache-2.0 输入在 source tree 上完成了真实 TensorRT 多输出 enqueue、双 tensor reference
校验、mask compose、source-image transform 和独立 PyTorch 后处理比较。

当前仍不可声明：

- 模型或派生 ONNX 已获公开再分发批准；
- `.Bridge`/managed NuGet clean consumer 已重复该案例；
- 公开包 URL/hash 或 post-publish 验证已完成；
- Owner 已接受该案例作为正式 release proof。

下一步应在仓库外 clean bridge-only package consumer 中重复相同输入、tensor reference 与 mask IoU 门禁；公开发布仍只允许
managed、项目自有 `.Bridge` 和源码，CUDA、cuDNN、TensorRT、NVRTC 继续由用户自行安装。
