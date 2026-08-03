# YOLOv8n Detection 官方模型 TensorRT + C# 完整验证

本文记录 `samples/YoloVision` 对 Ultralytics 官方 `v8.3.0` `yolov8n.pt` 的真实执行结果。模型、ONNX、图片、tensor、reference、SVG 和日志全部保存在 E 盘；仓库只提交可重复获取脚本、来源清单、短证据记录和教程。

这不是仅能截图的演示。验证同时覆盖：

- 官方权重、COCO labels、测试图片与许可证的固定来源和 SHA256；
- PyTorch CPU 与 ONNX Runtime CPU 的原始输出对照；
- C# letterbox tensor 与 Ultralytics tensor 的量化差异；
- TensorRT 10.11 对全部 705,600 个原始输出值的 reference 校验；
- C# score、class-aware NMS、box decode 和 source-space 坐标还原；
- 受控错误 reference 的 fail-closed 负例。

## 已验证环境

| 项目 | 值 |
| --- | --- |
| OS | Windows 11 Home Chinese |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU 6 GB |
| Driver | 576.02 |
| CUDA | 12.9 |
| TensorRT | 10.11.0.33 |
| .NET SDK | 10.0.301 |
| Ultralytics | 8.4.21 |
| PyTorch | 2.10.0+cpu |
| ONNX / ONNX Runtime | 1.15.0 / 1.15.0 |

NVIDIA CUDA、cuDNN 和 TensorRT 由使用者自行安装。仓库与发布包不再打包这些 NVIDIA 运行库。

## 固定资产

机器可读清单位于：

```text
samples/assets/yolovision-yolov8n-det-official-assets.json
```

关键资产如下：

| 资产 | 来源 | SHA256 |
| --- | --- | --- |
| `yolov8n.pt` | Ultralytics assets `v8.3.0` release asset id `195719301` | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| `coco.yaml` | Ultralytics source commit `6e43d1...b0a` | `bd6f98a2e18775c39a4d5214080c87fcb163d367c18a2fcf2609371bab00c0b8` |
| `bus.jpg` | 同一 source commit | `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63` |
| `LICENSE` | 同一 source commit | `0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0` |

许可证记录为 `AGPL-3.0-only`。本案例只批准本地验证，没有批准在本项目 Release、NuGet 或 GitHub Packages 中重新分发模型、图片或派生 ONNX。

## 获取资产

```powershell
$python = 'python'
$root = '..\downloads\yolov8n-det-ultralytics-v8.3.0'

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Acquire-YoloV8DetectionOfficialAssets.ps1 `
  -AssetRoot $root `
  -PythonPath $python
```

脚本拒绝把大资产写入 C 盘，逐个校验长度和 SHA256，并派生：

- `bus.ppm`：`810x1080` P6 RGB，SHA256 `6cdb4b...e8688`；
- `coco.names`：80 行，顺序与模型内嵌 `model.names` 完全一致，SHA256 `bd17f1...e730a`。

获取脚本不执行 ONNX 导出、TensorRT、发布或上传。

## 导出与独立 reference

```powershell
& $python .\eng\Invoke-YoloVisionDetectionReference.py `
  --weights "$root\source\yolov8n.pt" `
  --coco-yaml "$root\source\coco.yaml" `
  --image "$root\source\bus.jpg" `
  --onnx "$root\source\yolov8n.onnx" `
  --output-directory "$root\reports\independent-reference" `
  --export-onnx
```

固定导出参数为：

```text
format=onnx imgsz=640 opset=17 simplify=True dynamic=False batch=1 device=cpu
```

实际 ONNX 合同：

```text
images  float32 [1,3,640,640]
output0 float32 [1,84,8400]
```

84 个通道必须解释为 `4 box + 80 class probabilities`。官方 YOLOv8 detection head 没有独立 objectness 通道，也没有 graph-side NMS。YoloVision 命令因此显式使用：

```text
--layout channels-first --has-objectness false --class-count 80
```

导出的 ONNX 长度为 `12,851,047` 字节，SHA256 为 `db28a49ffbb0425f39ae56252e7e0b43d06b357416c7da58872e285560b4221e`。PyTorch CPU 与 ONNX Runtime CPU 对同一 Ultralytics letterbox tensor 的最大绝对误差为 `0.0020446777`。

## C# 预处理

```powershell
dotnet .\samples\YoloVision\bin\Release\net8.0\YoloVision.dll `
  --preprocess-only `
  --image "$root\derived\bus.ppm" `
  --preprocessed-output "$root\runtime\bus-csharp-letterbox-1x3x640x640.fp32.bin" `
  --input-shape 1x3x640x640 `
  --tensor-layout NCHW `
  --color-order RGB `
  --resize letterbox `
  --family v8 `
  --task det
```

实际几何信息：

```text
SourceSize=810x1080
Target=640x640
Resized=480x640
Pad=80,0
Scale=0.592593,0.592593
Normalize=True
ValueScale=1/255
Fill=114
```

C# tensor SHA256 为 `46a0278967f1230ef8db59b0b8311a3aba3dce45233f1ba68821418ae02a574d`。与 Ultralytics tensor 相比，最大绝对误差为 `1/255`，平均绝对误差为 `0.0003352034`。差异来自两套双线性插值的整数取整，不应通过修改运行时 reference 来隐藏。

reference 工具因此保留两层对照：

1. 官方 `model.predict(image)` 结果用于观察完整官方图像管线。
2. PyTorch CPU 对 C# 实际 tensor 执行官方 NMS，用于严格验证 TensorRT 与 C# decode。

## TensorRT 正例

```powershell
dotnet .\samples\YoloVision\bin\Release\net8.0\YoloVision.dll `
  --model "$root\source\yolov8n.onnx" `
  --labels "$root\derived\coco.names" `
  --image "$root\derived\bus.ppm" `
  --preprocessed-output "$root\runtime\bus-csharp-letterbox-1x3x640x640.fp32.bin" `
  --input-shape 1x3x640x640 `
  --tensor-rt-line 10 `
  --family v8 `
  --task det `
  --layout channels-first `
  --has-objectness false `
  --class-count 80 `
  --confidence 0.25 `
  --iou-threshold 0.45 `
  --top-k 10 `
  --nms-mode class-aware `
  --reference-outputs "output0:$root\reports\independent-reference\output0.reference.json" `
  --reference-abs-tolerance 0.02 `
  --reference-rel-tolerance 0.05 `
  --output-json "$root\runtime\yolovision-yolov8n-det-output.json" `
  --visualization "$root\runtime\yolovision-yolov8n-det-output.svg"
```

原始张量比较覆盖全部 `84 * 8400 = 705,600` 个值，正例结果为：

```text
ReferenceOutputValidation Requested=True Completed=True Passed=True
Mismatches=0
Postprocess Task=Detection;Detections=5
OutputValidated=True
YoloVision Passed=True
```

`abs=0.02 / rel=0.05` 是坐标与分数混合 tensor 的组合容差：框坐标使用相对容差，零附近的 class score 使用绝对容差。TensorRT 每次构建可为未进入最终结果的低置信候选选择不同 tactic，已观察到单个坐标约 `3.3%` 的漂移，因此 5% 是按重复执行结果确定的上界。最终有效框另有独立逐框校验，不能只依赖 raw tensor 的统一容差。

## 最终检测结果

5 个最终框为 4 个 `person` 和 1 个 `bus`，TensorRT score 约为：

| 排名 | 类别 | Score |
| --- | --- | --- |
| 1 | person | 0.8906 |
| 2 | person | 0.8829 |
| 3 | person | 0.8787 |
| 4 | bus | 0.8416 |
| 5 | person | 0.4380 |

对 C# 实际 tensor 的 PyTorch 官方 NMS reference，5 个框全部匹配：

- 最小 source-space box IoU：`0.999841`；
- 最大 score 绝对误差：`0.000103`；
- class id、class name、预测数和排序一致；
- 逐框门槛：IoU 不低于 `0.995`，score 误差不高于 `0.01`。

官方完整图像管线与 C# tensor 对大面积 bus 框的 IoU 约为 `0.973`，其余 4 个框均为 `0.9992+`。这是预处理插值差异的可观测结果，不是 NMS 或 box decoder 错误。

## 受控负例

reference 工具把 `output0` 第 0 个坐标值增加 `125`：

```text
ReferenceOutputValidation ... Passed=False
FirstMismatch=0
MaxAbs=124.998024
Diagnostic=... first actual=3.3027892, expected=128.30081
OutputValidated=False
YoloVision Passed=False
ExitCode=1
```

TensorRT 每次构建可能让少量低置信候选出现正常 tactic 数值漂移，因此负例的 mismatch 总数不是固定 API；固定合同是 `exitCode != 0`、`Passed=False`、`FirstMismatch=0` 且受控差值被捕获。

## C# 接口收口

纯 detection head 现在要求：

```text
channelCount == box/objectness channel count + configured classCount
```

配置的 class count 不再允许只消费部分 score 通道。seg、pose、OBB 的任务专属调用会显式允许 class scores 后的 mask/keypoint/angle 辅助通道，避免破坏合法的多任务输出。

`yolovision-output.v1` 的 `postprocess` 还会记录：

- `classCount`；
- `hasObjectness`；
- `layout`；
- `coordinateSpace=model-input-pixels`；
- confidence、IoU、Top-K 与 NMS mode。

## 证据与边界

短证据记录位于：

```text
samples/assets/yolovision-yolov8n-det-real-model-runtime-evidence.json
```

它只证明当前源码树、当前主机、当前模型与当前 C# bridge 的 `real-model-runtime`。以下状态仍为 false：

- 模型或图片公开再分发批准；
- clean package-consumer-runtime；
- public-package 与 post-publish proof；
- Owner release acceptance；
- tag、GitHub Release、NuGet 或 GitHub Packages 发布证明。

发布第一版 C# 包和 bridge-only 包时，用户仍需自行安装匹配版本的 NVIDIA CUDA、cuDNN、TensorRT 和 NVRTC。
