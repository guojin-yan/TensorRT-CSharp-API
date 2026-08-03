# YoloVision 官方 YOLOX-S 下载、构建与真实图片运行教程

本文使用 YOLOX 官方仓库的 Apache-2.0 资产，把 `samples/YoloVision` 从 ONNX 下载推进到 TensorRT 10.11 真实图片检测。流程覆盖来源固定、SHA256、JPG 到 PPM 派生、官方预处理、engine 构建、C# enqueue、raw output 解码、NMS、JSON/SVG 和严格证据校验。

本次已验证结果：

- ONNX：官方 `yolox_s.onnx`，opset 11，SHA256 `c5c2d13e...998063`。
- 输入：`images [1,3,640,640]`。
- 输出：`output [1,8400,85]`。
- TensorRT：10.11.0.33，FP32。
- GPU：NVIDIA GeForce RTX 3060 Laptop GPU。
- 检测：5 个，其中 `bicycle=0.954841`、`dog=0.913382`。
- 真实运行日志最后应包含 marker：`YoloVision Passed=True`。
- 严格 sample-run validator：`real-model-runtime`，owner-action 0。

这里的 `real-model-runtime` 只表示源码树中的真实模型运行。它不是 `package-consumer-runtime`，不代表资产已获准随仓库或 NuGet 公开发布。

## 1. 环境要求

从仓库根目录执行：

```powershell
cd .
dotnet --version
cmake --version
nvidia-smi
```

需要：

- .NET 8 SDK。
- Visual Studio 2022 C++ 工具链和 CMake。
- TensorRT 10.x，通过 `JYPPX_TENSORRT_ROOT` 指向用户安装目录。
- CUDA 12.x，本次 bridge 使用 `win-x64-trt10-cuda12-release` preset。
- NVIDIA driver 能运行 TensorRT 10.11。

模型、图片、engine 和 tensor 较大，获取脚本会拒绝 C 盘输出。本仓库默认把它们放到外层 E 盘目录：

```text
..\downloads\yolox-apache
```

## 2. 资产来源与固定版本

清单位于：

```text
samples/assets/yolovision-yolox-official-assets.json
```

除 release 模型外，raw 文件都固定到 YOLOX `0.1.1rc0` 对应提交：

```text
e1052df71842031413f6030723c3607b839c80ce
```

核心文件：

| 资产 | 长度 | SHA256 |
| --- | ---: | --- |
| `yolox_s.onnx` | 35,858,002 | `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063` |
| `LICENSE` | 11,352 | `577c03d505ec80f667ebf96ebd0cc4f6825c817ca1088ee348c48aaabd51bd92` |
| `dog.jpg` | 163,759 | `5a9522051c3cec2bbd2f6323fccba32e8fbf3ddcc2b3e2fd46b04c720bc6f866` |
| `coco_classes.py` | 1,296 | `b38193c481a73f1f674cedab9e551b15b39b1a7aaed3e09e16505362cc54ad51` |

清单还固定了官方 `onnx_inference.py` 和 `data_augment.py`，用于审计预处理与后处理语义。

## 3. 下载并派生可运行资产

在线获取：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Acquire-YoloXOfficialAssets.ps1
```

已有缓存时进行离线复核：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Acquire-YoloXOfficialAssets.ps1 `
  -Offline
```

脚本会完成四件事：

1. 拒绝任何位于 `C:\` 的 `OutputRoot`。
2. 下载或复用 E 盘文件，并核对长度和 SHA256。
3. 从官方 `coco_classes.py` 生成按原顺序排列的 80 类 `coco.names`。
4. 将官方 `dog.jpg` 转成内置图片解码器支持的 P6 RGB `dog.ppm`。

派生文件固定结果：

| 文件 | SHA256 |
| --- | --- |
| `derived/dog.ppm` | `6cb94c9cd0781412598fe179246b09041af4303d388a5ba3c55f760dff11ec2c` |
| `derived/coco.names` | `4d4aaea7bee6be2f675d9b53a9195ca36dfe6429f7479f29155da522a6c85930` |

获取报告位于：

```text
artifacts/yolovision/yolox-official-runtime/acquisition-report.json
```

## 4. YOLOX 预处理为什么不同

官方 YOLOX-S 需要：

- `NCHW`。
- `BGR`。
- float32 值保持 `0..255`，不除以 255。
- 保持宽高比，填充值 114。
- 图片放在左上角，剩余区域在右侧和底部填充。

因此 `--family yolox` 默认使用：

```text
Layout=NCHW Color=BGR Normalize=False ValueScale=1 Alignment=top-left Fill=114
```

可先只生成 tensor：

```powershell
dotnet run --project .\samples\YoloVision -- `
  --family yolox `
  --task det `
  --image ..\downloads\yolox-apache\derived\dog.ppm `
  --preprocessed-output ..\downloads\yolox-apache\derived\dog-yolox-s-1x3x640x640-bgr-top-left.fp32.bin `
  --input-shape 1x3x640x640 `
  --preprocess-only
```

本次 tensor SHA256：

```text
ca4e22bc6d8ebfe70f5aefeae8957d9ad15eb8d3bf99b6a42e016436dcbf1528
```

`--preprocess-only` 只是预处理证据，不是推理证明。

## 5. YOLOX raw output 解码

官方 ONNX 输出 `[1,8400,85]`，不是已经还原到像素坐标的最终 box。8400 来自三个 feature map：

```text
80*80 + 40*40 + 20*20 = 8400
```

`YoloXOutputDecoder` 按 stride `8,16,32` 和 row-major grid 执行：

```text
centerX = (rawX + gridX) * stride
centerY = (rawY + gridY) * stride
width   = exp(rawWidth) * stride
height  = exp(rawHeight) * stride
score   = objectness * bestClassScore
```

转换后再进入 class-aware 或 class-agnostic NMS。内置 YOLOX profile 明确只支持 detection；`cls/seg/obb/pose/sem` 会报告 unsupported family/task，而不是虚假落入通用 decoder。

## 6. 构建 native bridge

已有 bridge 时可跳过。否则设置本机 TensorRT 路径后构建：

```powershell
$env:TENSORRT_PATH = $env:JYPPX_TENSORRT_ROOT
cmake --preset win-x64-trt10-cuda12-release
cmake --build --preset win-x64-trt10-cuda12-release
```

输出：

```text
build-out/win-x64-trt10-cuda12-release/bin/Release/jyppxtrtbridge.dll
```

## 7. 用 trtexec 构建 engine

engine 继续留在 E 盘 downloads：

```powershell
$trtexec = Join-Path $env:JYPPX_TENSORRT_ROOT 'bin\trtexec.exe'
$onnx = '..\downloads\yolox-apache\source\yolox_s.onnx'
$engine = '..\downloads\yolox-apache\derived\yolox_s-trt10.11-fp32.engine'

& $trtexec `
  "--onnx=$onnx" `
  "--saveEngine=$engine" `
  --skipInference `
  --memPoolSize=workspace:1024 `
  --profilingVerbosity=detailed
```

本次结果：

```text
EngineLength=48241100
EngineSha256=9b31390a786e8f520d4f3f78fbb0444eb7563c4bc8c2dddd5d7861c5c69524b1
Bindings: images, output
```

engine 与 GPU、TensorRT 版本、builder 配置相关，不应把上述 engine hash 当作跨机器固定值。

## 8. 运行 YoloVision

先让 Windows loader 找到 bridge、TensorRT 和 CUDA DLL：

```powershell
$bridge = (Resolve-Path '.\build-out\win-x64-trt10-cuda12-release\bin\Release').Path
$trt = $env:JYPPX_TENSORRT_ROOT
$cuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9'
$env:PATH = "$bridge;$trt\bin;$trt\lib;$cuda\bin;$env:PATH"
```

执行真实图片检测：

```powershell
dotnet run --project .\samples\YoloVision -- `
  --model ..\downloads\yolox-apache\source\yolox_s.onnx `
  --labels ..\downloads\yolox-apache\derived\coco.names `
  --image ..\downloads\yolox-apache\derived\dog.ppm `
  --preprocessed-output ..\downloads\yolox-apache\derived\dog-yolox-s-1x3x640x640-bgr-top-left.fp32.bin `
  --output-json .\artifacts\yolovision\yolox-official-runtime\yolovision-output.json `
  --visualization .\artifacts\yolovision\yolox-official-runtime\yolovision-output.svg `
  --input-shape 1x3x640x640 `
  --tensor-rt-line 10 `
  --family yolox `
  --task det `
  --layout boxes-first `
  --has-objectness true `
  --nms-mode class-aware `
  --confidence 0.3 `
  --iou-threshold 0.45 `
  --top-k 20
```

关键输出：

```text
Input=images:[1, 3, 640, 640] Output=output:[1, 8400, 85]
Detection Class=bicycle Score=0.954841 ...
Detection Class=dog Score=0.913382 ...
Real run log final marker: YoloVision Passed=True
```

JSON 和 SVG 分别用于机器审计与人工检查，不能只保留截图而丢掉日志和 hash。

## 9. 严格验证

验证输出 report：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Test-YoloVisionOutputReport.ps1 `
  -InputPath .\artifacts\yolovision\yolox-official-runtime\yolovision-output.json `
  -OutputPath .\artifacts\yolovision\yolox-official-runtime\validation\yolovision-output-report-validation.json `
  -Strict
```

验证 sample-run record：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Test-SampleRunEvidenceRecord.ps1 `
  -InputPath .\artifacts\yolovision\yolox-official-runtime\sample-run-evidence-record.yolox-official.json `
  -OutputRoot .\artifacts\yolovision\yolox-official-runtime\validation `
  -RequireExistingLog `
  -FailOnNotProof
```

预期：

```text
ValidationState=real-model-runtime
CanPromoteRealModelRuntime=True
ErrorCount=0
OwnerActionRequiredCount=0
```

## 10. 证据边界与清理

这条链已证明：

- 官方 ONNX 可由 TRT10 parser/builder 构建。
- 官方图片可由内置 profile 生成正确 tensor。
- C# bridge 完成真实 enqueue 与 output readback。
- raw YOLOX 坐标、objectness、class score 和 NMS 可生成合理检测。
- 运行日志、JSON、SVG、模型、图片、labels 和 tensor hash 相互可追溯。

它没有证明：

- 公开 NuGet/GitHub package 的干净消费者运行。
- 资产可以随仓库、NuGet 或 GitHub Release 公开再分发。
- owner 已签署公开发布批准。

本地清理只删除外层 E 盘下载目录，不要删除 CUDA、TensorRT 或 Codex 自身依赖：

```powershell
Remove-Item -LiteralPath '..\downloads\yolox-apache' -Recurse -Force
```

再次运行 acquisition 脚本即可恢复全部外部资产。

## 小结

YOLOX 的关键不是多加一个 family 名字，而是把不同于 YOLOv8 的 raw grid/stride 输出和左上 letterbox 语义真正实现并验证。完成后，YoloVision 才能在同一套 C# API 中得到可复现、可诊断、可审计的官方 YOLOX-S 检测结果。
