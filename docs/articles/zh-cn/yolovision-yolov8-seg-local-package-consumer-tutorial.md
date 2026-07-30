# YoloVision YOLOv8n-seg 本地 Bridge-only 包消费者实战

本文验证一个仓库外、只使用 `PackageReference` 的 YOLOv8n-seg 消费者。消费者只还原三个项目自有包：

- `JYPPX.TensorRT.CSharp.API`
- `JYPPX.TensorRT.CSharp.API.YoloVision`
- `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge`

CUDA、cuDNN 和 TensorRT 由宿主机安装，绝不进入 `.nupkg`。模型、权重、raw reference、mask 二进制和运行日志继续留在 E 盘，不进入 Git。

本次结果属于 `local-package-consumer-runtime`。它不是公开 feed 下载证明，不是 post-publish 证明，不是模型再分发授权，也不是 Owner release acceptance。

## 验证边界

clean consumer 必须同时满足：

1. 临时项目位于仓库外的 E 盘工作区。
2. 项目中只有三个 `PackageReference`，没有 `ProjectReference`、`Reference` 或 `HintPath`。
3. `NuGet.config` 先执行 `<clear />`，随后只加入三个本地文件源。
4. 独立 NuGet cache 位于 E 盘，还原图中 `project` 类型依赖数为 0。
5. 输出目录中只有一个由 Bridge 包复制的 `jyppxtrtbridge.dll`。
6. 子进程删除继承的 `JYPPX_NATIVE_BRIDGE_PATH`，不以环境变量旁路 NuGet 资产选择。
7. TensorRT、CUDA 和 cuDNN 从明确的宿主机目录加载。
8. 正例必须比较两个 raw tensor，生成 source-image mask，并通过独立 PyTorch mask IoU。
9. 单值 reference 篡改和单字节 mask 篡改都必须非零退出。

## 固定资产

官方资产清单位于：

```text
samples/assets/yolovision-yolov8n-seg-official-assets.json
```

本地默认路径为：

```text
E:\GitSpace\TensorRT-CSharp-API-4.0\downloads\yolov8n-seg-ultralytics-v8.3.0\source\yolov8n-seg.onnx
E:\GitSpace\TensorRT-CSharp-API-4.0\downloads\yolov8n-seg-ultralytics-v8.3.0\source\yolov8n-seg.pt
E:\GitSpace\TensorRT-CSharp-API-4.0\downloads\yolov8n-seg-ultralytics-v8.3.0\reference\output0.reference.json
E:\GitSpace\TensorRT-CSharp-API-4.0\downloads\yolov8n-seg-ultralytics-v8.3.0\reference\output1.reference.json
E:\GitSpace\TensorRT-CSharp-API-4.0\downloads\yolox-apache\derived\coco.names
E:\GitSpace\TensorRT-CSharp-API-4.0\downloads\yolox-apache\derived\dog.ppm
```

runner 在 restore 前验证这些资产的固定 SHA256。`yolov8n-seg.pt` 来自 Ultralytics `v8.3.0`，许可证为 `AGPL-3.0-only`；当前只允许本地验证，公开再分发仍需 Owner 单独批准。

## 构建三个本地包

在仓库根目录执行：

```powershell
dotnet pack .\pack\JYPPX.TensorRT.CSharp.API\JYPPX.TensorRT.CSharp.API.csproj `
  -c Release `
  -o .\artifacts\managed `
  -p:JYPPXPackageVersion=4.0.0 `
  -p:UseSharedCompilation=false

dotnet pack .\samples\YoloVision\YoloVision.csproj `
  -c Release `
  -o .\artifacts\yolovision-nupkg `
  -p:JYPPXPackageVersion=4.0.0 `
  -p:UseSharedCompilation=false

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 `
  -SourceRuntimeKey win-x64-trt10.11-cuda12.9-cudnn9.22 `
  -Version 4.0.0 `
  -SplitPackageRole bridge `
  -SkipManagedPack `
  -SkipConsumerValidation
```

第三条命令只允许 `bridge` role。`cuda-cudnn`、`tensorrt`、`full-runtime`、meta 和 collection 均已退役，不能恢复为发布输入。

## 执行 clean consumer

Windows PowerShell 5.1 和 PowerShell 7 均可运行同一入口：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Test-YoloVisionSegmentationLocalPackageConsumer.ps1 `
  -RepositoryRoot $PWD `
  -PackageVersion 4.0.0
```

默认 runtime key 为 `win-x64-trt10.11-cuda12.9-cudnn9.22`。如本机目录不能由 `eng/Resolve-RuntimeRoots.ps1` 自动解析，可显式传入：

```powershell
-TensorRtRoot <TensorRT-root> `
-TensorRtRuntimeRoot <TensorRT-runtime-root> `
-CudaRoot <CUDA-root> `
-CudnnRoot <cuDNN-root> `
-PythonPath <ultralytics-cpu-python.exe>
```

runner 会复制 `samples/YoloVision.PackageConsumer` 到：

```text
E:\GitSpace\TensorRT-CSharp-API-4.0\consumer-workspaces\yolovision-yolov8n-seg-local-package-trt10
```

通过后整个临时工作区会被删除。NuGet 长包名可能超过 Windows PowerShell 的旧 `MAX_PATH` 边界，因此清理函数在 `Remove-Item` 失败时使用经过根目录检查的 `\\?\` 扩展路径删除；它不会调用全局 `dotnet build-server shutdown`，避免干扰其他工作区。

## 模型合同与正例

运行命令固定以下合同：

```text
images  : [1,3,640,640]     float32
output0 : [1,116,8400]      detection rows + 32 mask coefficients
output1 : [1,32,160,160]    mask prototypes
```

关键参数为：

```text
--family v8
--task seg
--output-role-map output0:det,output1:mask-prototypes
--mask-coefficient-count 32
--mask-threshold 0.5
--mask-spatial-transform
--mask-coordinate-space model-input
--mask-crop-to-box true
--reference-abs-tolerance 0.02
--reference-rel-tolerance 0.03
```

2026-07-31 的本地执行结果：

| 项目 | 结果 |
|---|---:|
| 本地包数 | 3 |
| ProjectReference | 0 |
| 直接 DLL 引用 | 0 |
| restore graph 中的 project library | 0 |
| raw tensor 数 | 2 |
| raw 比较值 | 1,793,600 |
| raw mismatch | 0 |
| 实例 mask | 4 |

四个实例的独立 Ultralytics/PyTorch CPU 比较如下：

| 类别 | box IoU | mask IoU |
|---|---:|---:|
| dog | 0.999667 | 0.995896 |
| bicycle | 0.999364 | 0.995729 |
| truck | 0.998436 | 0.996817 |
| car | 0.998673 | 0.991141 |

门槛为 box coordinate absolute error `<= 1.0`、score error `<= 0.01`、box IoU `>= 0.995`、mask IoU `>= 0.99`。独立脚本记录 Python、Ultralytics 和 PyTorch 版本，并把本次比较分类为 `local-package-consumer-runtime`。

## 两个受控负例

`eng/New-YoloVisionReferenceMutation.py` 通过 JSON parser 读取 `output0.reference.json`，只给索引 0 加 `10000`，并把来源标记为 `controlled-single-value-mutation`。同一个 package consumer 再运行一次后必须得到：

```text
exitCode=1
tensorName=output0
mismatchCount=1
firstMismatchIndex=0
YoloVision Passed=False
```

mask 完整性负例复制正例 mask 目录，只翻转第一份 `sourceThresholded` 文件中的一个字节，同时保持 manifest 中的 SHA256 不变。独立比较器必须在计算 IoU 前以以下诊断退出：

```text
Thresholded mask SHA256 does not match the manifest.
```

这两个负例分别证明 raw reference 门和 mask artifact 完整性门不是“只记录、不拦截”。

## 导出轻量证据

完整日志、mask 和独立参考位于被 Git 忽略的 `artifacts/yolovision/yolov8n-seg-local-package-consumer`。用以下命令验证并导出轻量记录：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Export-YoloVisionSegmentationLocalPackageConsumerEvidence.ps1 `
  -RepositoryRoot $PWD
```

可提交记录为：

```text
samples/assets/yolovision-yolov8n-seg-local-package-consumer-runtime-evidence.json
```

导出器会拒绝 package 数、引用数、shape、比较值、IoU、负例或 proof boundary 的任何漂移。轻量记录只保存包和证据 SHA256，不提交 `.nupkg`、模型、engine、raw tensor、mask 或日志。

## 尚未证明的事项

本次验证没有访问 nuget.org 或 GitHub Packages，也没有执行任何 publish/upload。因此以下值必须保持 false：

```text
publicPackageProof
packagesDownloadedFromPublicFeed
postPublishProof
publicRedistributionOwnerApproval
ownerReleaseAcceptance
releaseProof
performsPublish
uploadsAssets
```

只有 Owner 允许公开发布后，才能用公开 feed 的全新机器或全新容器重复下载、restore、运行和 hash 核对，再建立 post-publish 与 release 证据。
