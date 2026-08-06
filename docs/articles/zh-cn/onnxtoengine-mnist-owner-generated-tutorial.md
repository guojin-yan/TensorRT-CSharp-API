# OnnxToEngine 实战：自有数字图片、TensorRT 与 ONNX Runtime 双重验证

本文从模型取得开始，完整演示如何用 TensorRtSharp4.0 的 `applications/OnnxToEngine` 运行 MNIST ONNX、生成项目自有的数字 7 输入、构建 TensorRT engine、执行推理、导出结构化结果，并把实际像素、预测数字、置信度和十类概率绘制到结果图。最后再用 ONNX Runtime CPU 对同一个 float32 tensor 建立独立参考，并用错误期望数字验证失败关闭。

模型、engine、PGM、tensor 和原始日志都留在 Git 仓库外。仓库只保存代码、脱敏证据与允许再分发的项目自有配图；CUDA、cuDNN、TensorRT 和 NVRTC 继续由用户按版本安装。

## 本文使用的项目与库

| 项目或库 | 本文职责 |
| --- | --- |
| `JYPPX.TensorRtSharp` | 解析 ONNX、构建 engine、绑定 tensor 并执行 enqueue |
| `JYPPX.CudaSharp` | 管理 CUDA 设备、stream 和显存生命周期 |
| `JYPPX.TensorRtSharp.Tools` | 实现 MNIST PGM 读取、预处理、结果验证和 SVG 可视化 |
| `applications/OnnxToEngine` | 提供 `--mnist` 命令入口与 JSON、tensor、engine、结果图导出 |
| `tests/fixtures/mnist-onnx-runtime-reference` | 在隔离项目中使用 ONNX Runtime CPUExecutionProvider 生成独立参考 |
| TensorRT `10.11` | 本次真实 GPU 推理后端 |
| ONNX Runtime `1.23.2` | 同一输入 tensor 的独立 CPU 对照 |

第一版只验证主要功能链路，不把单张数字结果解释为完整 MNIST 精度评测。

## 模型获取与许可证

本文模型来自用户安装的 TensorRT `10.11.0.33` 示例数据目录。该目录 README 将 `mnist.onnx` 归因于 [ONNX Model Zoo MNIST](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist)，固定来源标识为 `TensorRT-10.11.0.33-sample-data`。

| 项目 | 固定值 |
| --- | --- |
| ONNX 长度 | `26,454` bytes |
| ONNX opset | `8` |
| SHA256 | `2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf` |
| 暂存位置 | `models/OnnxToEngine/MNIST/nvidia-tensorrt-10.11/mnist.onnx` |
| Git 跟踪 | 否 |
| 公开再分发批准 | 否 |

从用户安装目录复制时只操作外层模型工作区：

```powershell
$RepoRoot = (Resolve-Path .).Path
$WorkspaceRoot = Split-Path -Parent $RepoRoot
$ModelRoot = Join-Path $WorkspaceRoot 'models/OnnxToEngine/MNIST/nvidia-tensorrt-10.11'
New-Item -ItemType Directory -Force -Path $ModelRoot | Out-Null
Copy-Item (Join-Path $env:TENSORRT_PATH 'data/mnist/mnist.onnx') $ModelRoot
Get-FileHash (Join-Path $ModelRoot 'mnist.onnx') -Algorithm SHA256
```

输入图不再使用 TensorRT 附带的 `7.pgm`。仓库脚本生成无第三方来源的固定几何数字 7，并将生成结果标记为 CC0-1.0：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File ./eng/New-MnistOwnerGeneratedDigit.ps1
```

生成的 `digit-7.pgm` 为 P5、28×28、797 bytes，SHA256 是 `e2f39f47bae623e76b4a16dc4f2de66f11e3cae9b5e0eb2fa4f17858d7a3aa87`。文件仍放在外层 `downloads/mnist-owner-generated`，不提交 Git；文章结果图由它的像素和真实预测生成。

## ONNX 转换与暂存

这个模型已经是上游 ONNX，不需要再次经过 PyTorch 或 TensorFlow 导出。机器清单采用下面的固定转换说明：

```text
Copy mnist.onnx from the user-installed TensorRT data/mnist directory; no local framework-to-ONNX conversion is performed.
```

这里要区分两件事：复制并校验 ONNX 不是模型转换；TensorRT 把 ONNX 构建为版本相关的 `.plan` 也不是重新导出 ONNX。若以后改用训练 checkpoint，必须另行固定训练框架、opset、输入名称、shape 和导出脚本。

模型合同为：

```text
Input3:float32[1,1,28,28] -> Plus214_Output_0:float32[1,10]
```

PGM 像素按 `tensor[index] = 1 - pixel[index] / 255` 转换。本次 784 个 float32 输入值的 SHA256 为 `7f6cfd9ba7fadb5e2751bd150ea92275f90c413681fec0c74921f4b476f81f76`。

## 创建本地包消费项目

MNIST 的低层持久化 plan 消费已经由 `tests/fixtures/package-consumers/RefittedPlan.PackageConsumer` 覆盖。它只引用主 API 和匹配的 bridge-only 两个本地候选包，隔离 restore graph 中 `ProjectReference=0`，CUDA、cuDNN 与 TensorRT 继续来自用户安装目录。对应完整文章为 `tensorrtexec-refitted-plan-local-package-consumer.md`。

本文新增的是 `OnnxToEngine` 源码树真实模型与可视化证明，不把两种证据混写。正式发布前可先本地打包：

```powershell
dotnet pack ./pack/JYPPX.TensorRT.CSharp.API/JYPPX.TensorRT.CSharp.API.csproj `
  -c Release -o ./artifacts/managed -p:JYPPXPackageVersion=4.0.0
```

这些命令只生成本地候选包，不会推送 NuGet.org、GitHub Packages，也不会创建 Tag 或 Release。

## 编写程序入口

`OnnxToEngine` 的 `--mnist` 分支先构造 `MnistOnnxRuntimeOptions`，再执行模型并写出可视化：

```csharp
MnistOnnxRuntimeResult result = new MnistOnnxRuntimeService().Execute(options);

if (!string.IsNullOrWhiteSpace(visualizationPath))
{
    MnistVisualizationWriter.Write(visualizationPath, result);
}
```

`MnistVisualizationWriter` 会重新读取本次运行的 PGM，并拒绝非 28×28 输入、未执行推理、无效预测索引或不是十分类的结果。SVG 左侧绘制实际 784 个像素，右侧绘制预测数字、置信度和十类概率，因此不是手工填写的指标卡片。

## 编译并运行

先构建样例：

```powershell
dotnet build ./applications/OnnxToEngine/OnnxToEngine.csproj -c Release
```

再设置用户安装的 TensorRT 根目录和项目自己编译的 bridge。下面只使用变量，不绑定任何机器盘符：

```powershell
$AssetRoot = Join-Path $WorkspaceRoot 'downloads/mnist-owner-generated'
$ModelPath = Join-Path $ModelRoot 'mnist.onnx'
$App = './applications/OnnxToEngine/bin/Release/net8.0/OnnxToEngine.dll'

$env:JYPPX_TENSORRT_ROOT = $env:TENSORRT_PATH
$env:JYPPX_NATIVE_BRIDGE_PATH = '<bridge-build>/jyppxtrtbridge.dll'
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = 'true'

dotnet $App --mnist --tensor-rt-line 10 `
  --onnx $ModelPath `
  --mnistInput (Join-Path $AssetRoot 'digit-7.pgm') `
  --expectedDigit 7 --minimumConfidence 0.5 `
  --saveEngine (Join-Path $AssetRoot 'digit-7.plan') `
  --exportReport (Join-Path $AssetRoot 'digit-7-report.json') `
  --exportOutput (Join-Path $AssetRoot 'digit-7-output.json') `
  --exportPreprocessedInput (Join-Path $AssetRoot 'digit-7-input.fp32.bin') `
  --visualization (Join-Path $AssetRoot 'digit-7-result.svg')
```

用同一个 float32 tensor 运行 ONNX Runtime CPU 对照：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File ./eng/Test-TensorRtExecMnistOnnxRuntimeReference.ps1 `
  -ModelPath $ModelPath `
  -InputPath (Join-Path $AssetRoot 'digit-7-input.fp32.bin') `
  -TensorRtReferencePath (Join-Path $AssetRoot 'digit-7-tensorrt.reference.json') `
  -OutputRoot (Join-Path $WorkspaceRoot 'consumer-workspaces/mnist-owner-generated-ort') `
  -ReportDirectory (Join-Path $AssetRoot 'onnxruntime') `
  -Strict
```

脚本已兼容 Windows PowerShell 5.1 和 PowerShell 7。若目标机器不能运行 GPU 部分，应检查 bridge、TensorRT、CUDA、驱动和版本组合；这属于本机环境前置条件，不能把跳过写成推理通过。

## 已验证结果

本次运行环境为 Windows 11、NVIDIA GeForce RTX 3060 Laptop GPU、TensorRT `10.11.0`、CUDA `12.9`。TensorRT 预测数字 `7`，置信度 `0.99945575`，enqueue 记录为 `0.971776 ms`。

ONNX Runtime `1.23.2` 明确使用 `CPUExecutionProvider`，比较 10 个 logits，mismatch `0`、first mismatch `-1`、最大绝对误差 `6.198883e-6`、最大相对误差 `1.3311652e-6`。

受控负例保持模型和输入不变，只把 `--expectedDigit` 从 `7` 改为 `6`；程序仍预测 `7`，`OutputMatch=False`，状态为 `mnist-output-mismatch`，退出码为 `2`。

![项目自有数字 7 的真实 TensorRT 分类结果](../../images/onnxtoengine-mnist-owner-generated-result.png)

![OnnxToEngine MNIST 真实运行与独立参考对照](../../images/onnxtoengine-mnist-owner-generated-terminal.png)

终端截图来自本次真实运行的 stdout，只移除了机器路径并合并了独立参考与受控负例的关键行。两张图都来自同一次真实 TensorRT 执行：结果图由该次运行写出的 SVG 渲染，终端图保留相同输入哈希、预测值和置信度。

## 复查与边界

机器可读证据位于 `samples/assets/onnxtoengine-mnist-owner-generated-runtime-evidence.json`，配图来源位于 `samples/assets/onnxtoengine-mnist-owner-generated-article-visual-assets.json`。复查时至少确认：

1. ONNX 与 PGM SHA256 和本文一致。
2. 输入输出名称、shape 和预处理公式与实际模型一致。
3. TensorRT 与 ONNX Runtime 都预测 7，且 10 个 logits 在容差内匹配。
4. 结果图包含实际像素、预测数字、置信度和十类概率。
5. 错误期望数字返回非零退出码。

本文证明的是固定模型、项目自有输入、真实 TensorRT 推理、独立 ONNX Runtime CPU 对照和结果绘制。它不证明完整 MNIST 精度，不授权再分发 ONNX，不是公共包、post-publish、Owner release acceptance、Tag 或 GitHub Release 证明。模型继续只保存在外层 `models` 目录，当前没有发布任何包或 Release。
