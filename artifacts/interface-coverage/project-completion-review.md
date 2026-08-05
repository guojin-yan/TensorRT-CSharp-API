# TensorRtSharp4.0 当前完成情况审查

> 本文件是当前状态索引，不再充当按日期追加的开发日志。历史实现过程由 Git 提交记录保存；覆盖数字、运行证据和发布判断以本文引用的结构化产物为准。

## 结论

TensorRtSharp4.0 已具备首版候选所需的主要 C# API、TensorRT/CUDA bridge、命令行工具、示例项目、本地包消费验证和真实模型案例。当前仍处于开发收尾阶段，禁止创建 tag、GitHub Release、NuGet/GitHub Packages 发布或上传模型。

允许形成候选的产物只有：

- `JYPPX.TensorRT.CSharp.API` 托管接口包；
- `JYPPX.TensorRT.CSharp.API.YoloVision` 纯 C# 扩展包；
- 按 TensorRT/CUDA 版本编译、只含项目自有 `jyppxtrtbridge` 的 Bridge 包；
- 只包含 Git 跟踪文件的源码归档。

CUDA、cuDNN、TensorRT 和 NVRTC 由用户自行安装，不进入项目包或 Release。转换后的 ONNX 暂存在仓库外层 `models` 目录，等待独立 Model Zoo 接管，不上传当前 Git 仓库。

## 接口覆盖

当前 Git 跟踪 `214` 个 manifest 文件，共 `4046` 条 API 记录。其中 `common=11`、`cuda=675`、`tensorrt=3360`。

最近一次具备完整 TensorRT 8/10/11 与 CUDA toolkit 输入的 vendor-header 覆盖扫描基线为 `4013` 条，生成于 2026-08-03。此后新增的 `33` 条 manifest API 已进入源码和绑定生成验证，但尚未回填到该次逐版本 header-scan 行。因此下表是完整 SDK 矩阵扫描基线，不冒充当前 4046 条的重新扫描结果；正式候选冻结前要在完整 SDK 主机上重建覆盖摘要。

完整 SDK 扫描基线为：

| TensorRT 版本线 | 扫描 / 匹配 / 源码存在 | 已实现 | 仅 deferred |
| --- | ---: | ---: | ---: |
| 8.6 | `880 / 880 / 880` | `760` | `120` |
| 10.11 | `879 / 879 / 879` | `762` | `117` |
| 11.0 | `901 / 901 / 901` | `815` | `86` |

`deferred-only` 表示公开接口仍明确阻止不安全或未验证的路径，不等于遗漏。历史 deferred 记录不会为了提高覆盖率而删除；已安全实现的别名会归类为 `implemented-with-deferred-history`。

权威产物：

- `artifacts/interface-coverage/interface-coverage-summary.md`
- `artifacts/interface-coverage/tensorrt-interface-coverage.json`
- `artifacts/interface-coverage/tensorrt-interface-comparison.csv`
- `artifacts/interface-coverage/cuda-runtime-interface-coverage.json`
- `artifacts/interface-coverage/public-api-documentation-closure.json`

## 主要功能

已纳入源码、合同测试或真实运行证据的核心范围包括：

- TensorRT builder、network、runtime、engine、execution context、ONNX parser、plugin、refit、profiling 和 diagnostics；
- CUDA device、stream、event、graph、memory、IPC、kernel、texture/surface 与运行时诊断；
- owner-safe callback、allocator、debug listener 和执行上下文资源生命周期；
- `TensorRtExec`、`OnnxToEngine`、MNIST、Classification、YoloVision 和 smoke runners；
- TensorRT 8/10/11 与 CUDA 11/12/13 的 bridge 构建矩阵及严格 native entry allowlist。

“已经有接口”与“已经形成真实运行证明”必须分开理解。各 API 的状态以 coverage、runtime evidence 和 package-consumer evidence 三类文件共同判断。

## 真实模型案例

首批图像案例覆盖以下任务：

| 任务 | 模型 | ONNX 暂存位置 | 当前案例状态 |
| --- | --- | --- | --- |
| 检测 | YOLOv8n | `models/YoloVision/Detection/yolov8n-ultralytics-v8.3.0` | 真实 TensorRT、raw reference、标注图、终端截图 |
| End-to-End 检测 | YOLOv10n | `models/YoloVision/Detection/yolov10n-thu-mig-v1.1` | 真实 TensorRT、六列输出合同、标注图、终端截图 |
| 检测 | YOLOX-S | `models/YoloVision/Detection/yolox-s-megvii-v0.1.1rc0` | 真实 TensorRT、grid/stride 解码、标注图、终端截图 |
| 分类 | YOLOv8n-cls | `models/YoloVision/Classification/yolov8n-cls-ultralytics-v8.3.0` | 真实 TensorRT、Top-5、标注图、终端截图 |
| 实例分割 | YOLOv8n-seg | `models/YoloVision/InstanceSegmentation/yolov8n-seg-ultralytics-v8.3.0` | 真实 TensorRT、双输出、mask、标注图、终端截图 |
| 姿态 | YOLOv8n-pose | `models/YoloVision/Pose/yolov8n-pose-ultralytics-v8.3.0` | 真实 TensorRT、关键点、标注图、终端截图 |
| 旋转框 | YOLOv8n-obb | `models/YoloVision/OrientedBoundingBox/yolov8n-obb-ultralytics-v8.3.0` | 真实 TensorRT、rotated IoU、标注图、终端截图 |
| 语义分割 | LRASPP MobileNetV3 Large | `models/YoloVision/SemanticSegmentation/lraspp-mobilenet-v3-large-torchvision-v0.25.0` | 真实 TensorRT、逐像素 argmax、覆盖图、终端截图 |

OBB 与实例分割文章分别使用项目所有者提供并授权用于文章的 `plane.png` 和 `dog.jpg`。文章通过 `JYPPX_DEMO_IMAGE_ROOT` 获取本机素材，复制后严格校验 SHA256，不写作者机器的绝对路径。

模型获取地址、固定版本、许可证、权重哈希、ONNX 转换命令、外层暂存位置和运行步骤集中在：

- `docs/articles/zh-cn/demo-model-acquisition-and-onnx-conversion.md`
- `docs/articles/zh-cn/publication-catalog.json`
- `samples/assets/README.md`
- `samples/assets/yolovision-model-acquisition-and-conversion-manifest.json`

## 技术文章

`docs/articles/zh-cn` 同时包含可发布技术文章、项目文档、运行手册、审计记录和历史过程材料，不能把目录内每个 Markdown 都视为成稿。当前目录分类和成稿入口由以下文件维护：

- `docs/articles/zh-cn/README.md`
- `docs/articles/zh-cn/publication-catalog.json`

完整图像案例文章必须同时具备：项目与依赖介绍、模型获取和许可证、ONNX 转换、可执行的完整流程、真实结构化结果、原图叠加结果和真实终端或软件窗口截图。截图用于阅读，不能替代日志、JSON、SHA256 或受控负例。

## 工程脚本

`eng` 中的脚本分为生成、构建、验证、资产获取、真实运行和发布门禁。多数脚本由 Actions 或合同测试调用，不是面向最终用户的独立命令。脚本入口、调用边界和退役规则记录在 `eng/README.md`。

清理规则：

- 删除零引用、零测试、零工作流入口且已被替代的脚本；
- 相同任务只保留一个权威入口，薄包装器必须说明其转发目标；
- 生成文件由 exporter 重建，不手工复制同一事实；
- 内部审计、模板和 build-only 结果不得包装成真实运行证明；
- 发布脚本必须默认无副作用，并受 Owner 明确批准门禁保护。

## 当前门禁

本地和 Actions 可以继续执行构建、测试、pack dry-run、内容检查和本地 feed consumer。以下事项在项目完成前保持关闭：

- NuGet push；
- GitHub Packages push；
- Git tag 和 GitHub Release；
- Release asset 上传；
- 文档正式部署；
- 模型、engine、NVIDIA 运行库和本地证据资产上传。

既有 NuGet.org 包按 Owner 要求保留，不执行删除。历史 GitHub Release 和 GitHub Packages 的远程清理由只读清单与显式确认控制，不能夹带在普通构建或收尾脚本中。

## 尚未完成

首版正式发布前仍需完成：

1. 收敛 `eng`、文档和生成产物中的重复或失效内容，并保持引用与测试通过。
2. 完成目标兼容主机上的必要运行复验，特别是 Linux 和仍缺宿主环境的 runtime key。
3. 确认项目许可证及各候选包、源码归档的许可证元数据。
4. 完成最终源码、API surface、示例、技术文章和截图复审。
5. 取得 Owner 对发布版本、发布渠道、候选哈希和发布动作的明确批准。
6. 发布后再从公开源执行仓库外 clean consumer 与回滚验证。

## 验证入口

首版候选至少执行：

```powershell
dotnet test tests/JYPPX.ProjectQuality.Tests/JYPPX.ProjectQuality.Tests.csproj -c Release
dotnet build TensorRtSharp.sln -c Release
pwsh -NoProfile -File eng/Test-ExternalVendorRuntimePackagePolicy.ps1
pwsh -NoProfile -File eng/Test-ReleaseCandidateReadiness.ps1 -AllowRuntimeSmokeBlocked
pwsh -NoProfile -File eng/Invoke-WindowsBridgePackageMatrix.ps1 -SkipPack -SkipManagedPack -SkipConsumerValidation
```

DocFX 构建、严格 output validator、真实模型专项测试和远程 Actions 仍应按改动范围补充执行。任何绿色汇总都不能自动打开发布门禁。

## 维护方式

本文件只更新“当前结论、权威入口和未完成项”，不再追加每次提交的实现日记。新增事实应优先写入结构化 evidence、专门技术文章或 Git 提交；本文件只链接它们。这样既减少仓库当前工作树体积，也避免旧数字与新证据并存造成误读。
