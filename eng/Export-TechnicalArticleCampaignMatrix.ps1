[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-Article {
  param(
    [int]$Id,
    [string]$Track,
    [string]$Title,
    [string]$TargetAudience,
    [string]$ArticleType,
    [string[]]$DependentSampleCodePaths,
    [bool]$RequiresRealModelAssets,
    [string]$ModelAcquisitionPlaceholder,
    [string[]]$ScreenshotsOrImagesNeeded,
    [string]$Readiness,
    [string]$ProofBoundary
  )

  [pscustomobject]@{
    id = $Id
    track = $Track
    title = $Title
    targetAudience = $TargetAudience
    articleType = $ArticleType
    dependentSampleCodePaths = @($DependentSampleCodePaths)
    requiresRealModelAssets = $RequiresRealModelAssets
    modelAcquisitionPlaceholder = $ModelAcquisitionPlaceholder
    screenshotsOrImagesNeeded = @($ScreenshotsOrImagesNeeded)
    readiness = $Readiness
    proofBoundary = $ProofBoundary
    performsPublish = $false
  }
}

$commonBoundary = "不得声称 NuGet 已公开发布、post-publish 已验证、package-consumer-runtime 已通过；template/dry-run/build-only/local feed/ProjectReference/direct .nupkg/report/sidecar/readonly diagnostics 均不是 proof。"
$modelBoundary = "真实模型、labels、输入资产、license、SHA256、运行日志和输出截图由 owner 补齐前，只能作为教程规划，不能声明 real-model-runtime proof。"

$articles = @(
  New-Article -Id 1 -Track "项目宣发" -Title "TensorRtSharp 4.0 项目总览：把 TensorRT/CUDA 带到 C# 生产工作流" -TargetAudience "技术负责人、.NET AI 工程师、项目评估者" -ArticleType "宣发" -DependentSampleCodePaths @("README.md","README.zh-CN.md","docs/index.md") -RequiresRealModelAssets $false -ModelAcquisitionPlaceholder "不需要模型资产" -ScreenshotsOrImagesNeeded @("项目架构图","版本矩阵表") -Readiness "outline-ready" -ProofBoundary $commonBoundary
  New-Article -Id 2 -Track "架构设计" -Title "TensorRtSharp 4.0 架构拆解：Native ABI、Manifest、Generator 与 C# Wrapper" -TargetAudience "维护者、绑定生成器开发者" -ArticleType "深入技术" -DependentSampleCodePaths @("native/manifests","native/src","src/JYPPX.TensorRtSharp") -RequiresRealModelAssets $false -ModelAcquisitionPlaceholder "不需要模型资产" -ScreenshotsOrImagesNeeded @("ABI 分层图","binding 生成流程图") -Readiness "outline-ready" -ProofBoundary $commonBoundary
  New-Article -Id 3 -Track "版本矩阵" -Title "CUDA / TensorRT / cuDNN 版本矩阵：TRT8、TRT10、TRT11 的边界与选择" -TargetAudience "部署工程师、维护者" -ArticleType "教程" -DependentSampleCodePaths @("artifacts/interface-coverage/tensorrt-interface-comparison.csv","artifacts/final-release/final-release-pre-publish-audit-matrix.json") -RequiresRealModelAssets $false -ModelAcquisitionPlaceholder "不需要模型资产" -ScreenshotsOrImagesNeeded @("版本兼容矩阵","runtime package 表") -Readiness "outline-ready" -ProofBoundary $commonBoundary
  New-Article -Id 4 -Track "Wrapper 生命周期" -Title "C# Wrapper 生命周期管理：SafeHandle、borrowed pointer 与 no-throw ABI" -TargetAudience ".NET 库开发者、维护者" -ArticleType "深入技术" -DependentSampleCodePaths @("src/JYPPX.TensorRtSharp","native/src/tensorrt/common") -RequiresRealModelAssets $false -ModelAcquisitionPlaceholder "不需要模型资产" -ScreenshotsOrImagesNeeded @("对象生命周期图","ownership ledger 摘要表") -Readiness "outline-ready" -ProofBoundary $commonBoundary
  New-Article -Id 5 -Track "NuGet 包" -Title "TensorRtSharp NuGet 包结构：Managed 包与 Native Runtime 包如何协作" -TargetAudience ".NET 使用者、发布维护者" -ArticleType "教程" -DependentSampleCodePaths @("src/JYPPX.TensorRtSharp","artifacts/final-release/release-evidence-bundle.json") -RequiresRealModelAssets $false -ModelAcquisitionPlaceholder "不需要模型资产" -ScreenshotsOrImagesNeeded @("包依赖关系图","runtime asset copy 表") -Readiness "outline-ready" -ProofBoundary $commonBoundary
  New-Article -Id 6 -Track "安装部署" -Title "Windows 上安装 TensorRtSharp：CUDA、TensorRT、PATH 与 runtime package 排查" -TargetAudience "Windows .NET 用户" -ArticleType "教程" -DependentSampleCodePaths @("docs/articles/zh-cn/cuda-error-35-troubleshooting.md","smoke") -RequiresRealModelAssets $false -ModelAcquisitionPlaceholder "可选 smoke 模型由 owner 提供" -ScreenshotsOrImagesNeeded @("环境变量截图","smoke 输出截图占位") -Readiness "outline-ready" -ProofBoundary $commonBoundary
  New-Article -Id 7 -Track "安装部署" -Title "Linux 上安装 TensorRtSharp：Runner、容器、Native 包与权限边界" -TargetAudience "Linux 部署工程师、CI 维护者" -ArticleType "教程" -DependentSampleCodePaths @("eng","docs/articles/zh-cn/compatible-host-runtime-proof-runbook.md") -RequiresRealModelAssets $false -ModelAcquisitionPlaceholder "可选 smoke 模型由 owner 提供" -ScreenshotsOrImagesNeeded @("Linux runner 表","ld path 诊断截图") -Readiness "outline-ready" -ProofBoundary $commonBoundary
  New-Article -Id 8 -Track "环境排查" -Title "CUDA/TensorRT/cuDNN 环境排查手册：从 driver mismatch 到 native asset copy" -TargetAudience "支持工程师、最终用户" -ArticleType "FAQ" -DependentSampleCodePaths @("docs/articles/zh-cn/cuda-error-35-troubleshooting.md","eng/Test-PackageConsumer.ps1") -RequiresRealModelAssets $false -ModelAcquisitionPlaceholder "不需要模型资产" -ScreenshotsOrImagesNeeded @("错误分类表","排查决策树") -Readiness "outline-ready" -ProofBoundary $commonBoundary
  New-Article -Id 9 -Track "OnnxToEngine" -Title "OnnxToEngine 快速上手：从 ONNX 到 serialized engine" -TargetAudience ".NET 推理工程师" -ArticleType "教程" -DependentSampleCodePaths @("samples/OnnxToEngine","docs/articles/zh-cn/onnx-to-engine-quickstart.md") -RequiresRealModelAssets $true -ModelAcquisitionPlaceholder "owner 提供可公开引用的 ONNX、license、SHA256 和 input shape" -ScreenshotsOrImagesNeeded @("命令行输出","engine 文件 hash") -Readiness "needs-real-model-assets" -ProofBoundary "$commonBoundary $modelBoundary"
  New-Article -Id 10 -Track "OnnxToEngine" -Title "OnnxToEngine 与官方 trtexec 对照：参数、报告与不可替代 proof 边界" -TargetAudience "TensorRT 迁移用户" -ArticleType "深入技术" -DependentSampleCodePaths @("samples/OnnxToEngine","applications/TensorRtExec","docs/articles/zh-cn/onnx-to-engine-trtexec-conversion-guide.md") -RequiresRealModelAssets $true -ModelAcquisitionPlaceholder "owner 提供对照 ONNX、trtexec 命令和输出 hash" -ScreenshotsOrImagesNeeded @("参数对照表","report JSON 摘要") -Readiness "needs-owner-proof" -ProofBoundary "$commonBoundary TensorRtExec report、OnnxToEngine report 和 sidecar 不能替代 runtime proof。"
  New-Article -Id 11 -Track "TensorRtExec" -Title "TensorRtExec CLI 教程：用 C# 复刻 trtexec 风格的 engine build 工作流" -TargetAudience "CLI 用户、自动化脚本维护者" -ArticleType "教程" -DependentSampleCodePaths @("applications/TensorRtExec/Console","applications/TensorRtExec/Core") -RequiresRealModelAssets $true -ModelAcquisitionPlaceholder "owner 提供教程 ONNX、输入 shape、build report 与 license" -ScreenshotsOrImagesNeeded @("CLI --help","build report 输出") -Readiness "needs-real-model-assets" -ProofBoundary "$commonBoundary CLI build/report 不是 package-consumer-runtime proof。"
  New-Article -Id 12 -Track "TensorRtExec" -Title "TensorRtExec WinForms 教程：可视化配置 ONNX Build、Profile 与 Report" -TargetAudience "桌面工具用户、演示场景" -ArticleType "教程" -DependentSampleCodePaths @("applications/TensorRtExec/WinForms","applications/TensorRtExec/Core") -RequiresRealModelAssets $true -ModelAcquisitionPlaceholder "owner 提供可演示模型、截图许可和输出报告" -ScreenshotsOrImagesNeeded @("WinForms 主界面","Profile 配置截图","Report 输出截图") -Readiness "needs-screenshots-and-assets" -ProofBoundary "$commonBoundary GUI screenshot 不是 runtime proof。"
  New-Article -Id 13 -Track "TensorRtExec" -Title "TensorRtExec GUI/CLI Parity 设计：一个选项如何同时进入 CLI、WinForms 和 Preview" -TargetAudience "维护者、工具开发者" -ArticleType "深入技术" -DependentSampleCodePaths @("artifacts/final-release/tensor-rt-exec-gui-cli-parity-checklist.json","applications/TensorRtExec") -RequiresRealModelAssets $false -ModelAcquisitionPlaceholder "不需要模型资产" -ScreenshotsOrImagesNeeded @("parity checklist 表","command preview 截图") -Readiness "outline-ready" -ProofBoundary "$commonBoundary parity checklist 是实现一致性证据，不是 runtime proof。"
  New-Article -Id 14 -Track "YoloVision" -Title "YoloVision 总览：一个样例覆盖 YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom" -TargetAudience "视觉模型用户、样例贡献者" -ArticleType "宣发" -DependentSampleCodePaths @("samples/YoloVision","docs/articles/zh-cn/yolovision-sample-overview.md") -RequiresRealModelAssets $true -ModelAcquisitionPlaceholder "owner 提供每个 family 的公开模型来源、license、labels、SHA256" -ScreenshotsOrImagesNeeded @("模型 family 表","det/cls/seg/obb/pose/sem 覆盖表") -Readiness "needs-real-model-assets" -ProofBoundary "$commonBoundary $modelBoundary"
  New-Article -Id 15 -Track "YoloVision" -Title "YOLOv5 detection 教程：预处理、engine build、后处理与结果可视化" -TargetAudience "YOLOv5 使用者" -ArticleType "教程" -DependentSampleCodePaths @("samples/YoloVision") -RequiresRealModelAssets $true -ModelAcquisitionPlaceholder "owner 提供 YOLOv5 ONNX、COCO labels、测试图片、license、SHA256" -ScreenshotsOrImagesNeeded @("检测框输出图","命令输出") -Readiness "needs-real-model-assets" -ProofBoundary "$commonBoundary $modelBoundary"
  New-Article -Id 16 -Track "YoloVision" -Title "YOLOv6 detection 教程：TensorRT engine 构建与输出解析" -TargetAudience "YOLOv6 使用者" -ArticleType "教程" -DependentSampleCodePaths @("samples/YoloVision") -RequiresRealModelAssets $true -ModelAcquisitionPlaceholder "owner 提供 YOLOv6 ONNX、labels、测试图片、license、SHA256" -ScreenshotsOrImagesNeeded @("检测结果截图","asset manifest") -Readiness "needs-real-model-assets" -ProofBoundary "$commonBoundary $modelBoundary"
  New-Article -Id 17 -Track "YoloVision" -Title "YOLOv7 detection/pose 教程：多输出解析和 keypoint 可视化" -TargetAudience "YOLOv7 使用者" -ArticleType "教程" -DependentSampleCodePaths @("samples/YoloVision") -RequiresRealModelAssets $true -ModelAcquisitionPlaceholder "owner 提供 YOLOv7 det/pose ONNX、labels、图片、license、SHA256" -ScreenshotsOrImagesNeeded @("检测结果","pose keypoint 图") -Readiness "needs-real-model-assets" -ProofBoundary "$commonBoundary $modelBoundary"
  New-Article -Id 18 -Track "YoloVision" -Title "YOLOv8 det/seg/pose/obb/cls 全任务教程" -TargetAudience "Ultralytics YOLOv8 用户" -ArticleType "教程" -DependentSampleCodePaths @("samples/YoloVision","docs/articles/zh-cn/yolovision-preprocess-postprocess.md") -RequiresRealModelAssets $true -ModelAcquisitionPlaceholder "owner 提供 YOLOv8 det/seg/pose/obb/cls 模型、labels、图片、license、SHA256" -ScreenshotsOrImagesNeeded @("det 框","seg mask","pose keypoint","obb 旋转框","cls TopK") -Readiness "needs-real-model-assets" -ProofBoundary "$commonBoundary $modelBoundary"
  New-Article -Id 19 -Track "YoloVision" -Title "YOLOv9 教程：新 family 接入、输出头识别与后处理策略" -TargetAudience "YOLOv9 使用者、样例维护者" -ArticleType "教程" -DependentSampleCodePaths @("samples/YoloVision") -RequiresRealModelAssets $true -ModelAcquisitionPlaceholder "owner 提供 YOLOv9 ONNX、labels、图片、license、SHA256" -ScreenshotsOrImagesNeeded @("输出头摘要","检测结果图") -Readiness "needs-real-model-assets" -ProofBoundary "$commonBoundary $modelBoundary"
  New-Article -Id 20 -Track "YoloVision" -Title "YOLOv10 教程：NMS 边界、输出布局与 TensorRT 推理链路" -TargetAudience "YOLOv10 使用者" -ArticleType "教程" -DependentSampleCodePaths @("samples/YoloVision") -RequiresRealModelAssets $true -ModelAcquisitionPlaceholder "owner 提供 YOLOv10 ONNX、labels、图片、license、SHA256" -ScreenshotsOrImagesNeeded @("输出布局表","检测结果图") -Readiness "needs-real-model-assets" -ProofBoundary "$commonBoundary $modelBoundary"
  New-Article -Id 21 -Track "YoloVision" -Title "YOLOv11 教程：从 ONNX 到 C# 后处理的完整链路" -TargetAudience "YOLOv11 使用者" -ArticleType "教程" -DependentSampleCodePaths @("samples/YoloVision") -RequiresRealModelAssets $true -ModelAcquisitionPlaceholder "owner 提供 YOLOv11 ONNX、labels、图片、license、SHA256" -ScreenshotsOrImagesNeeded @("命令输出","检测结果图") -Readiness "needs-real-model-assets" -ProofBoundary "$commonBoundary $modelBoundary"
  New-Article -Id 22 -Track "YoloVision" -Title "YOLOv26/custom 前向兼容方案：如何安全接入未知输出布局" -TargetAudience "自定义模型团队、样例贡献者" -ArticleType "深入技术" -DependentSampleCodePaths @("samples/YoloVision") -RequiresRealModelAssets $true -ModelAcquisitionPlaceholder "owner 提供 custom ONNX、输出 schema、labels、license、SHA256" -ScreenshotsOrImagesNeeded @("custom schema 表","兼容策略图") -Readiness "needs-real-model-assets" -ProofBoundary "$commonBoundary $modelBoundary"
  New-Article -Id 23 -Track "YoloVision" -Title "语义分割 sem 教程：mask tensor、palette 与结果落盘" -TargetAudience "分割模型使用者" -ArticleType "教程" -DependentSampleCodePaths @("samples/YoloVision") -RequiresRealModelAssets $true -ModelAcquisitionPlaceholder "owner 提供 sem ONNX、palette、测试图片、license、SHA256" -ScreenshotsOrImagesNeeded @("mask overlay","palette 表") -Readiness "needs-real-model-assets" -ProofBoundary "$commonBoundary $modelBoundary"
  New-Article -Id 24 -Track "YoloVision" -Title "OBB 旋转框后处理教程：角度、坐标系与可视化" -TargetAudience "遥感/工业视觉用户" -ArticleType "教程" -DependentSampleCodePaths @("samples/YoloVision") -RequiresRealModelAssets $true -ModelAcquisitionPlaceholder "owner 提供 OBB ONNX、测试图、labels、license、SHA256" -ScreenshotsOrImagesNeeded @("旋转框结果图","坐标系说明图") -Readiness "needs-real-model-assets" -ProofBoundary "$commonBoundary $modelBoundary"
  New-Article -Id 25 -Track "YoloVision" -Title "Pose keypoint 教程：关键点 tensor、骨架连接与评分过滤" -TargetAudience "姿态估计使用者" -ArticleType "教程" -DependentSampleCodePaths @("samples/YoloVision") -RequiresRealModelAssets $true -ModelAcquisitionPlaceholder "owner 提供 pose ONNX、测试图片、skeleton 配置、license、SHA256" -ScreenshotsOrImagesNeeded @("keypoint 输出图","skeleton 配置表") -Readiness "needs-real-model-assets" -ProofBoundary "$commonBoundary $modelBoundary"
  New-Article -Id 26 -Track "INT8/Calibration" -Title "INT8 与 Calibration Ownership：为什么 callback/allocator 不能仓促开放" -TargetAudience "性能优化工程师、维护者" -ArticleType "深入技术" -DependentSampleCodePaths @("docs/articles/zh-cn/allocator-owner-ledger-design.md","native/src/tensorrt") -RequiresRealModelAssets $false -ModelAcquisitionPlaceholder "后续 INT8 smoke 需要 owner 提供校准集" -ScreenshotsOrImagesNeeded @("ownership 风险表","calibration 生命周期图") -Readiness "design-ready" -ProofBoundary "$commonBoundary INT8 diagnostic/cache 不能替代真实 calibration runtime proof。"
  New-Article -Id 27 -Track "Plugin" -Title "Plugin Registry Inventory：安全只读 API 如何避免 borrowed pointer 泄露" -TargetAudience "TensorRT plugin 用户、维护者" -ArticleType "深入技术" -DependentSampleCodePaths @("src/JYPPX.TensorRtSharp/Plugins/TensorRtPluginRegistryInventory.cs","native/src/tensorrt") -RequiresRealModelAssets $false -ModelAcquisitionPlaceholder "不需要模型资产" -ScreenshotsOrImagesNeeded @("plugin creator 表","只读 copy 模式图") -Readiness "outline-ready" -ProofBoundary "$commonBoundary plugin inventory 是只读诊断，不证明 plugin load/register/deregister 或 enqueue。"
  New-Article -Id 28 -Track "Diagnostics" -Title "Engine Inspector 与只读 Diagnostics：如何读懂 layer info 而不越界声明 proof" -TargetAudience "推理调优工程师" -ArticleType "深入技术" -DependentSampleCodePaths @("src/JYPPX.TensorRtSharp/Engine/TensorRtEngineInspector.Trt11Diagnostics.cs","applications/TensorRtExec") -RequiresRealModelAssets $true -ModelAcquisitionPlaceholder "owner 提供可公开 engine 或 ONNX、layer info 输出、license、SHA256" -ScreenshotsOrImagesNeeded @("layer info 表","profile 摘要") -Readiness "needs-real-model-assets" -ProofBoundary "$commonBoundary layer/profile diagnostics 不是输出正确性 proof。"
  New-Article -Id 29 -Track "Proof" -Title "Package Consumer Runtime Proof：为什么必须用仓库外 clean consumer" -TargetAudience "release owner、维护者、企业用户" -ArticleType "深入技术" -DependentSampleCodePaths @("artifacts/final-release/clean-consumer-runtime-proof-execution-checklist.json","eng/Test-PackageConsumerRuntimeProofRecord.ps1") -RequiresRealModelAssets $false -ModelAcquisitionPlaceholder "不需要模型资产，但需要 owner 提供真实运行日志和 hash" -ScreenshotsOrImagesNeeded @("proof checklist","validator 输出") -Readiness "owner-action-required" -ProofBoundary "$commonBoundary 只有 strict validator 通过的 clean package consumer runtime smoke 才能提升 proof。"
  New-Article -Id 30 -Track "Proof" -Title "发布前 Proof 与 Post-Publish Verification：从 release hold 到最终关闭" -TargetAudience "release owner、项目维护者" -ArticleType "深入技术" -DependentSampleCodePaths @("artifacts/final-release/final-proof-owner-handoff-pack.json","artifacts/final-release/final-release-close-blocker-dashboard.json") -RequiresRealModelAssets $false -ModelAcquisitionPlaceholder "不需要模型资产，但需要 owner 提供真实 post-publish 安装日志" -ScreenshotsOrImagesNeeded @("release hold dashboard","post-publish validator 表") -Readiness "owner-action-required" -ProofBoundary "$commonBoundary post-publish 未真实验证前不能声称 release closed。"
  New-Article -Id 31 -Track "FAQ" -Title "TensorRtSharp 4.0 常见问题：安装、版本、模型、性能与 proof 边界" -TargetAudience "所有用户" -ArticleType "FAQ" -DependentSampleCodePaths @("docs/index.md","README.zh-CN.md") -RequiresRealModelAssets $false -ModelAcquisitionPlaceholder "不需要模型资产" -ScreenshotsOrImagesNeeded @("FAQ 分类表") -Readiness "outline-ready" -ProofBoundary $commonBoundary
  New-Article -Id 32 -Track "性能调优" -Title "TensorRT 性能调优案例：FP16、profile、workspace、timing cache 与 benchmark 边界" -TargetAudience "性能优化工程师" -ArticleType "案例" -DependentSampleCodePaths @("applications/TensorRtExec","samples/OnnxToEngine") -RequiresRealModelAssets $true -ModelAcquisitionPlaceholder "owner 提供 benchmark 模型、输入、host metadata、日志、license、SHA256" -ScreenshotsOrImagesNeeded @("benchmark 表","profile 摘要图") -Readiness "needs-real-model-assets" -ProofBoundary "$commonBoundary benchmark-shaped settings 不是 package-consumer-runtime proof。"
  New-Article -Id 33 -Track "多平台部署" -Title "多平台部署案例：win-x64、linux-x64、CUDA 12/13 与 TensorRT 10/11" -TargetAudience "部署工程师、CI 维护者" -ArticleType "案例" -DependentSampleCodePaths @("eng","artifacts/final-release/final-release-pre-publish-audit-matrix.json") -RequiresRealModelAssets $false -ModelAcquisitionPlaceholder "不需要模型资产，但需要各平台 owner proof" -ScreenshotsOrImagesNeeded @("平台矩阵","runner proof 表") -Readiness "owner-action-required" -ProofBoundary "$commonBoundary Linux runner proof 和 package-consumer-runtime proof 必须由真实 host 产生。"
  New-Article -Id 34 -Track "Samples" -Title "Classification 样例教程：从模型资产清单到 TopK 输出" -TargetAudience "分类模型用户" -ArticleType "教程" -DependentSampleCodePaths @("samples/Classification","samples/assets") -RequiresRealModelAssets $true -ModelAcquisitionPlaceholder "owner 提供 classifier ONNX、labels、图片、license、SHA256" -ScreenshotsOrImagesNeeded @("TopK 输出","asset manifest") -Readiness "needs-real-model-assets" -ProofBoundary "$commonBoundary $modelBoundary"
  New-Article -Id 35 -Track "Samples" -Title "Dynamic Shape Optimization Profile：C# 中的 min/opt/max shape 实战" -TargetAudience "动态 shape 模型用户" -ArticleType "教程" -DependentSampleCodePaths @("samples/OnnxToEngine","src/JYPPX.TensorRtSharp.Tools") -RequiresRealModelAssets $true -ModelAcquisitionPlaceholder "owner 提供动态 shape ONNX、shape 范围、license、SHA256" -ScreenshotsOrImagesNeeded @("profile 配置表","build report") -Readiness "needs-real-model-assets" -ProofBoundary "$commonBoundary profile build 不是每个 shape 均已 runtime 验证的 proof。"
  New-Article -Id 36 -Track "Samples" -Title "CUDA Stream/Event 多流教程：异步边界、同步策略与 C# 调用习惯" -TargetAudience "CUDA/C# 性能用户" -ArticleType "教程" -DependentSampleCodePaths @("src/JYPPX.Cuda","smoke") -RequiresRealModelAssets $false -ModelAcquisitionPlaceholder "不需要模型资产，可用 CUDA smoke" -ScreenshotsOrImagesNeeded @("多流时序图","smoke 输出") -Readiness "outline-ready" -ProofBoundary $commonBoundary
  New-Article -Id 37 -Track "API 教程" -Title "Inference Bindings 教程：输入输出 tensor、buffer、shape 与 copy 策略" -TargetAudience ".NET 推理开发者" -ArticleType "教程" -DependentSampleCodePaths @("src/JYPPX.TensorRtSharp","samples/YoloVision") -RequiresRealModelAssets $true -ModelAcquisitionPlaceholder "owner 提供可运行 ONNX/engine、输入输出样本、license、SHA256" -ScreenshotsOrImagesNeeded @("binding 表","输出 tensor 摘要") -Readiness "needs-real-model-assets" -ProofBoundary "$commonBoundary binding/output proof 需要真实 runtime log 和 hash。"
  New-Article -Id 38 -Track "API 教程" -Title "ONNX Parser 到 Serialized Engine：错误诊断、profile 与 report 联动" -TargetAudience "模型转换工程师" -ArticleType "教程" -DependentSampleCodePaths @("samples/OnnxToEngine","src/JYPPX.TensorRtSharp.Tools") -RequiresRealModelAssets $true -ModelAcquisitionPlaceholder "owner 提供 ONNX、parser log、license、SHA256" -ScreenshotsOrImagesNeeded @("parser error 表","report 摘要") -Readiness "needs-real-model-assets" -ProofBoundary "$commonBoundary parser/build report 不能替代 runtime proof。"
  New-Article -Id 39 -Track "API 教程" -Title "Refit Weights 指南：权重替换、校验和 release 风险" -TargetAudience "模型维护者、高级用户" -ArticleType "深入技术" -DependentSampleCodePaths @("src/JYPPX.TensorRtSharp","native/src/tensorrt") -RequiresRealModelAssets $true -ModelAcquisitionPlaceholder "owner 提供 refit 测试模型、权重、输出对照、license、SHA256" -ScreenshotsOrImagesNeeded @("refit 流程图","权重映射表") -Readiness "needs-real-model-assets" -ProofBoundary "$commonBoundary refit 修改必须由真实输出对照和 runtime log 验证。"
  New-Article -Id 40 -Track "发布宣发" -Title "TensorRtSharp 4.0 发布故事：从 deferred 边界提升到可审计 release gate" -TargetAudience "社区读者、技术管理者、贡献者" -ArticleType "宣发" -DependentSampleCodePaths @("plan","diary","artifacts/final-release") -RequiresRealModelAssets $false -ModelAcquisitionPlaceholder "不需要模型资产" -ScreenshotsOrImagesNeeded @("阶段路线图","release gate 截图") -Readiness "outline-ready" -ProofBoundary $commonBoundary
)

$tracks = @(
  [pscustomobject]@{ id = "project-promo"; name = "项目宣发与发布故事"; articleIds = @(1, 40); purpose = "面向公众号/博客解释项目价值与 release 边界。" }
  [pscustomobject]@{ id = "architecture"; name = "架构与 ABI 安全"; articleIds = @(2, 3, 4, 26, 27, 28); purpose = "解释 Native ABI、Wrapper 生命周期、Plugin/Diagnostics 只读边界。" }
  [pscustomobject]@{ id = "install-packaging"; name = "安装、NuGet 与环境排查"; articleIds = @(5, 6, 7, 8, 31, 33); purpose = "降低新用户安装、版本选择和排障成本。" }
  [pscustomobject]@{ id = "onnx-tensorrtexec"; name = "OnnxToEngine 与 TensorRtExec"; articleIds = @(9, 10, 11, 12, 13, 35, 38); purpose = "形成 engine build、GUI/CLI parity 和 trtexec 对照教程线。" }
  [pscustomobject]@{ id = "yolovision"; name = "YoloVision 全家族教程"; articleIds = @(14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25); purpose = "覆盖 YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom 与 det/cls/seg/obb/pose/sem。" }
  [pscustomobject]@{ id = "proof-release"; name = "Proof 与发布收口"; articleIds = @(29, 30); purpose = "解释 package-consumer-runtime、post-publish verification 和 release close gate。" }
  [pscustomobject]@{ id = "advanced-samples"; name = "高级样例与性能"; articleIds = @(32, 34, 36, 37, 39); purpose = "补齐性能调优、Classification、CUDA stream/event、bindings 和 refit 场景。" }
)

$sourceCodeSampleLinks = @(
  "samples/YoloVision",
  "samples/OnnxToEngine",
  "samples/Classification",
  "applications/TensorRtExec",
  "src/JYPPX.TensorRtSharp",
  "src/JYPPX.TensorRtSharp.Tools",
  "src/JYPPX.Cuda",
  "native/src/tensorrt",
  "native/manifests/tensorrt",
  "eng",
  "artifacts/final-release"
)

$assetRequirements = @(
  [pscustomobject]@{ id = "public-yolo-models"; description = "YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom ONNX, labels, images, license, SHA256."; blockingArticleIds = @(14, 15, 16, 17, 18, 19, 20, 21, 22) }
  [pscustomobject]@{ id = "yolo-task-assets"; description = "det/cls/seg/obb/pose/sem task-specific assets and screenshots."; blockingArticleIds = @(18, 23, 24, 25) }
  [pscustomobject]@{ id = "onnx-engine-assets"; description = "ONNX models, shape profiles, engine hashes, build reports, parser logs."; blockingArticleIds = @(9, 10, 35, 38) }
  [pscustomobject]@{ id = "tensorrtexec-gui-assets"; description = "TensorRtExec CLI/WinForms screenshots and report screenshots."; blockingArticleIds = @(11, 12, 13) }
  [pscustomobject]@{ id = "runtime-proof-assets"; description = "Clean consumer logs, package hashes, host metadata, strict validator output, post-publish install logs."; blockingArticleIds = @(29, 30, 33) }
  [pscustomobject]@{ id = "advanced-sample-assets"; description = "Classification, refit, performance and bindings sample models with licenses and output references."; blockingArticleIds = @(32, 34, 37, 39) }
)

$articleArray = @($articles)
$articlesRequiringAssets = @($articleArray | Where-Object { [bool]$_.requiresRealModelAssets })
$ownerActionArticles = @($articleArray | Where-Object { $_.readiness -match "owner|needs" })
$p0Articles = @($articleArray | Where-Object { $_.readiness -eq "outline-ready" -or $_.readiness -eq "design-ready" -or $_.track -match "overview|tooling|release" })
$articleExecutionStages = @(
  [pscustomobject]@{ id = "stage-01-p0-public-drafts"; title = "P0 public drafts"; objective = "Complete project overview, install, source build, native build, OnnxToEngine, TensorRtExec, YoloVision overview, and release evidence ladder drafts."; articleIds = @(1, 2, 3, 4, 5, 9, 10, 11, 12, 15, 29, 30); ownerAction = "Draft text only; do not claim public publish or post-publish proof."; status = "ready-to-write" }
  [pscustomobject]@{ id = "stage-02-real-asset-walkthroughs"; title = "Real asset walkthroughs"; objective = "Prepare YoloVision det/cls/seg/obb/pose/sem, Classification, DynamicShape, and sample evidence articles around owner-provided assets."; articleIds = @(16, 17, 18, 19, 20, 21, 22, 23, 24, 32, 36); ownerAction = "Wait for model/image/label/license/hash bundles before public-ready claims."; status = "blocked-owner-assets" }
  [pscustomobject]@{ id = "stage-03-package-consumer-and-post-publish"; title = "Package consumer and post-publish proof articles"; objective = "Explain clean external consumer, public package channels, strict validators, forbidden substitutes, and close gate boundaries."; articleIds = @(5, 7, 8, 29, 30, 33); ownerAction = "Use validator outputs only after real owner execution result import passes."; status = "blocked-owner-proof" }
  [pscustomobject]@{ id = "stage-04-advanced-boundary-roadmap"; title = "Advanced boundary roadmap"; objective = "Cover callback, allocator, listener, Plugin Registry inventory, Engine Inspector, and deferred boundary next steps."; articleIds = @(9, 10, 11, 27, 28, 37); ownerAction = "Keep callback/allocator/listener write-ups as boundary design until safe wrappers and runtime proof exist."; status = "design-ready" }
)
$contentReadinessDefinition = [pscustomobject]@{
  outlineReady = "Can be drafted from repository source and existing docs."
  ownerAssetsRequired = "Requires owner-provided real model assets, screenshots, logs, licenses, and SHA256 values."
  ownerProofRequired = "Requires strict-validator accepted owner proof records before any public-ready proof claim."
  publicClaimBlockedUntil = @("real-model-runtime proof", "package-consumer-runtime proof", "post-publish verification", "release close strict validation")
}

$record = [pscustomobject]@{
  recordKind = "technical-article-campaign-matrix"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  campaignState = "publication-campaign-planning-non-proof"
  articleCount = $articleArray.Count
  articleTracks = @($tracks)
  articleExecutionStages = @($articleExecutionStages)
  sourceCodeSampleLinks = @($sourceCodeSampleLinks)
  assetRequirements = @($assetRequirements)
  publishingReadiness = [pscustomobject]@{
    state = "blocked-real-assets-and-proof-before-public-claims"
    outlineReadyCount = @($articleArray | Where-Object { $_.readiness -eq "outline-ready" -or $_.readiness -eq "design-ready" }).Count
    p0DraftCandidateCount = $p0Articles.Count
    ownerActionRequiredCount = $ownerActionArticles.Count
    articlesRequiringRealModelAssets = $articlesRequiringAssets.Count
    executionStageCount = $articleExecutionStages.Count
    publicClaimBoundary = $commonBoundary
  }
  contentReadinessDefinition = $contentReadinessDefinition
  missingAssetCount = $assetRequirements.Count
  requiresRealModelAssets = $true
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  yoloFamilies = @("yolov5","yolov6","yolov7","yolov8","yolov9","yolov10","yolov11","yolov26","custom")
  yoloTasks = @("det","cls","seg","obb","pose","sem")
  forbiddenClaims = @(
    "NuGet 已公开发布",
    "post-publish 已验证",
    "package-consumer-runtime 已通过",
    "release close 已完成",
    "文章或截图可替代 proof",
    "candidate/dashboard/dry-run 可晋级 proof",
    "canPublishPublicly 被置为 true",
    "canCloseReleaseIssue 被置为 true",
    "canPromoteRuntimeProof 被置为 true"
  )
  forbiddenSubstituteMarkers = @(
    "candidate",
    "draft",
    "dashboard",
    "dry-run",
    "local feed",
    "ProjectReference",
    "direct .nupkg",
    "template",
    "build-only",
    "blocked-by-cuda-driver"
  )
  proofPromotionBoundary = [pscustomobject]@{
    articlesAreNotProof = $true
    screenshotsAreNotProof = $true
    matricesAreNotProof = $true
    ownerAssetsAreRequiredForCaseStudies = $true
    strictValidatorRequiredForProofClaims = $true
    publicReleaseClaimsBlocked = $true
  }
  proofBoundary = $commonBoundary
  articles = $articleArray
}

$jsonPath = Join-Path $OutputRoot "technical-article-campaign-matrix.json"
$markdownPath = Join-Path $OutputRoot "technical-article-campaign-matrix.md"
$record | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$trackRows = $tracks | ForEach-Object {
  "| ``$($_.id)`` | $(ConvertTo-MarkdownCell $_.name) | $($_.articleIds -join ', ') | $(ConvertTo-MarkdownCell $_.purpose) |"
}
$stageRows = $articleExecutionStages | ForEach-Object {
  "| ``$($_.id)`` | $(ConvertTo-MarkdownCell $_.title) | $($_.articleIds -join ', ') | $(ConvertTo-MarkdownCell $_.status) | $(ConvertTo-MarkdownCell $_.ownerAction) |"
}
$articleRows = $articleArray | ForEach-Object {
  $paths = ($_.dependentSampleCodePaths -join "<br>")
  $images = ($_.screenshotsOrImagesNeeded -join "<br>")
  "| $($_.id) | $(ConvertTo-MarkdownCell $_.track) | $(ConvertTo-MarkdownCell $_.title) | ``$($_.articleType)`` | $(ConvertTo-MarkdownCell $_.targetAudience) | $paths | ``$($_.requiresRealModelAssets)`` | $(ConvertTo-MarkdownCell $_.readiness) | $images |"
}
$assetRows = $assetRequirements | ForEach-Object {
  "| ``$($_.id)`` | $(ConvertTo-MarkdownCell $_.description) | $($_.blockingArticleIds -join ', ') |"
}

$markdown = @"
# Technical Article Campaign Matrix

| Field | Value |
| --- | --- |
| recordKind | ``$($record.recordKind)`` |
| campaignState | ``$($record.campaignState)`` |
| articleCount | ``$($record.articleCount)`` |
| executionStageCount | ``$($record.publishingReadiness.executionStageCount)`` |
| p0DraftCandidateCount | ``$($record.publishingReadiness.p0DraftCandidateCount)`` |
| missingAssetCount | ``$($record.missingAssetCount)`` |
| requiresRealModelAssets | ``$($record.requiresRealModelAssets)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |
| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |

## Tracks

| ID | Name | Article IDs | Purpose |
|---|---|---|---|
$($trackRows -join "`r`n")

## Execution Stages

| ID | Title | Article IDs | Status | Owner Action |
|---|---|---|---|---|
$($stageRows -join "`r`n")

## Asset Requirements

| ID | Description | Blocking Article IDs |
|---|---|---|
$($assetRows -join "`r`n")

## Articles

| ID | Track | Title | Type | Target Audience | Sample/Code Paths | Needs Real Assets | Readiness | Screenshot/Image Needs |
|---:|---|---|---|---|---|---:|---|---|
$($articleRows -join "`r`n")

## YOLO Scope

- families: ``$($record.yoloFamilies -join ', ')``
- tasks: ``$($record.yoloTasks -join ', ')``

## Proof Boundary

$($record.proofBoundary)

## Proof Promotion Boundary

- articlesAreNotProof: ``$($record.proofPromotionBoundary.articlesAreNotProof)``
- screenshotsAreNotProof: ``$($record.proofPromotionBoundary.screenshotsAreNotProof)``
- strictValidatorRequiredForProofClaims: ``$($record.proofPromotionBoundary.strictValidatorRequiredForProofClaims)``
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Technical article campaign matrix written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ArticleCount=$($record.articleCount) MissingAssetCount=$($record.missingAssetCount) CanPublish=$($record.canPublishPublicly)"
