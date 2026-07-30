# TensorRtSharp4.0 项目总览：把 TensorRT 带到可发布的 C# 工程实践里

很多 C# 开发者第一次接触 TensorRT 时，都会遇到同一个问题：官方生态最成熟的路径集中在 C++ 和 Python，而真正落到 .NET 桌面程序、服务端推理、工业视觉和内部工具链时，往往要在 P/Invoke、CUDA DLL、TensorRT ABI、模型转换、NuGet 分发和 runtime proof 之间来回补洞。

TensorRtSharp4.0 想解决的不是“把几个函数声明搬到 C#”这么简单的事。它的目标是提供一套面向工程交付的 TensorRT / CUDA C# API：底层有清晰的 native bridge 和跨版本 guard，上层有更符合 .NET 使用习惯的 wrapper，中间有 OnnxToEngine、TensorRtExec、YoloVision、runtime packages、smoke、quality gate 和 release evidence，让用户能从项目介绍一路走到真实验证路径。

本文是面向微信公众号、博客和项目主页的完整项目介绍稿。它可以宣传项目能力，但仍保持 release proof boundary：文章、截图、matrix、build-only report、template 和 local feed 都不是 package-consumer-runtime proof。

## 适合谁阅读

- 正在用 C#/.NET 做 AI 推理工具、桌面应用、工业视觉或服务端部署的开发者。
- 想把 ONNX 模型转换为 TensorRT engine，但不希望每次都切到 C++/Python 工具链的工程团队。
- 需要同时维护 CUDA、TensorRT 8/10/11、cuDNN、多 runtime 包和 NuGet/GitHub 发布路线的库维护者。
- 想理解 TensorRT C# binding 如何处理 ABI、ownership、deferred API 与真实 proof 的技术读者。
- 需要评估 TensorRtSharp4.0 是否适合项目落地、文章宣传和后续二次开发的技术负责人。

## 项目一句话

TensorRtSharp4.0 是一个把 NVIDIA TensorRT / CUDA 能力带入 .NET 工程工作流的 C# API 项目。它通过 C++ native bridge 隔离 TensorRT/CUDA ABI，通过 generated interop 保持 manifest 和入口一致，通过 SafeHandle 与高层 wrapper 管理生命周期，通过 samples/applications 提供真实使用入口，通过 quality gate 和 release evidence 防止把“能编译”误写成“能发布”。

你可以把它理解为四层结构：

```text
C# user code
  -> high-level wrappers and tools
  -> generated interop and NativeBridgeApi
  -> C ABI native bridge
  -> CUDA / TensorRT / cuDNN
```

对应仓库路径：

```text
src/JYPPX.TensorRtSharp
src/JYPPX.CudaSharp
src/JYPPX.TensorRtSharp.Tools
src/JYPPX.Shared
native/src/tensorrt
native/src/cuda
native/manifests/tensorrt/v8
native/manifests/tensorrt/v10
native/manifests/tensorrt/v11
native/manifests/cuda
samples/OnnxToEngine
samples/YoloVision
applications/TensorRtExec
pack/runtime
pack/runtime-split
artifacts/final-release
```

## 为什么不是简单 P/Invoke

TensorRT 的 C++ API 不是为跨语言裸调用设计的。对象生命周期、borrowed pointer、callback、plugin registry、builder config、engine/context、error recorder、allocator 和外部资源如果处理不好，会出现三类问题：

- ABI 层面：TRT8、TRT10、TRT11 的函数、类型和行为并不完全一致。
- 生命周期层面：C# 如果直接暴露无语义 `IntPtr`，用户很容易持有悬空对象或释放顺序错误。
- 诊断层面：调用失败后如果没有状态码、日志、copied metadata、preflight 和 smoke 证据，问题很难定位。

所以 TensorRtSharp4.0 使用 native bridge 把跨 ABI 异常隔离在边界内，用 manifest 和 generated binding 保证入口一致，再在 C# 层提供更安全的封装，例如：

```text
TensorRtBuilder
TensorRtBuilderConfig
TensorRtRuntime
TensorRtEngine
TensorRtExecutionContext
TensorRtOnnxParser
TensorRtOnnxParserRefitter
TensorRtPluginRegistryInventory
TensorRtEngineInspector
CudaDevice
CudaEnvironmentProbe
TensorRtEnvironmentProbe
```

这也是项目持续推进 deferred uplift 的原因：入口存在不等于真实 API 可用。只有 native 实现、version guard、托管路由、高层 wrapper、文档、smoke 和质量门禁一起闭环，才能把 deferred 接口提升为用户可调用能力。

## 四条主线

第一条是底层 API 与 ABI 稳定性。

仓库维护 TensorRT 8、TensorRT 10、TensorRT 11 以及 CUDA runtime 的 manifest、native bridge、generated interop 和 C# route。关键证据包括：

```text
native/generated/bridge_api_catalog.g.h
native/generated/bridge_entrypoints.g.h
src/JYPPX.Shared/Generated/GeneratedApiCatalog.g.cs
src/JYPPX.Shared/Generated/GeneratedEntryPointNames.g.cs
src/JYPPX.Shared/Generated/GeneratedNativeMethods.g.cs
src/JYPPX.TensorRtSharp/Internal/Interop/Generated/GeneratedTensorRtManifestNativeMethods.g.cs
src/JYPPX.CudaSharp/Internal/Interop/Generated/GeneratedCudaManifestNativeMethods.g.cs
artifacts/interface-coverage/tensorrt-interface-comparison.csv
artifacts/interface-coverage/project-completion-review.md
```

当前主线已经从“missing 接口清零”切到“deferred 边界提升”。完成度不能只看 manifest/source 是否 100% 对齐，还要看 wrapper、runtime smoke、package consumer 和 proof boundary。

第二条是样例与应用。

`samples/OnnxToEngine` 面向 ONNX 到 TensorRT engine 的转换，目标是贴近官方 `trtexec` 的模型转换能力，并输出 build/report/readback evidence。

`applications/TensorRtExec` 是 trtexec-like 应用，既支持 CLI，也支持 WinForms。它通过 `TensorRtExecCommand.cs`、`TensorRtExecService.cs`、`TensorRtExecReport.cs`、`MainForm.cs`、`TrtexecLikeParser` 和 `TrtexecLikeOptions` 把命令行、GUI、report 和 parity matrix 串起来。

`samples/YoloVision` 替代早期过窄的 detection demo，统一覆盖 YOLOv5、YOLOv6、YOLOv7、YOLOv8、YOLOv9、YOLOv10、YOLO11、YOLO26、YOLOX 和 custom 模型，并支持 det、cls、seg、obb、pose、sem 六任务。

第三条是安装与包。

项目保留两条分发路线：

```text
GitHub Release managed + bridge assets
NuGet managed + bridge package route
```

两条路线都只交付 C# 核心 API、工具库和项目自有 C++ bridge，不打包 NVIDIA 原厂 runtime。GitHub Release 提供不可变 URL 与 digest，NuGet-compatible source 提供标准 `PackageReference`；用户自行安装 TensorRT、CUDA、cuDNN 和可选 NVRTC。相关 manifest 和文章包括：

```text
pack/runtime/runtime-packages.manifest.json
pack/runtime-split/split-runtime-packages.manifest.json
docs/articles/zh-cn/publishing/package-strategy-public-article.md
docs/articles/zh-cn/publishing/nuget-install-runtime-package-public-article.md
docs/articles/zh-cn/publishing/native-bridge-build-public-article.md
docs/articles/zh-cn/publishing/source-build-windows-public-article.md
docs/articles/zh-cn/publishing/cuda-tensorrt-dll-troubleshooting-public-article.md
```

第四条是 release evidence。

项目不把 build-only、template、local feed、ProjectReference 或 direct `.nupkg` 当成发布 proof。真实 proof 必须来自真实输入、真实日志、hash、host metadata、owner review 和 strict validator。关键文件包括：

```text
artifacts/final-release/release-evidence-bundle.json
artifacts/final-release/release-close-preflight.json
artifacts/final-release/final-release-close-blocker-dashboard.md
artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json
artifacts/final-release/post-publish-verification-record.json
artifacts/final-release/technical-article-publication-matrix.md
artifacts/final-release/technical-article-campaign-matrix.md
```

## 用户从哪里开始

如果你只是想了解项目，可以从这些入口开始：

```text
README.md
README.zh-CN.md
docs/index.md
docs/toc.yml
docs/articles/zh-cn/project-overview.md
docs/articles/zh-cn/tensorrtsharp-4-project-overview-campaign.md
docs/articles/zh-cn/project-release-story-and-boundaries.md
```

如果你想安装和排查：

```text
docs/articles/zh-cn/publishing/nuget-install-runtime-package-public-article.md
docs/articles/zh-cn/publishing/cuda-tensorrt-dll-troubleshooting-public-article.md
docs/articles/zh-cn/runtime-package-selection.md
docs/articles/zh-cn/runtime-package-native-load-troubleshooting.md
docs/articles/zh-cn/package-consumer-runtime-proof-clean-consumer-guide.md
```

如果你想转换模型：

```text
docs/articles/zh-cn/publishing/onnx-to-engine-public-article.md
docs/articles/zh-cn/publishing/onnxtoengine-trtexec-parity-public-article.md
samples/OnnxToEngine/Program.cs
applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json
applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json
```

如果你想做 YOLO 系列案例：

```text
docs/articles/zh-cn/publishing/yolovision-overview-public-article.md
samples/YoloVision/README.md
samples/YoloVision/yolo-model-matrix.json
samples/YoloVision/yolovision-task-output-contract.json
samples/assets/yolovision-article-case-pack.json
samples/assets/yolovision-real-asset-owner-backfill-pack.json
```

## 典型使用路径

第一步，检查项目入口：

```powershell
dotnet build .\TensorRtSharp.sln -c Debug --no-restore
dotnet run --project .\applications\TensorRtExec\TensorRtExec.csproj -- --help
dotnet run --project .\samples\OnnxToEngine\OnnxToEngine.csproj -- --help
dotnet run --project .\samples\YoloVision\YoloVision.csproj -- --help
```

第二步，用 OnnxToEngine 或 TensorRtExec 构建 engine：

```powershell
dotnet run --project .\applications\TensorRtExec\TensorRtExec.csproj -- `
  --onnx E:\TensorRtSharpAssets\models\model.onnx `
  --saveEngine E:\TensorRtSharpAssets\engines\model.plan `
  --minShapes images:1x3x640x640 `
  --optShapes images:1x3x640x640 `
  --maxShapes images:4x3x640x640 `
  --fp16 `
  --buildOnly `
  --exportReport E:\TensorRtSharpAssets\reports\model-build-report.json
```

第三步，用 YoloVision 或自己的程序加载模型/engine：

```powershell
dotnet run --project .\samples\YoloVision\YoloVision.csproj -- `
  --model E:\TensorRtSharpAssets\models\yolov8-det.onnx `
  --labels E:\TensorRtSharpAssets\models\coco.names `
  --image E:\TensorRtSharpAssets\images\dog.ppm `
  --preprocessed-output E:\TensorRtSharpAssets\tensors\dog-yolov8-fp32.bin `
  --input-shape 1x3x640x640 `
  --family v8 `
  --task det `
  --layout auto `
  --has-objectness auto `
  --nms-mode class-aware `
  --output E:\TensorRtSharpAssets\reports\yolov8-det-output.json `
  --visualization E:\TensorRtSharpAssets\reports\yolov8-det-output.svg
```

第四步，如果要作为 release proof，使用 owner validator 或外部 consumer validator 记录证据：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionRealAssetCandidate.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof
```

对应脚本路径：

```text
eng/Test-YoloVisionRealAssetCandidate.ps1
eng/Test-SampleRunEvidenceRecord.ps1
eng/Test-ExternalRuntimeProofRecord.ps1
```

这四步的证据级别不同：help/build/report 证明入口和构建路径；YoloVision 证明某个样例和模型资产可运行；sample-run evidence 只可能晋级 real-model-runtime；公开包的 package-consumer-runtime proof 还需要仓库外 clean consumer、public package source、真实 smoke、hash、host metadata 和 strict validator。

## 当前完成度怎么看

项目已经有大量接口、样例、文档和质量门禁，但仍然明确保留边界：

```text
manifest/source match != runtime proof
generated interop != high-level wrapper
build-only report != package-consumer-runtime proof
local feed != public package source
ProjectReference != package consumer
direct .nupkg install != post-publish verification
YoloVision matrix != real-model-runtime proof
TensorRtExec report != release close approval
```

真正的完成度要看这些证据是否同时成立：

- TRT8/TRT10/TRT11 manifest、native implementation、generated interop、托管路由和 version guard 一致。
- 高层 C# wrapper 不暴露无语义 `IntPtr` 或不透明生命周期。
- 对只读、查询型、部署关键型 API 有 smoke/quality gate。
- 对 callback、allocator、plugin lifecycle、borrowed pointer、external resource、runtime deserialization ownership 保持 owner 设计门禁。
- package-consumer-runtime、Linux runner proof、real-model-runtime、owner authorization 和 post-publish verification 真实存在。

## 文章矩阵

项目已经规划 30+ 篇技术和宣发文章，文章不是 API 文档复制，而是面向公众号、博客和项目主页的完整内容。当前矩阵入口包括：

```text
docs/articles/zh-cn/publishing/technical-and-promo-article-matrix-30plus.md
docs/articles/zh-cn/publishing/article-roadmap-30plus.md
docs/articles/zh-cn/publishing/article-roadmap-30plus.json
artifacts/final-release/technical-article-publication-matrix.md
artifacts/final-release/technical-article-campaign-matrix.md
```

文章主题覆盖项目总览、ABI wrapper、deferred boundary、源码构建、NuGet/runtime package、OnnxToEngine、TensorRtExec、YoloVision、多模型案例、CUDA/TensorRT DLL 排查、package consumer proof、release evidence ladder 和 owner final action sequence。每篇文章都应该有受众、背景、命令、证据路径、配图建议、下一步和 proof boundary。

## Proof 边界

以下内容可以用于开发、排查或文章说明，但不能替代 package-consumer-runtime proof：

- build-only。
- dry-run。
- template。
- input draft。
- local feed。
- ProjectReference。
- direct `.nupkg`。
- GitHub Actions dry-run。
- dependency-probe-only。
- GUI screenshot。
- command preview。
- TensorRtExec report。
- OnnxToEngine report。
- YoloVision matrix。
- output JSON。
- SVG visualization。
- sidecar-only metadata。
- blocked-by-cuda-driver。

公开发布命令必须由 owner 在确认包、hash、外部 consumer proof、真实模型 proof、Linux runner proof、post-publish verification 和 release close gate 后手动执行。在这些真实证据缺失时，`canPublishPublicly=false`、`canCloseReleaseIssue=false` 和 `blocked-real-proof-required` 必须保持不变。

## 配图建议

- 项目架构图：C# wrapper -> generated interop -> native bridge -> CUDA/TensorRT/cuDNN。
- 证据梯度图：tutorial -> build report -> sample run -> real-model-runtime -> package-consumer-runtime -> post-publish verification。
- Runtime package 双路线图：GitHub Release managed + bridge assets 与 NuGet managed + bridge package route。
- 应用截图：TensorRtExec CLI 输出、WinForms 界面、YoloVision det/seg/pose/obb/cls/sem 结果。
- Deferred uplift 风险图：readonly/query/deployment API 与 callback/allocator/plugin/borrowed pointer 高风险 API。

## 下一步

后续工作会继续沿两条线推进：一边把 deferred 接口按低 ownership 风险逐批提升为真实只读 API，一边把 OnnxToEngine、TensorRtExec 和 YoloVision 做成可以支撑文章、样例和发布验证的完整体验。等 owner 回填真实模型资产、公开包、外部 consumer proof、Linux runner proof 和 post-publish verification 后，项目才能进入最终发布动作；在此之前，文章可以继续完善，但不能把 guidance 写成发布完成声明。
