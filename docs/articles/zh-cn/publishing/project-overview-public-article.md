# TensorRtSharp4.0 项目总览：把 TensorRT 带到可发布的 C# 工程实践里

很多 C# 开发者第一次接触 TensorRT 时，都会遇到同一个问题：官方生态最成熟的路径集中在 C++ 和 Python，而真正落到 .NET 桌面程序、服务端推理、内部工具链时，往往要在 P/Invoke、CUDA DLL、TensorRT ABI、模型转换和 NuGet 分发之间来回补洞。

TensorRtSharp4.0 想解决的不是“把几个函数声明搬到 C#”这么简单的事。它的目标是提供一套面向工程交付的 TensorRT / CUDA C# API：底层有清晰的 native bridge 和跨版本 guard，上层有更符合 .NET 使用习惯的 wrapper，中间还要有 smoke、quality gate、package consumer proof 和样例应用来证明这些能力真的能被用户拿走使用。

## 这个项目适合谁

- 正在用 C#/.NET 做 AI 推理工具、桌面应用、工业视觉或服务端部署的开发者。
- 想把 ONNX 模型转换为 TensorRT engine，但不希望每次都切到 C++ 工具链的工程团队。
- 需要同时维护 CUDA、TensorRT 8/10/11、多 runtime 包和 NuGet 分发的库维护者。
- 想理解 TensorRT C# binding 如何处理 ABI、ownership、deferred API 与真实 proof 的技术读者。

## 项目现在包含什么

项目按四条线收口。

第一条是底层 API。仓库中维护 TensorRT 8、TensorRT 10、TensorRT 11 以及 CUDA runtime 的 manifest、native bridge、generated interop 和托管封装。当前主线已经从 “missing 接口清零” 切换到 “deferred 边界提升”：也就是说，入口存在并不等于接口真正可用，真实完成度要看 native 是否非 deferred、高层 wrapper 是否覆盖、测试和 smoke 是否能证明调用路径。

第二条是样例与工具。`samples/OnnxToEngine` 面向模型转换，目标是尽量贴近官方 `trtexec` 的转换能力；`samples/YoloVision` 替代早期过窄的检测样例命名，目标是统一 YOLOv5、v6、v7、v8、v9、v10、v11、v26 以及 detection、classification、segmentation、OBB、pose、semantic segmentation 等任务；`applications/TensorRtExec` 则是更完整的 trtexec-like 应用，既要支持命令行，也要支持 Windows 图形界面。

第三条是验证体系。项目不把 build-only、template、local feed、ProjectReference 或 direct `.nupkg` 当成发布 proof。真实 package-consumer-runtime proof 必须来自干净外部 consumer 使用公开包的执行结果；真实模型 proof 必须由 owner 回填模型、输入、日志、SHA256 和运行摘要。

第四条是文章与发布材料。仓库已经规划 30+ 篇技术和宣发文章，覆盖项目介绍、接口边界、NuGet 安装、TensorRtExec、YoloVision、多模型案例、排障和 release proof。文章不是 API 文档的复制，而是面向微信公众号、博客和项目主页的完整内容。

## 为什么不能只做 P/Invoke

TensorRT 的 C++ API 不是为跨语言裸调用设计的。对象生命周期、borrowed pointer、callback、插件 registry、builder config、engine/context、error recorder 等边界如果处理不好，会出现三类问题：

- ABI 层面：不同 TensorRT 主版本的函数、类型和行为并不完全一致。
- 生命周期层面：C# 如果直接暴露无语义 `IntPtr`，用户很容易持有悬空对象。
- 诊断层面：调用失败后如果没有状态码、日志、copied metadata 和 smoke 证据，问题很难定位。

所以 TensorRtSharp4.0 使用 native bridge 把跨 ABI 异常隔离在边界内，用 manifest 和 generated binding 保证入口一致，再在 C# 层提供 `TensorRtBuilder`、`TensorRtRuntime`、`TensorRtEngine`、`TensorRtExecutionContext`、`TensorRtPluginRegistryInventory` 等更安全的封装。

## 一个典型使用路径

用户最常见的路径可以分成三步：

```powershell
# 1. 准备 ONNX 模型，并用 TensorRtExec 或 OnnxToEngine 构建 engine
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model.plan `
  --fp16 `
  --buildOnly `
  --exportReport .\artifacts\model-build-report.json

# 2. 用样例或自己的程序加载模型/engine
dotnet run --project .\samples\YoloVision -- `
  --model .\models\yolov8-det.onnx `
  --labels .\models\coco.names `
  --input-data .\models\yolov8-det-fp32.bin `
  --input-shape 1x3x640x640 `
  --family v8 `
  --task det

# 3. 若要作为 release proof，使用 owner validator 或外部 consumer validator 记录证据
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionRealAssetCandidate.ps1 -Strict
```

这三步的证据级别不同：第一步证明 build/report 路径；第二步证明某个样例和模型资产可运行；第三步才进入可审计 proof。发布前还需要干净外部 consumer 验证公开 NuGet 包。

## 当前边界

项目已经有大量接口、样例、文档和质量门禁，但仍然明确保留一些边界：

- deferred API 不等于真实可用 API。
- callback trampoline、plugin instance create/enqueue、resource acquire/release、borrowed pointer 暴露仍需谨慎推进。
- YoloVision 的 template JSON 不是 runtime proof。
- TensorRtExec 的 build-only/report 不是 package-consumer-runtime proof。
- 公开发布命令必须由 owner 在确认包、hash、外部 consumer proof 后手动执行。

这些边界不是缺陷，而是为了避免把“能编译”和“能发布”混在一起。

## 配图建议

- 项目架构图：C# wrapper -> interop -> native bridge -> CUDA/TensorRT。
- 证据梯度图：tutorial -> build report -> sample run -> real-model-runtime -> package-consumer-runtime。
- 应用截图：TensorRtExec CLI 输出、WinForms 界面、YoloVision 检测/分割结果。

## 下一步

后续工作会继续沿两条线推进：一边把 deferred 接口按低 ownership 风险逐批提升为真实只读 API，一边把 YoloVision 和 TensorRtExec 做成可以支撑文章、样例和发布验证的完整体验。等 owner 回填真实模型资产与外部 consumer proof 后，项目就可以进入更接近公开发布的状态。
