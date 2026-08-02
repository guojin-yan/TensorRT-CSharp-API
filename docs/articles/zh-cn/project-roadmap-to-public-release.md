# TensorRtSharp 4 走向公开发布的路线图

## 适用读者

本文面向项目维护者、试用用户和发布负责人，用一条路线图说明 TensorRtSharp 4 从接口追平、deferred 边界提升、文档补强、package consumer proof 到公开发布还差哪些关键步骤。

## 解决问题

接口 manifest/source 100% 匹配并不等于项目可发布。本文把剩余工作拆成 API 可用性、wrapper 生命周期、样例真实资产、runtime package、clean consumer proof、post-publish verification 和 owner approval，明确 build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 不是 runtime proof。

## 背景与场景

当前项目已经完成大量接口覆盖、文档、样例和发布门禁，但最终公开发布仍需要外部真实输入：干净 consumer restore/build/run、兼容 CUDA host runtime proof、公开包 hash 核对和发布后验证。路线图的目标是让后续每一轮工作都减少真实 blocker，而不是重复生成不可晋级材料。

## 目标变化与当前主线

当前主线已经明确从“继续堆 release proof 文档”切回“真实 deferred 接口提升 + 可发布项目完整度”。后续每轮工作必须优先减少真实缺口：

1. 将 TensorRT/CUDA deferred 接口从 no-arg placeholder 推进到有真实参数、返回结构、native source、C# interop、高层 wrapper、XML 注释和 quality/smoke 证据的可调用 API。
2. 对只读、查询型、部署关键 API 优先推进；对 callback、allocator、borrowed pointer、plugin instance、external resource、StreamReader/Writer callback、DimensionExpr owner 不清楚的路径继续保持 deferred 或 design gate。
3. `manifest/source 100%` 只能说明接口被追踪，不能声明 `100% runtime 可用`；真实完成度以非 deferred 实现、高层 C# wrapper、smoke/package-consumer 验证为准。
4. 不再把 TensorRtExec report、OnnxToEngine report、YoloVision matrix、readonly diagnostics、build-only、dry-run、template、local feed、ProjectReference 或 direct `.nupkg` 当成可晋级材料；它们只能辅助定位，不能替代真实外部执行证据。

## 操作路径

1. 持续提升 deferred readonly API，优先选择无 ownership 风险的高价值接口。
2. 完成高层 C# wrapper lifetime 收口，避免 public API 暴露裸 borrowed pointer。
3. 为 YoloVision 和 OnnxToEngine 回填真实模型资产记录、hash、输出 schema 和 validator。
4. 为 runtime packages 补齐 Windows/Linux clean consumer 安装、加载和运行 proof。
5. 在 owner 授权后执行公开发布，并用 post-publish clean consumer 记录回填 close record。

## 源码编译教程路线

宣传文章和用户文档必须补上“用户自己编译 native bridge”的完整路径，尤其是 C++/CMake 部分。该路线不是简单列命令，而是要能让用户复现：

- 环境需求：Windows Visual Studio C++ toolchain、CMake、.NET SDK、PowerShell、匹配的 CUDA / TensorRT / cuDNN roots，以及必要的 PATH / INCLUDE / LIB 配置。
- 版本矩阵：至少解释 TensorRT 8/10/11 与 CUDA 11/12/13、cuDNN 组合的区别，以及 `win-x64-trt11-cuda13-release`、TRT8 兼容 preset 等典型 preset 的适用场景。
- 本地配置：说明 `pack/runtime/runtime-packages.local.json`、`JYPPX_TENSORRT_ROOT`、`JYPPX_CUDA_ROOT`、`JYPPX_NATIVE_BRIDGE_PATH` 的用途。
- 必跑命令：`eng/Generate-Bindings.ps1`、`eng/Test-BindingGeneratorOutputs.ps1`、`eng/Export-InterfaceCoverageMatrix.ps1`、`cmake --preset ...`、`cmake --build --preset ... --parallel`、`dotnet build`、定向 `dotnet test`。
- 排障：覆盖 CUDA driver/runtime mismatch、TensorRT/cuDNN DLL 未找到、CMake preset roots 不匹配、bridge load failure、blocked-by-cuda-driver 的真实含义。

## 双发布渠道

包发布策略按两个公开渠道并行维护；两个渠道都禁止捆绑 NVIDIA 原厂运行库：

| 路线 | 分发位置 | 包内容 | 用户前置条件 | 用途 |
| --- | --- | --- | --- | --- |
| GitHub Release | GitHub Releases | managed API、YoloVision、按版本编译的项目自有 C++ bridge、源码与文档 | 用户按 runtime key 自行安装 CUDA / TensorRT / cuDNN，并配置 native library 搜索路径 | 固定版本资产、SHA256 校验、源码归档 |
| NuGet | nuget.org | C# 核心 API、YoloVision 与按版本拆分的项目自有 C++ bridge | 用户自行安装 CUDA / TensorRT / cuDNN，并配置本机 native library 搜索路径 | 标准 `PackageReference` 消费、公开生态采用 |

两个渠道都只发布项目源码编译得到的 managed/bridge 包；TensorRT、CUDA、cuDNN、NVRTC、parser、plugin 和 builder-resource 均不进入发布包。发布关闭仍需真实 clean consumer、runtime proof 和 post-publish verification。

## 样例与应用发布化路线

1. `samples/YoloVision` 是统一 YOLO-family 样例，旧 YOLO detection 样例名不再作为 live sample 回流。目标覆盖 YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom，以及 det、cls、seg、obb、pose、sem。每篇案例文章都要给出模型获取、许可证提示、ONNX 导出/转换、metadata、运行命令、输出 schema 和 evidence 回填方法。
2. `samples/OnnxToEngine` 要继续对齐官方 `trtexec` 的模型转换能力，明确 implemented / parse-only / report-only / owner-input-required 的参数层级。
3. `applications/TensorRtExec` 是 trtexec-like 应用层入口，必须同时覆盖 CLI 和 WinForms GUI；功能矩阵、参数分层、report schema、GUI/CLI 字段一致性都应进入质量门禁。
4. 宣传文章不少于 30 篇，但质量优先。文章面向微信公众号、博客和用户采用，不只是 API 文档；必须有完整开头、场景、步骤、代码位置、命令、预期输出、风险边界和下一步。

## 代码与文件入口

- `plan`：阶段目标和下一步计划。
- `diary`：阶段开发日记和下一阶段提示词。
- `TensorRtSharp4.0/artifacts/interface-coverage/project-completion-review.md`：完成度复审。
- `TensorRtSharp4.0/docs/articles/zh-cn/package-consumer-runtime-proof-playbook.md`：runtime proof playbook。
- `TensorRtSharp4.0/docs/articles/zh-cn/release-evidence-non-substitute-guide.md`：非替代项指南。

## 图示建议

建议用里程碑时间线：`接口追平 -> deferred 提升 -> wrapper 硬化 -> real asset proof -> clean consumer proof -> owner publish -> post-publish verification`。每个节点列出输入、输出和 blocker。

## 边界说明

路线图是执行规划，不是发布授权。TensorRtExec report、OnnxToEngine report、YoloVision matrix、readonly diagnostics、build-only、dry-run、template、local feed、ProjectReference 和 direct `.nupkg` 只能支持分析，不能替代真实 runtime / public package / post-publish 证据链。

## 下一步

下一轮应从路线图中选择最高价值 blocker：优先筛选一批 8 到 15 个低 ownership 风险的 deferred 接口，补齐 native/manifest/C# wrapper/quality gate；并同步补源码编译教程、双发布路线质量门、YoloVision/OnnxToEngine/TensorRtExec 的发布化缺口。没有真实 owner proof 输入时，不要继续扩 proof dashboard；只允许推进真实 API、样例、应用、包和文章质量。
