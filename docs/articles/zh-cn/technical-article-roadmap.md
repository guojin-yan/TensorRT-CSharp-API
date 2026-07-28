# 技术文章矩阵规划

> 状态：规划稿
> 适用范围：项目宣传、发布说明、使用教程和案例教程。
> 重要边界：当前 package readiness 已清零，`real-callback-runtime-evidence-schema` 为 `schema-ready`，但 full package consumer runtime smoke 在本机被 CUDA driver/runtime compatibility 阻塞为 `blocked-by-cuda-driver`，真实 callback runtime proof 仍为 `false`。

## 规划原则

文章目标不是凑数量，而是把项目真实完成度、可用路径、部署方式、案例教程和风险边界讲清楚。每篇案例文章都必须能追溯到仓库中的样例、smoke、文档或 plan/diary 证据；凡涉及外部模型、标签或图片资产，都必须说明模型来源、授权注意事项、转换步骤、运行命令和验证输出。

在真实 callback runtime proof 完成前，所有宣传材料都必须保留以下边界：

- `manifest/source 100%` 不等于 `100% runtime 可用`。
- `readiness blockers: 0` 表示当前包完整性和消费端证据达标，不表示 callback runtime proof 完成。
- `SmokeResult=passed` 不能自动证明 callback 由 TensorRT runtime 触发。
- `blocked-by-cuda-driver` 是环境阻塞证据，不是 API 缺失，也不是 callback proof。
- `IGpuAllocator::*`、`IGpuAsyncAllocator::*`、`IOutputAllocator::*`、`IDebugListener::processDebugTensor` direct callback rows 在真实 proof 前必须继续 deferred。

## 样例与文章底座

| 仓库路径 | 当前状态 | 可支撑文章 |
| --- | --- | --- |
| `samples/MultiStream` | 可运行，已补 README | CUDA stream/event、跨 stream ordering、内存 copy 教程 |
| `samples/DynamicShape` | 可运行，已补 README | TensorRT dynamic shape、optimization profile、binding 教程 |
| `samples/InferenceBindings` | 可运行，已有 README | `TensorRtInferenceBindings` 输入输出、enqueue、readback 教程 |
| `samples/OnnxToEngine` | 可运行，已有 README，已补 trtexec-like 参数模型；构建服务已抽到 `src/JYPPX.TensorRtSharp.Tools` | ONNX parser、engine build、deserialize、round-trip、shape profile、precision/workspace 参数教程 |
| `samples/Classification` | 需要用户提供 ONNX/labels/image 资产 | 分类模型完整教程，需先选定可再分发模型和预处理说明 |
| `samples/YoloVision` | 需要用户提供 ONNX/labels/image 资产，已补 family/task/profile、layout、score filtering、class-aware/class-agnostic NMS、seg/pose/OBB/sem 托管辅助底座，并新增 YOLOX-S 候选资产示例 | YOLO 检测完整教程，需先选定 YOLO-family ONNX 和 COCO labels |
| `applications/TensorRtExec` | trtexec-like CLI + WinForms 工具，直接引用 `JYPPX.TensorRtSharp.Tools` 生成 build/report 证据，外部模型推理仍需显式 binding/output 语义 | ONNX 转 engine 工具教程、桌面部署工作流、参数排障 |
| `smoke/*` | 28 个验证 runner | 发布门禁、API 完成度、layer coverage、排障文章 |
| `artifacts/package-readiness/runtime-package-readiness-summary.md` | 当前 readiness evidence | package readiness、runtime smoke、blocked evidence 文章 |

## 文章矩阵

### 文章质量门禁字段

每篇进入发布排期的文章都必须能被审计，而不是只保留标题。后续新增、改名或拆分文章时，正文或矩阵必须明确覆盖以下字段：

| 字段 | 要求 |
| --- | --- |
| article id | 使用连续编号，不能重复，不能跳过当前矩阵主线。 |
| 标题 | 与正文文件、docs index、toc 和发布索引保持一致。 |
| 类型 | 标明项目总览、接口体系、样例教程、应用教程、发布证据、安全边界或宣发素材。 |
| 对应 sample/application | 明确链接 `samples/OnnxToEngine`、`samples/YoloVision`、`applications/TensorRtExec`、`smoke/*` 或无样例依赖。 |
| 模型/资产 | 写明是否需要 ONNX、labels、输入图片、sidecar、sample-run-evidence 或无外部资产。 |
| 模型获取方式 | 外部模型必须说明 owner 自备、公开下载、export 步骤或不能再分发的边界。 |
| license/hash 要求 | 外部资产必须记录 license notes、model SHA256、labels SHA256、image SHA256 和日志 SHA256。 |
| 命令 | 给出可复制命令；TensorRtExec/OnnxToEngine 文章必须区分 `--dryRun`、`--buildOnly`、真实 runner 和 validator 命令。 |
| 输出 | 标明预期 stdout/stderr、report 字段、evidence line、`Passed=True` 或 blocked 状态。 |
| 截图/图示需求 | GUI、文章配图或发布素材需要说明截图、报告截图或流程图需求；无视觉资产时写明无。 |
| proof boundary | 明确 `build-only`、`parse-only`、`sidecar-only`、`synthetic-input-runtime`、`real-model-runtime`、`package-consumer-runtime` 和 `post-publish verification` 的边界。 |
| 发布优先级 | 标明低资产依赖优先、外部资产回填、owner proof 或最终发布前审计。 |
| 完成状态 | 使用正文已起草、待扩写长文、规划稿、需补资产清单、需真实 proof 回填等可执行状态。 |

### 内容收口与 Proof 状态

文章正文完成和外部 proof 完成是两个独立维度；机器 ledger 分别使用 `contentState` 与 `proofState`：

- `完整教程已收口`：canonical 正文已有完整问题背景、代码/工件、可执行命令、输出、排障、proof boundary 和 checklist。
- `完整教程已由 <ID> 收口`：当前编号保留历史选题，但正文由后续 canonical 编号维护，避免重复文章继续漂移。
- owner、compatible host、real model、package consumer、Linux runner 或 post-publish 输入未到位时，即使文章完整，proof 状态仍是 blocked。
- template、draft、runbook、build-only、local feed、ProjectReference、direct nupkg 或文章长度都不能改变 release flag。

机器可读 closure ledger：

- `docs/articles/zh-cn/publishing/technical-article-closure-ledger.json`
- `docs/articles/zh-cn/publishing/technical-article-closure-ledger.md`
- exporter：`eng/Export-TechnicalArticleClosureLedger.ps1`
- 第一批基础文章审计：`docs/articles/zh-cn/publishing/technical-article-foundations-first-batch-audit.json`
- 第一批基础文章审计：`docs/articles/zh-cn/publishing/technical-article-foundations-first-batch-audit.md`
- 第一批审计 exporter：`eng/Export-TechnicalArticleFoundationsFirstBatchAudit.ps1`

矩阵中已废弃的旧检测专用样例名不得重新作为文章、样例或发布 proof 入口出现；统一入口是 `samples/YoloVision`。`applications/TensorRtExec` 与 `samples/OnnxToEngine` 可以产生 build/report/sidecar 证据，但不能替代 Classification/YoloVision 真实模型 runner，也不能替代 release proof record。

| 编号 | 系列 | 标题 | 主要内容 | 样例/证据 | 资产要求 | 状态 |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | 项目总览 | TensorRtSharp4.0 是什么 | 项目定位、C# 到 TensorRT/CUDA bridge、适用人群、当前边界 | `docs/articles/zh-cn/project-overview.md`、`docs/articles/zh-cn/blog-project-introduction.md` | 无 | 博客长文初稿已补 |
| 2 | 项目总览 | 为什么不是简单 P/Invoke | ABI 稳定、no-throw C ABI、对象生命周期、跨版本 guard | `docs/articles/zh-cn/why-not-plain-pinvoke.md` | 无 | 完整教程已收口 |
| 3 | 项目总览 | 从接口清零到 deferred 边界提升 | 解释 manifest/source 覆盖和真实 API 可用性的区别 | `docs/articles/zh-cn/interface-zero-to-deferred-boundary.md` | 无 | 完整教程已收口 |
| 4 | 项目总览 | TRT8/TRT10/TRT11 跨版本策略 | version guard、manifest、native 实现、托管路由一致性 | `docs/articles/zh-cn/trt-cross-version-strategy.md` | 无 | 完整教程已收口 |
| 5 | 安装部署 | Windows 本地开发环境准备 | .NET、CMake、CUDA、TensorRT、cuDNN、development probing | `docs/articles/zh-cn/windows-local-dev-environment.md` | 本机 NVIDIA SDK | 完整教程已收口 |
| 6 | 安装部署 | runtime package 和 split package 怎么选 | managed、bridge、cuda-cudnn、tensorrt、collection/full package | `docs/articles/zh-cn/runtime-package-selection.md` | 本地 nupkg | 完整教程已收口 |
| 7 | 安装部署 | NuGet 消费端验证全流程 | restore/build/native asset copy、consumer report、signed output | `docs/articles/zh-cn/nuget-package-consumer-validation-flow.md`、`docs/articles/zh-cn/blog-package-consumer-evidence-chain.md` | 本地包源 | 博客长文初稿已补 |
| 7.1 | 源码编译 | C++ 原生桥接源码编译总教程 | Visual Studio C++、CMake preset、TensorRT/CUDA/cuDNN roots、binding generator、native bridge、managed package consumer、GitHub full runtime 包、NuGet small core/bridge 包和非 proof 边界 | `docs/articles/zh-cn/tensorrtsharp-source-build-cpp-guide.md`、`docs/articles/zh-cn/source-build-windows-cpp-bridge.md`、`docs/articles/zh-cn/source-build-cmake-presets-and-bindings.md`、`docs/articles/zh-cn/nuget-github-dual-package-strategy.md` | 本机 NVIDIA SDK + C++ toolchain | 正文已起草 |
| 8 | 安装部署 | package readiness summary 怎么读 | `Overall=ready`、split/full、vendor blockers、runtime smoke | `docs/articles/zh-cn/readiness-summary-guide.md` | 无 | 正文已起草 |
| 9 | 安装部署 | CUDA error 35 与驱动兼容排查 | `blocked-by-cuda-driver`、driver/runtime mismatch、不是 API 缺口 | `docs/articles/zh-cn/cuda-error-35-troubleshooting.md` | 兼容/不兼容驱动对照 | 正文已起草 |
| 10 | 接口体系 | TensorRT Builder/Runtime/Engine 对象模型 | logger、builder、config、network、runtime、engine、context | `docs/articles/zh-cn/tensorrt-object-model.md`、`smoke/TensorRtSmokeRunner` | 无 | 完整教程已收口 |
| 11 | 接口体系 | ExecutionContext 与 inference binding | tensor address、shape inference、enqueue、readback | `docs/articles/zh-cn/inference-bindings-tutorial.md` | 无 | 正文已起草 |
| 12 | 接口体系 | Dynamic Shape 与 Optimization Profile | min/opt/max、profile index、runtime shape | `docs/articles/zh-cn/dynamic-shape-optimization-profile-tutorial.md` | 无 | 正文已起草 |
| 13 | 接口体系 | ONNX Parser 到 Serialized Engine | parser、profile、host memory、deserialize、round-trip | `docs/articles/zh-cn/onnx-parser-to-serialized-engine-tutorial.md` | 无 | 正文已起草 |
| 14 | 接口体系 | Plugin Inventory 只读 API | creator count、name/version/namespace、lookup、只读边界 | `docs/articles/zh-cn/plugin-inventory-readonly-api.md`、`smoke/PluginRegistryInventorySmokeRunner` | 无 | 正文已起草 |
| 15 | 接口体系 | Plugin Serialization Paths | runtime/plugin path 诊断、序列化部署路径 | `docs/articles/zh-cn/plugin-serialization-paths.md`、`smoke/PluginSerializationPathsSmokeRunner` | 无 | 完整教程已收口 |
| 16 | CUDA | CUDA memory wrapper 入门 | device/pinned/managed/pitched memory、copy、error map | `docs/articles/zh-cn/cuda-memory-wrapper.md`、`smoke/CudaSmokeRunner` | CUDA runtime | 完整教程已收口 |
| 17 | CUDA | CUDA stream/event 与跨 stream 同步 | non-blocking stream、event record/wait/synchronize | `docs/articles/zh-cn/cuda-stream-event-multistream-tutorial.md` | CUDA runtime | 正文已起草 |
| 18 | CUDA | CUDA Graph 当前能力与边界 | graph node、debug dot、event/memcpy nodes、kernel attrs | `docs/articles/zh-cn/cuda-graph-capabilities-boundary.md`、`smoke/CudaGraphSmokeRunner` | CUDA runtime | 正文已起草 |
| 19 | CUDA | CUDA memory range APIs | range attributes、advise、prefetch、accessed-by devices | `docs/articles/zh-cn/cuda-memory-range-apis.md`、`src/JYPPX.CudaSharp`、相关 tests | CUDA runtime | 完整教程已收口 |
| 20 | 案例教程 | 最小 identity network 推理 | 不依赖外部模型的端到端 inference | `docs/articles/zh-cn/inference-bindings-tutorial.md` | 无 | 正文已起草 |
| 21 | 案例教程 | Dynamic batch 推理教程 | batch 1..4、profile 校验、输出一致性 | `docs/articles/zh-cn/dynamic-shape-optimization-profile-tutorial.md` | 无 | 正文已起草 |
| 22 | 案例教程 | ONNX 转 TensorRT engine 教程 | 内置 identity ONNX、engine file round-trip | `docs/articles/zh-cn/onnx-parser-to-serialized-engine-tutorial.md` | 无 | 正文已起草 |
| 23 | 案例教程 | 分类模型部署教程 | 模型获取、labels、预处理、Top-K 输出 | `docs/articles/zh-cn/classification-real-asset-walkthrough.md`、`samples/Classification` | 用户自备分类 ONNX/labels/image | 正文已起草 |
| 24 | 案例教程 | ResNet/MobileNet 分类实战 | 选一个公开模型，说明下载、转换、运行、验证 | `docs/articles/zh-cn/classification-real-asset-walkthrough.md`、`samples/assets/classification-assets.template.json` | 用户自备模型 URL、labels、测试图 | 正文已起草 |
| 25 | 案例教程 | YOLO 检测部署教程 | YOLO ONNX、COCO labels、layout、confidence、NMS 边界 | `docs/articles/zh-cn/yolovision-detection-tutorial.md`、`samples/YoloVision` | 需 YOLO-family ONNX 与图片 | 完整教程已由 74 收口 |
| 26 | 案例教程 | YOLO 输出布局排查 | `[1,84,8400]` 与 `[1,8400,84]`、objectness、threshold | `docs/articles/zh-cn/yolovision-detection-tutorial.md`、`samples/YoloVision` | 需示例输出或模型 | 完整教程已由 74 收口 |
| 27 | 案例教程 | 多 stream 预处理管线雏形 | 使用 CUDA stream/event 支撑未来图像预处理 | `docs/articles/zh-cn/cuda-stream-event-multistream-tutorial.md` | CUDA runtime | 正文已起草 |
| 28 | 案例教程 | Refit weights 使用场景 | refitter inspector、权重更新、限制 | `docs/articles/zh-cn/refit-weights-guide.md`、`docs/articles/zh-cn/blog-refit-weights-guide.md`、`smoke/RefitWeightsSmokeRunner` | 无 | 完整教程已收口 |
| 29 | 高级主题 | TensorRT 11 modern layers | TRT11 专属 layer、metadata、guard | `docs/articles/zh-cn/trt11-modern-layers-guide.md`、`smoke/NetworkTrt11ModernLayersSmokeRunner` | TRT11 | 完整教程已收口 |
| 30 | 高级主题 | Network layer coverage 导览 | convolution、pooling、resize、slice、topk、quantize | `docs/articles/zh-cn/network-layer-coverage-guide.md`、`docs/articles/zh-cn/blog-network-layer-coverage-guide.md`、`smoke/Network*SmokeRunner` | 无 | 完整教程已收口 |
| 31 | 高级主题 | ErrorRecorder snapshot 与诊断 | runtime/builder/refitter error recorder copied snapshots | `docs/articles/zh-cn/error-recorder-snapshot-guide.md`、`docs/articles/zh-cn/error-recorder-diagnostics-design-gate.md`、tests | 无 | 完整教程已收口 |
| 32 | 高级主题 | Managed logger/profiler/progress monitor | managed callback safe controls 与非 proof 边界 | `docs/articles/zh-cn/managed-logger-profiler-progress-monitor.md`、`smoke/Managed*CallbackSmokeRunner` | 无 | 完整教程已收口 |
| 33 | 边界专题 | Allocator owner ledger safety gate | 为什么不直接开放 allocator callback，当前安全门禁 | `allocator-owner-ledger-safety-gate.md` | 无 | 可立即撰写 |
| 34 | 边界专题 | OutputAllocator 与 DebugListener 当前边界 | owner design、precheck、borrowed pointer 风险、real callback runtime proof | `docs/articles/zh-cn/callback-allocator-safety-bridge-roadmap.md`、`docs/articles/zh-cn/callback-allocator-listener-readonly-safety-gates.md` | 无 | 完整教程已由 79 收口 |
| 35 | 边界专题 | Real callback runtime proof 准入条件 | required markers、package consumer smoke、proof 语义 | `real-callback-runtime-evidence-schema.md` | 驱动兼容环境 | 可立即撰写规划，正文待 proof |
| 36 | 发布排障 | 常见问题排查总表 | DLL missing、PATH、application control、CUDA error 35、NuGet restore | `docs/articles/zh-cn/troubleshooting-index.md` | 无 | 正文已起草 |
| 37 | 发布证据 | Linux Runner Evidence 回填指南 | handoff/template、record validator、真实 Linux x64 runner proof 晋级条件 | `docs/articles/zh-cn/blog-linux-runner-evidence-guide.md` | Linux runner 回填 JSON | 博客长文初稿已补 |
| 38 | 样例博客 | Dynamic Shape 博客版 | dynamic batch、optimization profile、runtime shape、binding readiness | `docs/articles/zh-cn/blog-dynamic-shape-optimization-profile.md`、`samples/DynamicShape` | 无 | 完整教程已收口 |
| 39 | 样例博客 | InferenceBindings Identity Network 博客版 | tensor address binding、device buffer、readiness、readback | `docs/articles/zh-cn/blog-inference-bindings-identity-network.md`、`samples/InferenceBindings` | 无 | 完整教程已收口 |
| 40 | 样例博客 | ONNX Parser Engine RoundTrip 博客版 | 内置 ONNX、parser、serialized engine、deserialize、output match | `docs/articles/zh-cn/blog-onnx-parser-engine-roundtrip.md`、`samples/OnnxToEngine` | 无 | 完整教程已收口 |
| 41 | 样例博客 | MultiStream CUDA Stream/Event 博客版 | non-blocking stream、event record/wait、跨 stream ordering | `docs/articles/zh-cn/blog-multistream-cuda-stream-event.md`、`samples/MultiStream` | CUDA runtime | 完整教程已收口 |
| 42 | 接口博客 | Plugin Inventory 只读 API 博客版 | registry exists、creator metadata、lookup、无 borrowed pointer | `docs/articles/zh-cn/blog-plugin-inventory-readonly-api.md`、`smoke/PluginRegistryInventorySmokeRunner` | TensorRT runtime | 完整教程已收口 |
| 43 | CUDA 博客 | CUDA Memory Wrapper 博客版 | device/pinned/managed/pitched memory、copy/readback、owner 边界 | `docs/articles/zh-cn/blog-cuda-memory-wrapper.md`、`smoke/CudaSmokeRunner` | CUDA runtime | 完整教程已收口 |
| 44 | 高级博客 | Refit Weights 博客版 | refitter entries、set weights、refit engine、before/after evidence | `docs/articles/zh-cn/blog-refit-weights-guide.md`、`smoke/RefitWeightsSmokeRunner` | TensorRT runtime | 完整教程已收口 |
| 45 | 覆盖博客 | Network Layer Coverage 博客版 | Network smoke runner 家族、layer coverage 读法、模型精度边界 | `docs/articles/zh-cn/blog-network-layer-coverage-guide.md`、`smoke/Network*SmokeRunner` | TensorRT runtime | 完整教程已收口 |
| 46 | 应用教程 | TensorRtExec 工具入门 | CLI/WinForms 双入口、ONNX、engine、precision、shape profile、workspace、build-only 边界 | `docs/articles/zh-cn/tensorrtexec-tool-getting-started.md`、`applications/TensorRtExec`、`samples/OnnxToEngine`、`src/JYPPX.TensorRtSharp.Tools` | 用户自备 ONNX，可先用 build-only | 正文已起草 |
| 47 | 样例教程 | YOLO 全系列配置底座 | v5-v26 family、det/cls/seg/obb/pose/sem task、layout/objectness/NMS、资产清单 | `docs/articles/zh-cn/yolo-family-profile-and-postprocess-guide.md`、`samples/YoloVision`、`samples/assets/yolovision-assets.template.json` | 用户自备 YOLO ONNX/labels/image | 正文已起草 |
| 48 | 案例教程 | YoloVision 真实资产接入 | YOLOX-S 候选、ONNX export、hash、manifest、TensorRtExec build-only、YoloVision 运行日志 | `docs/articles/zh-cn/yolovision-real-asset-walkthrough.md`、`samples/assets/yolovision-yolox-s-example.json` | 用户自备 YOLOX 权重、labels、图片 | 正文已起草 |
| 49 | 应用教程 | TensorRtExec 外部 ONNX 构建报告 | report 字段、build evidence、runtime proof 边界、plugin/timing cache 诊断 | `docs/articles/zh-cn/tensorrtexec-external-onnx-build-report.md`、`applications/TensorRtExec`、`src/JYPPX.TensorRtSharp.Tools` | 用户自备 ONNX | 正文已起草 |
| 50 | 样例教程 | YoloVision 多输出 Metadata 指南 | seg/pose/obb 多输出 tensor role、metadata、asset manifest、真实 smoke 边界 | `docs/articles/zh-cn/yolovision-multi-output-metadata-guide.md`、`samples/YoloVision`、`samples/assets/yolovision-assets.template.json` | 用户自备 YOLO ONNX/labels/image | 正文已起草 |
| 51 | 应用教程 | ONNX 到 TensorRT Engine 转换指南 | OnnxToEngine、TensorRtExec、trtexec-like 参数、build-only report、runtime proof 边界 | `docs/articles/zh-cn/onnx-to-engine-trtexec-conversion-guide.md`、`samples/OnnxToEngine`、`applications/TensorRtExec` | 用户自备 ONNX，可先用内置 identity | 正文已起草 |
| 52 | 发布证据 | 真实模型 Owner 回填 Checklist | Classification/YoloVision/YOLOX-S 资产、hash、license、sidecar、sample run evidence、release bundle | `docs/articles/zh-cn/real-model-owner-backfill-checklist.md`、`samples/assets/README.md`、`eng/Test-SampleAssetManifest.ps1` | 用户或 release owner 自备真实资产 | 正文已起草 |
| 53 | 应用教程 | TensorRtExec GUI 使用教程 | WinForms 字段、command preview、report、plugin/timing cache 诊断边界 | `docs/articles/zh-cn/tensorrtexec-gui-user-guide.md`、`applications/TensorRtExec` | 用户自备 ONNX | 正文已起草 |
| 54 | 样例发布化 | 样例证据分层：precheck/build/runtime/proof | `precheck`、`build-only`、`synthetic-input-runtime`、`real-model-runtime`、`package-consumer-runtime` 的边界和晋级条件 | `docs/articles/zh-cn/sample-evidence-ladder.md`、`samples/README.md`、`applications/TensorRtExec/README.md`、`src/JYPPX.TensorRtSharp.Tools` | 无 | 正文已起草 |
| 55 | 样例发布化 | Classification 真实模型证据链 | 分类模型资产、labels、input image、sidecar、sample-run-evidence、Top-K 输出和 manifest audit | `samples/Classification`、`samples/assets/classification-assets.template.json`、`docs/articles/zh-cn/classification-real-asset-walkthrough.md` | 用户自备分类模型 | 规划稿 |
| 56 | 样例发布化 | YoloVision 真实模型证据链 | YOLO-family 资产、layout、后处理 metadata、TensorRtExec build-only、真实 `YoloVision Passed=True` 日志和证据回填 | `samples/YoloVision`、`samples/assets/yolovision-assets.template.json`、`docs/articles/zh-cn/yolovision-multi-output-metadata-guide.md` | 用户自备 YOLO 资产 | 规划稿 |
| 57 | 应用教程 | OnnxToEngine 与 TensorRtExec 如何分工 | identity round-trip 样例、外部 ONNX build/report 工具、何时需要 sample runner 补真实输出语义 | `docs/articles/zh-cn/onnxtoengine-and-tensorrtexec-boundary.md`、`samples/OnnxToEngine`、`applications/TensorRtExec`、`docs/articles/zh-cn/onnx-to-engine-trtexec-conversion-guide.md` | 可先无外部资产 | 正文已起草 |
| 58 | 发布证据 | 从工具报告到 release proof record | 为什么 build report、sidecar、manifest、runbook、collection bundle 都不能代替 `external-runtime-proof-record.json` | `docs/articles/zh-cn/tool-report-to-release-proof-record.md`、`docs/articles/zh-cn/external-runtime-proof-record.md`、`docs/articles/zh-cn/compatible-host-runtime-proof-collection-bundle.md` | 兼容主机 smoke 日志 | 正文已起草 |
| 59 | 排障专题 | 发布前 stale claim 自查 | 搜索 `package-consumer-runtime`、`blocked-by-cuda-driver`、`Passed=True`、`ready-needs-manual-approval` 等过度声明 | `docs/articles/zh-cn/stale-claim-prepublish-audit.md`、`eng/Test-StaleReleaseClaims.ps1`、`artifacts/final-release/stale-release-claims-audit.md` | 无 | 正文已起草 |
| 60 | 发布专题 | 完整项目发布前最后一公里 | API 完成度、deferred 边界、样例证据、runtime package、release owner approval、post-publish verification 的串联检查 | `docs/articles/zh-cn/publish-final-mile-checklist.md`、`docs/articles/zh-cn/release-publish-execution-checklist.md`、`docs/articles/zh-cn/post-publish-verification-record.md` | owner approval + proof record | 正文已起草 |
| 61 | 发布交接 | Release Owner Handoff 总入口 | owner action、acceptable proof、non-substitute examples、close preflight、evidence bundle、stale claim audit 的交接闭环 | `docs/articles/zh-cn/release-owner-handoff.md`、`artifacts/final-release/owner-action-required.md`、`docs/articles/zh-cn/release-close-preflight.md` | owner 授权与真实 proof | 正文已起草 |
| 62 | 发布交接 | Owner Action Required 执行清单 | owner authorization、external runtime proof、post publish verification、real-model-runtime、Linux runner evidence 的执行顺序 | `artifacts/final-release/owner-action-required.md`、`docs/articles/zh-cn/release-owner-handoff.md` | owner 在真实环境执行 | 正文已起草 |
| 63 | 宣发总览 | 面向博客的项目能力与边界总览 | TensorRT/CUDA bridge、C# wrapper、samples、applications、runtime packages、deferred boundary、release proof record | `docs/articles/zh-cn/project-release-story-and-boundaries.md`、`docs/articles/zh-cn/project-overview.md`、`docs/articles/zh-cn/blog-project-introduction.md` | 无 | 完整教程已由 81 收口 |
| 64 | 应用教程 | TensorRtExec 参数分层深挖 | implemented、parse/report-only、TrtexecAlignmentStatus=parse-only、OptionImplementationStatus、build-only 报告 | `docs/articles/zh-cn/tensorrtexec-option-layering-deep-dive.md`、`artifacts/user-acceptance/trtexec-option-coverage.md` | 用户自备 ONNX | 完整教程已由 72 收口 |
| 65 | 样例教程 | YoloVision 全任务系列文章合集 | det、cls、seg、obb、pose、sem 的 family/task/profile、metadata、真实资产和 sample-run-evidence 路径 | `docs/articles/zh-cn/yolovision-all-task-overview.md`、`docs/articles/zh-cn/yolovision-detection-tutorial.md`、`docs/articles/zh-cn/yolovision-segmentation-tutorial.md`、`docs/articles/zh-cn/yolovision-pose-tutorial.md`、`docs/articles/zh-cn/yolovision-obb-tutorial.md`、`docs/articles/zh-cn/yolovision-classification-semantic-tutorial.md` | 用户自备 YOLO-family ONNX | 完整系列已由 73-78 收口 |
| 66 | 证据教程 | package-consumer-runtime proof 实操 | clean consumer、no ProjectReference、runtime package key、nupkg SHA256、stdout/stderr、blocked-by-cuda-driver 边界 | `docs/articles/zh-cn/package-consumer-runtime-proof-playbook.md`、`docs/articles/zh-cn/external-runtime-proof-record.md` | 兼容 CUDA 主机 | 完整教程已由 69 收口 |
| 67 | 证据教程 | post publish verification proof 实操 | 真实渠道、package URL、下载后 hash、clean consumer scan、post-publish record validator | `docs/articles/zh-cn/post-publish-verification-proof-playbook.md`、`docs/articles/zh-cn/post-publish-verification-record.md`、`docs/articles/zh-cn/post-publish-clean-consumer-project-scan.md` | owner 完成真实发布后 | 完整教程已由 70 收口 |
| 68 | 安全边界 | callback 与 allocator 安全桥接路线 | owner ledger、borrowed pointer、nothrow callback、real callback runtime proof、deferred boundary | `docs/articles/zh-cn/callback-allocator-safety-bridge-roadmap.md`、`docs/articles/zh-cn/real-callback-runtime-evidence-schema.md`、`docs/articles/zh-cn/allocator-owner-ledger-safety-gate.md` | 真实 callback runtime proof | 完整教程已由 79 收口 |
| 69 | 证据教程 | Package Consumer Runtime Proof Playbook | clean consumer、runtime package key、nupkg SHA256、host metadata、stdout/stderr summary、真实 smoke log SHA256 | `docs/articles/zh-cn/package-consumer-runtime-proof-playbook.md`、`docs/articles/zh-cn/external-runtime-proof-record.md`、`artifacts/final-release/owner-action-required.md` | 兼容 CUDA/TensorRT 主机 | 完整教程已收口 |
| 70 | 证据教程 | Post Publish Verification Proof Playbook | 真实渠道 package identity、downloaded hash、clean consumer restore/build/smoke、post-publish validator | `docs/articles/zh-cn/post-publish-verification-proof-playbook.md`、`docs/articles/zh-cn/post-publish-verification-record.md`、`docs/articles/zh-cn/post-publish-clean-consumer-project-scan.md` | owner 完成真实渠道发布后 | 完整教程已收口 |
| 71 | 样例教程 | Real Model Evidence Backfill Playbook | Classification/YoloVision 模型、labels、input、license、hash、TensorRtExec sidecar、sample-run-evidence | `docs/articles/zh-cn/external-model-evidence-case-study.md`、`docs/articles/zh-cn/real-model-evidence-backfill-playbook.md`、`docs/articles/zh-cn/real-model-owner-backfill-checklist.md` | owner 提供真实模型资产 | 完整教程已由 80 收口 |
| 72 | 应用教程 | TensorRtExec 参数分层深挖 | implemented、parse/report-only、OptionImplementationStatus、TrtexecAlignmentStatus=parse-only、build-only/report 边界 | `docs/articles/zh-cn/tensorrtexec-option-layering-deep-dive.md`、`applications/TensorRtExec`、`artifacts/user-acceptance/trtexec-option-coverage.md` | 用户自备 ONNX | 完整教程已收口 |
| 73 | 样例教程 | YoloVision 全任务系列总览 | v5/v6/v7/v8/v9/v10/v11/v26、custom、det/cls/seg/obb/pose/sem、support matrix 与 real-model-runtime 边界 | `docs/articles/zh-cn/yolovision-all-task-overview.md`、`samples/YoloVision`、`docs/articles/zh-cn/real-model-evidence-backfill-playbook.md` | 用户自备 YOLO-family ONNX | 完整教程已收口 |
| 74 | 样例教程 | YoloVision Detection 教程 | detection layout、objectness、NMS、TensorRtExec build-only、sample-run-evidence | `docs/articles/zh-cn/yolovision-detection-tutorial.md`、`samples/YoloVision` | 用户自备检测模型 | 完整教程已收口 |
| 75 | 样例教程 | YoloVision Segmentation 教程 | mask prototype、coefficients、多输出 metadata、sidecar-only 与 real-model-runtime | `docs/articles/zh-cn/yolovision-segmentation-tutorial.md`、`samples/YoloVision` | 用户自备分割模型 | 完整教程已收口 |
| 76 | 样例教程 | YoloVision Pose 教程 | keypoint metadata、SourceIndex、output layout、坐标边界、sample-run-evidence | `docs/articles/zh-cn/yolovision-pose-tutorial.md`、`samples/YoloVision` | 用户自备姿态模型 | 完整教程已收口 |
| 77 | 样例教程 | YoloVision OBB 教程 | angle unit/range、SourceIndex、axis-aligned 与 rotated NMS 边界、真实 evidence | `docs/articles/zh-cn/yolovision-obb-tutorial.md`、`samples/YoloVision` | 用户自备 OBB 模型 | 完整教程已收口 |
| 78 | 样例教程 | YoloVision Classification 与 Semantic Segmentation 教程 | labels、Top-K、semantic output layout、argmax、真实资产需求 | `docs/articles/zh-cn/yolovision-classification-semantic-tutorial.md`、`samples/YoloVision` | 用户自备 cls/sem 模型 | 完整教程已收口 |
| 79 | 安全边界 | Callback 与 Allocator 安全桥接路线 | owner ledger、borrowed pointer、nothrow callback、real callback runtime proof、deferred boundary | `docs/articles/zh-cn/callback-allocator-safety-bridge-roadmap.md`、`docs/articles/zh-cn/real-callback-runtime-evidence-schema.md` | 真实 callback runtime proof | 完整教程已收口 |
| 80 | 证据教程 | 外部模型 Evidence 回填案例总览 | build-only、sidecar-only、sample-run-evidence、real-model-runtime、package-consumer-runtime、post publish verification | `docs/articles/zh-cn/external-model-evidence-case-study.md`、`docs/articles/zh-cn/real-model-evidence-backfill-playbook.md` | owner 提供真实模型和 proof | 完整教程已收口 |
| 81 | 宣发总览 | TensorRtSharp4.0 项目能力与发布边界 | 项目价值、接口覆盖、C# wrapper、samples、TensorRtExec、YoloVision、release proof 边界 | `docs/articles/zh-cn/project-release-story-and-boundaries.md`、`docs/articles/zh-cn/project-overview.md` | 无 | 完整教程已收口 |
| 82 | 发布教程 | Owner Release Execution Package | owner 执行顺序、manual publish placeholder、package-consumer-runtime、real-model-runtime、post-publish verification、blocked-by-cuda-driver | `docs/articles/zh-cn/owner-release-execution-package.md`、`artifacts/final-release/owner-release-execution-package.md`、`eng/Export-OwnerReleaseExecutionPackage.ps1` | owner 执行真实 proof 回填 | 正文已起草 |
| 83 | 发布教程 | Compatible Host Proof Backfill Package | compatible host、package-consumer-runtime、Linux runner、YoloVision/Classification real-model-runtime、post-publish verification | `docs/articles/zh-cn/compatible-host-proof-backfill-package.md`、`artifacts/final-release/compatible-host-proof-backfill-package.md`、`eng/Export-CompatibleHostProofBackfillPackage.ps1` | owner 在兼容主机执行 proof 回填 | 正文已起草 |
| 84 | 发布教程 | Real Model And Package Proof Input Package | package-consumer-runtime、Classification/YoloVision real-model-runtime、post-publish verification、ProjectReference/sidecar-only/blocked-by-cuda-driver 边界 | `docs/articles/zh-cn/real-model-and-package-proof-input-package.md`、`artifacts/final-release/real-model-and-package-proof-input-package.md`、`eng/Export-RealModelAndPackageProofInputPackage.ps1` | owner 填写真实模型与包消费 proof input | 正文已起草 |
| 85 | 发布教程 | Release Close Gap Dashboard | owner authorization、package-consumer-runtime、Linux runner、real-model-runtime、post-publish verification、non-substitute proof 边界 | `docs/articles/zh-cn/release-close-gap-dashboard.md`、`artifacts/final-release/release-close-gap-dashboard.md`、`eng/Export-ReleaseCloseGapDashboard.ps1` | owner 按剩余 blocker 执行真实 proof 回填 | 正文已起草 |
| 86 | 发布教程 | Compatible Host Proof Execution Pack | owner authorization、package-consumer-runtime、Linux runner、real-model-runtime、post-publish verification、validator commands、non-substitute proof 边界 | `docs/articles/zh-cn/compatible-host-proof-execution-pack.md`、`artifacts/final-release/compatible-host-proof-execution-pack.md`、`eng/Export-CompatibleHostProofExecutionPack.ps1` | owner 一站式执行真实 proof 回填 | 正文已起草 |
| 87 | 发布教程 | Release Candidate Final Evidence Freeze | freezeState、remaining blockers、validator commands、non-substitute proof、stale claim audit、release close 边界 | `docs/articles/zh-cn/release-candidate-final-evidence-freeze.md`、`artifacts/final-release/release-candidate-final-evidence-freeze.md`、`eng/Export-ReleaseCandidateFinalEvidenceFreeze.ps1` | owner 最终实跑前冻结证据链 | 正文已起草 |
| 88 | 发布审计 | 发布前最终审计地图 | final evidence freeze、release close preflight、stale claim audit、owner blocker、package-consumer-runtime、real-model-runtime、post-publish verification | `docs/articles/zh-cn/release-final-audit-map.md`、`artifacts/final-release/release-candidate-final-evidence-freeze.md`、`artifacts/final-release/release-close-preflight.md` | owner 按审计地图补齐真实 proof | 正文已起草 |
| 89 | 宣发素材 | 项目对外介绍与发布边界素材包 | 项目一句话介绍、TensorRtExec、YoloVision、deferred boundary、blocked-by-cuda-driver、package-consumer-runtime、post-publish verification | `docs/articles/zh-cn/release-public-story-pack.md`、`docs/articles/zh-cn/project-release-story-and-boundaries.md`、`docs/articles/zh-cn/project-overview.md` | 无 | 正文已起草 |
| 90 | Owner Backlog | Release Owner Proof Backlog | owner authorization、package-consumer-runtime、Linux runner、real-model-runtime、post-publish verification、validator commands | `docs/articles/zh-cn/release-owner-proof-backlog.md`、`docs/articles/zh-cn/release-owner-handoff.md`、`artifacts/final-release/owner-action-required.md` | owner 执行真实 proof 回填 | 正文已起草 |
| 91 | 发布边界 | Release Proof 不可替代清单 | helper/template/draft/runbook/collection package/input package/local feed/ProjectReference/build-only/parse-only/sidecar-only/blocked-by-cuda-driver 不可替代 proof | `docs/articles/zh-cn/release-proof-non-substitutes.md`、`eng/Test-StaleReleaseClaims.ps1`、`docs/articles/zh-cn/stale-claim-prepublish-audit.md` | 无 | 正文已起草 |
| 92 | 发布索引 | 发布文章索引与推荐发布顺序 | 项目定位、入门、样例、TensorRtExec、YoloVision、runtime package、release proof、callback/allocator/debug listener 边界 | `docs/articles/zh-cn/release-article-index-and-publishing-order.md`、`docs/articles/zh-cn/technical-article-roadmap.md` | 无 | 正文已起草 |
| 93 | README 门面 | README 前台入口检查清单 | README、README.zh-CN、docs index、toc、samples、applications、YoloVision、build-only、package-consumer-runtime 边界 | `docs/articles/zh-cn/release-readme-frontpage-checklist.md`、`README.md`、`README.zh-CN.md` | 无 | 正文已起草 |
| 94 | Owner 顺序 | Release Owner 最后一公里执行顺序 | owner authorization、package-consumer-runtime、real-model-runtime、Linux runner、post-publish verification、validator、blocked-by-cuda-driver | `docs/articles/zh-cn/release-final-owner-action-sequence.md`、`docs/articles/zh-cn/release-owner-proof-backlog.md` | owner 执行真实 proof 回填 | 正文已起草 |
| 95 | 最终总检 | README 前台与 Proof Boundary 最终审计 | README、README.zh-CN、docs index、toc、samples、applications、release owner docs、package-consumer-runtime、real-model-runtime、post-publish verification | `docs/articles/zh-cn/release-frontpage-and-proof-boundary-final-audit.md`、`README.md`、`README.zh-CN.md` | 无真实外部 proof 时执行最终一致性审计 | 正文已起草 |
| 96 | 发布候选总检 | Release Candidate 最终总检 | final evidence freeze、frontpage final audit、owner backlog、proof non-substitutes、technical roadmap、package-consumer-runtime、real-model-runtime、post-publish verification | `docs/articles/zh-cn/release-candidate-final-cross-check.md`、`docs/articles/zh-cn/release-candidate-final-evidence-freeze.md`、`docs/articles/zh-cn/release-frontpage-and-proof-boundary-final-audit.md` | 无真实外部 proof 时执行最终总检 | 正文已起草 |
| 97 | 文章矩阵 | Release Candidate 文章矩阵总结 | 项目定位、安装、样例、TensorRtExec、YoloVision、runtime package、release proof、callback/allocator/debug listener、package-consumer-runtime、real-model-runtime | `docs/articles/zh-cn/release-candidate-article-matrix-summary.md`、`docs/articles/zh-cn/release-article-index-and-publishing-order.md`、`README.md` | 无真实外部 proof 时执行文章矩阵收尾 | 正文已起草 |
| 98 | 发布总结 | Release Candidate 发布总结 | release candidate final state、README/frontpage、article matrix、proof boundary、owner blocker、package-consumer-runtime、real-model-runtime、post-publish verification | `docs/articles/zh-cn/release-candidate-publication-summary.md`、`README.md`、`README.zh-CN.md` | 无真实外部 proof 时执行发布候选总结 | 正文已起草 |
| 99 | Final hold | Release Candidate Final Hold 与 Owner 等待状态 | blocked-real-proof-required、owner authorization、package-consumer-runtime、Linux runner proof、real-model-runtime、post-publish verification、proof non-substitutes、stale claim audit | `docs/articles/zh-cn/release-candidate-final-hold-owner-waiting.md`、`README.md`、`README.zh-CN.md`、`eng/Test-StaleReleaseClaims.ps1` | 无真实外部 proof 时固定最终等待 owner 状态 | 正文已起草 |
| 100 | Owner 清单 | Release Owner Action Checklist Final Hold | owner authorization、package-consumer-runtime、Linux runner proof、real-model-runtime、post-publish verification、validator、ProjectReference、local feed、build-only、parse-only、sidecar-only、blocked-by-cuda-driver | `docs/articles/zh-cn/release-owner-action-checklist-final-hold.md`、`README.md`、`README.zh-CN.md`、`eng/Test-StaleReleaseClaims.ps1` | 无真实外部 proof 时给 owner 固定最终执行清单 | 正文已起草 |
| 101 | 最终巡检 | Release Hold Final Inspection | README front door、docs index/toc、technical roadmap、final evidence freeze、stale claim audit、owner checklist、YoloVision、proof validators、ProjectReference、local feed、build-only、parse-only、sidecar-only、blocked-by-cuda-driver | `docs/articles/zh-cn/release-hold-final-inspection.md`、`README.md`、`README.zh-CN.md`、`artifacts/final-release/release-candidate-final-evidence-freeze.json`、`artifacts/final-release/stale-release-claims-audit.json` | 无真实外部 proof 时执行 release hold 最终巡检 | 正文已起草 |
| 102 | Release close | Release Issue Close Record 最终关闭门禁 | owner final close decision、release-issue-close-record-validation、evidence bundle SHA256、release close preflight、post-publish verification、rollback plan、blocked-template-only | `docs/articles/zh-cn/release-final-owner-action-sequence.md`、`artifacts/final-release/release-issue-close-record-validation.json`、`eng/Test-ReleaseIssueCloseRecord.ps1` | 无真实 owner close record 时保持 canCloseReleaseIssue=false | 正文已起草 |
| 103 | CUDA 安全边界 | Stream Capture To Graph 的 owner-safe session | `cudaStreamBeginCaptureToGraph`、CUDA 12.3+ guard、stream/graph dispose 阻止、same-graph handle 验证、deferred history、ToGraph smoke | `docs/articles/zh-cn/cuda-stream-capture-to-graph-owner-safety.md`、`artifacts/interface-coverage/cuda-stream-capture-to-graph-candidate-audit.md`、`src/JYPPX.CudaSharp/CudaStreamCaptureToGraphSession.cs`、`smoke/CudaGraphSmokeRunner` | 兼容 CUDA 主机；文章与 smoke 不是 clean package-consumer proof | 完整教程已收口 |

## 样例发布化重点路线

下一阶段文章和样例发布化不再追求“又增加几个文档标题”，而是把用户可以实际复现的路径做扎实：

1. 低资产依赖路径先闭环：`DynamicShape`、`InferenceBindings`、`OnnxToEngine` 和 `TensorRtExec --dryRun/--buildOnly` 必须有清晰命令、报告字段和失败边界。
2. 外部资产路径分层推进：`Classification` 和 `YoloVision` 先写资产选择、license、hash、manifest 和 sidecar，再写真实 runner 日志回填。
3. 证据分级必须固定：build-only 是构建证据，real-model-runtime 是真实样例证据，package-consumer-runtime 只属于 release proof record。
4. 每次新增或改名文章时，同步检查 `docs/index.md`、`docs/toc.yml`、`samples/README.md` 和相关 quality tests。
5. 所有宣传材料都必须保留 `blocked-by-cuda-driver` 的真实语义：它是兼容主机待执行项，不是通过、不是真实 runtime proof。

## 案例正文准入清单

写具体案例正文前，应先满足对应准入项：

1. 仓库中存在 `samples/<CaseName>` 或明确指向 `smoke/<RunnerName>`。
2. README 写明用途、运行命令、参数、预期 evidence lines。
3. 如果依赖模型，文档必须列出模型来源、许可证注意事项、输入 tensor 名称、layout、shape、数据类型和预处理。
4. 如果依赖 labels 或图片，必须说明文件格式和可再分发要求。
5. 如果本机环境不能运行，文章必须写明 `blocked-by-cuda-driver`、`blocked-by-application-control` 或其它环境状态，不能写成 API 缺失或 proof。
6. 如果涉及 callback、allocator、debug listener 或 borrowed pointer，必须链接到真实 callback runtime evidence schema，并保持 proof=false 边界。

## 近期执行顺序

1. 先写项目总览、安装部署、package readiness、NuGet consumer、CUDA error 35 排查等低资产依赖文章。
2. 再写 `MultiStream`、`DynamicShape`、`InferenceBindings`、`OnnxToEngine` 四篇可立即验证的样例教程。
3. 然后为 `Classification` 和 `YoloVision` 选定可再分发模型、labels 和图片资产，先补充样例 README，再写模型案例正文。
4. 最后写 callback 边界专题，避免用户误以为 deferred callback 已完成真实 runtime proof。
