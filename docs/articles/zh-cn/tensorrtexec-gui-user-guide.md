# TensorRtExec GUI 使用教程：在 Windows 桌面完成 ONNX 到 Engine 构建

命令行适合自动化，但不是所有用户都愿意从一长串参数开始。`applications/TensorRtExec` 提供了 WinForms 入口，把 ONNX、engine、TensorRT line、precision、shape profile、workspace 和报告路径放在一个桌面页面里。它不是玩具 GUI：页面背后调用同一个 `JYPPX.TensorRtSharp.Tools` 服务层，和 CLI 共用 parser/options/build/report 语义。

本文介绍 GUI 怎么用、每个字段对应什么命令参数、报告如何解读，以及哪些事情现在仍然不能宣称完成。

## 启动方式

从仓库根目录运行：

```powershell
dotnet run --project .\applications\TensorRtExec -- --ui
```

如果不带参数启动，应用默认打开 WinForms 页面：

```powershell
dotnet run --project .\applications\TensorRtExec
```

也可以直接走命令行：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model.plan `
  --buildOnly
```

## 页面字段

当前 WinForms 页面覆盖以下 build/report 参数：

| 页面字段 | 对应参数 | 说明 |
| --- | --- | --- |
| ONNX | `--onnx` | 外部 ONNX 模型路径 |
| Save Engine | `--saveEngine` | serialized engine 输出路径 |
| Load Engine | `--loadEngine` | 已有 engine 的 preflight 路径 |
| TensorRT | `--tensor-rt-line` | 8、10 或 11 |
| Precision | `--fp16 --int8 --bf16 --noTF32` | precision flag；INT8 calibrator 尚未完整实现 |
| Workspace MiB | `--workspace` | workspace memory pool limit |
| Min Shapes | `--minShapes` | dynamic shape profile 最小形状 |
| Opt Shapes | `--optShapes` | dynamic shape profile 最优形状 |
| Max Shapes | `--maxShapes` | dynamic shape profile 最大形状 |
| Plugins | `--plugins` | 当前记录诊断，不加载 library |
| Timing Cache | `--timingCacheFile` | 当前记录诊断，不导入/导出 cache |
| Profiling | `--profilingVerbosity` | 记录 profiling verbosity |
| Opt Level | `--builderOptimizationLevel` | 应用到 builder config |
| Aux Streams | `--maxAuxStreams` | 应用到 builder config |
| Device | `--device` | 记录部署目标设备，不切换进程当前 CUDA device |
| DLA Core | `--useDLACore` | 当前记录诊断，不做 layer device placement |
| Deployment | `--allowGPUFallback --directIO --stronglyTyped` | 当前记录诊断 |
| Tactics | `--tacticSources` | 当前记录诊断 |
| Mem Pools | `--memPoolSize` | 当前记录诊断；实际 workspace 仍由 `--workspace` 设置 |
| Input IO | `--inputIOFormats` | 当前记录诊断 |
| Output IO | `--outputIOFormats` | 当前记录诊断 |
| Calib Cache | `--calib` | 当前记录诊断，不启用 calibrator callback |
| Sparsity | `--sparsity` | 当前记录诊断 |
| Layer Info | `--exportLayerInfo` | 当前记录诊断，等待更完整 layer-info 支持 |
| Report | `--exportReport` | JSON 或 Markdown 报告 |
| Mode | `--buildOnly --skipInference --dryRun` | 外部模型推荐先 dry-run 预检，再 build-only |
| Preview | command preview | 展示等价命令行 |
| Run | 执行服务层 | 调用 `TensorRtExecService` |

## 推荐操作流程

1. 选择 `ONNX`。
2. 设置 `Save Engine`，例如 `models\model.plan`。
3. 选择 TensorRT line，通常与本机 runtime 对齐。
4. 如果模型有动态输入，填写 min/opt/max shapes：

```text
input:1x3x640x640
input:1x3x640x640
input:4x3x640x640
```

5. 选择 precision，例如 FP16。
6. 设置 workspace，例如 `512` MiB。
7. 选择 report 输出，例如 `models\model-build-report.json`。
8. 首次整理参数时勾选 `Dry run`，确认命令、shape profile、报告路径和部署参数都进入 preview/report。
9. 去掉 `Dry run`，保持 `Build only` 和 `Skip inference` 开启。
10. 点击 `Preview` 检查等价命令。
11. 点击 `Run`。

Preview 示例：

```text
--tensor-rt-line 10 --onnx .\models\model.onnx --saveEngine .\models\model.plan --workspace 512 --minShapes input:1x3x640x640 --optShapes input:1x3x640x640 --maxShapes input:4x3x640x640 --profilingVerbosity detailed --exportReport .\models\model-build-report.json --batch 2 --iterations 10 --warmUp 200 --duration 3 --streams 1 --builderOptimizationLevel 4 --maxAuxStreams 2 --memPoolSize workspace:512 --fp16 --buildOnly --skipInference --dryRun
```

## 为什么默认 Build only

GUI 面向用户，但不能替用户猜模型语义。外部 ONNX 的输入名、输出名、图像预处理、输出 layout、后处理和 NMS 规则都可能不同。默认 build-only/skip-inference 是为了让“engine 构建证据”和“真实模型推理证据”分开。

如果要证明真实推理，应进入对应 sample：

- 分类模型：`samples/Classification`
- YOLO-family：`samples/YoloVision`
- 自定义模型：编写基于 `TensorRtInferenceBindings` 的具体绑定代码

## 报告与日志

运行后，日志框会显示服务层输出。报告文件由 `OnnxEngineBuildDiagnostics` 写入。JSON 报告里最关键的字段是：

```json
"Parsed": true,
"EngineSaved": true,
"DryRun": false,
"InferenceRan": false,
"NormalizedCommandLine": "--tensor-rt-line 10 --workspace 512 --buildOnly --skipInference",
"NormalizedCommandSha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
"DeploymentOptions": {
  "BuilderOptimizationLevel": 4,
  "MaxAuxStreams": 2
},
"ProofClassification": "build-only",
"BuildEvidenceOnly": true,
"IsRealModelRuntimeProof": false,
"IsPackageConsumerRuntimeProof": false,
"IsRuntimeExecutionProof": false
```

如果 `InferenceRan=false` 或 `ProofClassification=build-only`，就不能把报告写成 runtime proof。它仍然是有价值的 build evidence，适合用于模型转换记录、发布前检查和问题排查。GUI 导出的报告不会声明 `package-consumer-runtime`，这个级别只属于 release proof record。

勾选 `Dry run` 时，报告会写出 `State=dry-run-precheck`、`DryRun=true`、`ProofClassification=precheck` 和 `BuildEvidenceOnly=true`。这只证明 GUI/CLI 共享参数模型可以解析并归一化命令，不会探测 TensorRT runtime，不会读取 ONNX，不会构建 engine，也不会解除 `blocked-by-cuda-driver` 这类发布阻塞。

## Plugin、Timing Cache、Layer Info

这些字段现在已经进入 GUI 和 command preview，但它们的语义分两层：

- `Opt Level`、`Aux Streams` 和 `Profiling`：会在 build 阶段应用到 builder config。
- `Plugins`：不主动加载 plugin library。
- `Timing Cache`：不导入/导出 cache。
- `DLA`、`Tactics`、`Mem Pools`、`Input IO`、`Output IO`、`Calib Cache`、`Sparsity`、`Strongly typed`：记录到 diagnostics 和 report，等待模型级 runtime 与 callback/lifecycle 阶段补足。
- `Layer Info`：记录 layer-info 请求，完整导出取决于后续 TensorRT runtime 支持。

这样设计是为了避免在 GUI 里提前开放 lifecycle 和 ownership 不清楚的功能。

## 常见问题

**Q：为什么选择 ONNX 后点击 Run 仍然失败？**
A：先看日志里的 `SkipReason` 或异常。常见原因是 CUDA/TensorRT runtime 不在探测路径、TensorRT line 不匹配、模型有 TensorRT 不支持的算子，或者 dynamic shape 没有完整提供。

**Q：可以直接拿 GUI 当官方 trtexec 替代品吗？**
A：当前目标是复刻模型转换/build 关键路径，不是完整 benchmark 工具。性能测试、DLA、calibration、plugin 加载等能力还需要分阶段补齐。

**Q：为什么 INT8 只是参数记录？**
A：INT8 真正可用需要 calibrator、calibration cache、量化策略和模型验证。当前 GUI 保留参数入口和 diagnostics，但不宣称完整 INT8 工作流。

**Q：报告里的 `IsRuntimeExecutionProof=false` 是失败吗？**
A：不是。它表示这次证据只到 build/sample 层，没有进行真实输出匹配。build-only 的外部模型报告应当如此。

## 结语

TensorRtExec GUI 的价值在于把复杂命令参数变成可见、可预览、可记录的构建工作流。它让 Windows 用户能更容易完成 ONNX 到 TensorRT engine 的第一步，同时仍然保留项目最重要的边界：构建成功不是推理成功，工具可用不是发布 proof，候选资产不是 smoke passed。

## 第三批正文门禁

### 适用读者

本文适合需要用 Windows 桌面页面完成 ONNX 到 engine 构建的用户，也适合负责 CLI/WinForms parity 的维护者。

### 解决问题

GUI 文章解决的是用户如何填写参数、如何生成报告、如何理解 build-only 和 dry-run，而不是替用户证明真实模型推理正确。

### 背景与场景

TensorRtExec 同时支持命令行和 WinForms。两条路径必须共享 options/service/report schema，避免 GUI 能配置但 CLI 无法复现，或 CLI 能解析但 GUI 状态落后。

### 代码与文件入口

- `applications/TensorRtExec/README.md`
- `applications/TensorRtExec/tensor-rt-exec-feature-matrix.json`
- `src/JYPPX.TensorRtSharp.Tools`
- `tests/JYPPX.ProjectQuality.Tests/TensorRtExecApplicationTests.cs`

### 操作路径

先用 dry-run 检查参数，再用 build-only 构建 engine/report，最后把真实模型 runtime proof 留给样例 runner、host metadata、hash 和 validator。

### 边界说明

build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 都不是 runtime proof。GUI 截图和报告也不是 public package proof 或 post-publish proof。

### 下一步

下一步继续补 GUI 控件截图、参数分组和真实模型案例，把用户教程润色成可直接发布的图文长文。
