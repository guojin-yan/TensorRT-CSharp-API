# ONNX 到 TensorRT Engine 转换指南：从 OnnxToEngine 到 TensorRtExec

TensorRT 部署的第一道门槛通常不是 C# API，而是把 ONNX 模型稳定转换成 serialized engine。官方 `trtexec` 是最常用的命令行工具；TensorRtSharp4.0 里的 `samples/OnnxToEngine` 和 `applications/TensorRtExec` 则把同一类 build 工作流带到 .NET 项目中，方便用户在 C# 工程、自动化脚本和 Windows GUI 之间复用一套参数语义。

本文从一个实际转换流程讲起：先用 `OnnxToEngine` 理解最小构建链路，再用 `TensorRtExec` 处理外部 ONNX、shape profile、precision、workspace、报告和 GUI。

## 两个入口的分工

`samples/OnnxToEngine` 是最小样例。它能生成一个内置 dynamic identity ONNX，用来证明 parser、optimization profile、serialized engine、runtime deserialize、binding 和 output readback 在兼容环境下可以跑通。

`applications/TensorRtExec` 是面向用户的应用。它有命令行和 WinForms 双入口，直接复用 `src/JYPPX.TensorRtSharp.Tools` 中的：

- `TrtexecLikeParser`
- `TrtexecLikeOptions`
- `OnnxEngineBuildOptions`
- `OnnxEngineBuildService`
- `OnnxEngineBuildDiagnostics`

这样做的好处是：CLI、GUI 和 sample 不需要各自维护一套影子参数模型。

## 内置模型快速验证

先跑不依赖外部资产的样例：

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"

dotnet run --project .\samples\OnnxToEngine -- `
  --tensor-rt-line 10 `
  --batch 2
```

理想证据包括：

```text
Parsed=True
ProfileIndex=0
EngineFileRoundTrip=True
BindingReport Ready=True
OutputMatch=True
OnnxToEngine Passed=True
```

这条路径证明的是项目自己的最小 ONNX round-trip。它不代表任意外部模型都能推理，因为外部模型的 input/output name、dynamic shape、plugin、后处理都可能不同。

## 外部 ONNX dry-run 预检

在真正触发 TensorRT runtime probe、ONNX parser 和 engine build 之前，可以先用 dry-run 检查命令是否能被工具层解析、归一化并输出报告：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --save-engine .\models\model.plan `
  --minShapes input:1x3x640x640 `
  --optShapes input:1x3x640x640 `
  --maxShapes input:4x3x640x640 `
  --builderOptimizationLevel 4 `
  --maxAuxStreams 2 `
  --exportReport .\models\model-precheck-report.json `
  --previewOnly
```

`--dryRun` 和 `--previewOnly` 等价。它允许 ONNX 或 engine 路径暂时不存在，输出 `State=dry-run-precheck`、`ProofClassification=precheck`、`BuildEvidenceOnly=true` 和 `DryRun=true`，并生成 `NormalizedCommandLine` 与 `NormalizedCommandSha256`。这条路径不会探测 CUDA/TensorRT runtime，不会解析 ONNX，不会构建 engine，也不会执行推理；它只适合作为命令预检和交接证据。

## 外部 ONNX build-only

对真实模型，建议先做 build-only：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --save-engine .\models\model.plan `
  --minShapes input:1x3x640x640 `
  --optShapes input:1x3x640x640 `
  --maxShapes input:4x3x640x640 `
  --fp16 `
  --workspace 1GiB `
  --builderOptimizationLevel 4 `
  --maxAuxStreams 2 `
  --memPoolSize workspace:512MiB,tacticDram:1GiB `
  --timingCache .\models\model.cache `
  --profilingVerbosity detailed `
  --iterations 10 `
  --warmUp 200 `
  --duration 3 `
  --streams 1 `
  --useCudaGraph `
  --evidenceSidecar .\models\model-evidence.sidecar.json `
  --exportReport .\models\model-build-report.json `
  --buildOnly `
  --skipInference
```

这对应官方 `trtexec` 的常见 build 参数思路：

| 官方 trtexec 概念 | TensorRtSharp 工具参数 | 当前语义 |
| --- | --- | --- |
| ONNX 输入 | `--onnx` | 外部 ONNX 路径 |
| 保存 engine | `--saveEngine` / `--save-engine` / `--engine` | 写出 serialized engine |
| 加载 engine | `--loadEngine` / `--load-engine` | load-engine preflight + bounded runtime output |
| Dry run | `--dryRun --previewOnly` | 只做参数解析、命令归一化和报告，不触发 runtime/build/inference |
| FP16 | `--fp16` | 设置 builder flag |
| BF16 | `--bf16` | 设置 builder flag |
| INT8 | `--int8` | 参数记录；calibrator 尚未完整实现 |
| TF32 | `--noTF32` | 关闭 TF32，否则默认开启 |
| Workspace | `--workspace <MiB/GiB/MB/GB/KiB/KB/B>` | workspace memory pool limit；无后缀默认 MiB |
| Builder optimization | `--builderOptimizationLevel <0..5>` | 应用到 builder config |
| Aux streams | `--maxAuxStreams <n>` | 应用到 builder config |
| Dynamic shapes | `--minShapes --optShapes --maxShapes` | optimization profile |
| Timing cache | `--timingCacheFile` / `--timingCache` | 当前记录诊断，不导入/导出 cache |
| Plugin | `--plugins` / `--plugin` / `--dynamicPlugins` / `--setPluginsToSerialize` | 当前记录诊断，支持重复参数和逗号/分号列表，不加载 plugin library |
| DLA | `--useDLACore --allowGPUFallback` | 当前记录诊断，不做 layer device placement |
| Tactic sources | `--tacticSources` | 当前记录诊断 |
| Memory pools | `--memPoolSize workspace:512MiB,tacticDram:1GiB` | 当前记录诊断；每项必须换算为整 MiB；实际 workspace 仍由 `--workspace` 设置 |
| IO formats | `--inputIOFormats --outputIOFormats --directIO` | 当前记录诊断，真实绑定语义由具体 sample 负责 |
| Calibration cache | `--calib` | 当前记录诊断，不启用 calibrator callback |
| Sparsity/strong type | `--sparsity --stronglyTyped` | 当前记录诊断 |
| Profiling verbosity | `--profilingVerbosity` / `--verbose` | 归一化为 `none` / `layer_names_only` / `detailed` |
| Layer info | `--dumpLayerInfo --exportLayerInfo` | 记录到诊断，等待更完整 runtime 支持 |
| Runtime timing | `--iterations --warmUp --duration --streams --useCudaGraph` | 进入报告和 GUI 参数预览；CUDA graph 仍是边界诊断 |
| Runtime/output artifacts | `--loadInputs --dumpOutput --dumpRawBindingsToFile --exportOutput --exportTimes --exportProfile --saveProfile` | build-only 只写边界占位；synthetic runtime 可写最小输出证据 |
| Report | `--exportReport` | 输出 JSON 或 Markdown 报告 |

## 参数归一化与 GUI 对齐

`samples/OnnxToEngine`、`applications/TensorRtExec` CLI 和 WinForms 现在都复用同一个 trtexec-like 参数模型。官方风格的 `--save-engine`、`--load-engine`、`--timingCache`、`--verbose` 会在报告中归一化为项目内部稳定参数名；memory 参数支持显式单位，归一化命令里仍以 MiB 记录，便于测试和 evidence diff。

GUI 中的 Runs/Warm/Duration/Streams、Runtime Flags、Load Inputs、Raw Bindings、Output JSON、Times/Profile、Save Profile 会进入同一条 `NormalizedCommandLine`。因此 CLI 与 GUI 的差异应只体现在用户输入方式，不应体现在 build/report 语义。

## 为什么外部模型默认不做通用推理

一个 ONNX 能 build 成 engine，不等于项目知道如何给它准备输入、绑定输出、解释结果。分类模型、YOLO 检测、segmentation、pose、语义分割的输出语义都不同。TensorRtSharp4.0 当前把外部 ONNX 默认视为 build/report 证据，除非具体 sample 提供了绑定和后处理。

这个边界很重要：

- `Parsed=True` 说明 parser/build 通过。
- `EngineSaved=True` 说明 engine 文件写出。
- `InferenceRan=False` 说明没有做真实推理。
- `IsRuntimeExecutionProof=False` 说明这不是 runtime proof。
- `ProofClassification=build-only` 说明它只能作为模型转换或构建证据。

## 报告文件怎么读

使用：

```powershell
--exportReport .\models\model-build-report.json
```

JSON 报告会包含：

```json
{
  "Success": true,
  "State": "external-onnx-build-only",
  "Parsed": true,
  "EngineSaved": true,
  "InferenceRan": false,
  "OutputMatch": false,
  "DryRun": false,
  "NormalizedCommandLine": "--tensor-rt-line 10 --workspace 512 --buildOnly --skipInference",
  "NormalizedCommandSha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "DeploymentOptions": {
    "BuilderOptimizationLevel": 4,
    "MaxAuxStreams": 2
  },
  "ProofClassification": "build-only",
  "EvidenceClassifications": [
    "build-only",
    "dependency-probe-only",
    "precheck",
    "synthetic-input-runtime",
    "real-model-runtime",
    "package-consumer-runtime"
  ],
  "BuildEvidenceOnly": true,
  "IsRealModelRuntimeProof": false,
  "IsPackageConsumerRuntimeProof": false,
  "StdoutSummary": "Parsed=True ...",
  "StderrSummary": "",
  "IsRuntimeExecutionProof": false
}
```

Markdown 报告会写明：

```text
This report is build/sample evidence only. build-only and dependency-probe-only are not runtime proof; synthetic-input-runtime is not real model proof; package-consumer-runtime is tracked by release proof records.
```

这句话不是“保守措辞”，而是为了防止把构建证据、runtime proof、发布 proof 混为一谈。

`NormalizedCommandLine` 用来记录工具层归一化后的参数，`NormalizedCommandSha256` 用来给这条归一化命令生成稳定摘要，方便 issue、日志和 release evidence 对账。`DryRun=true` 与 `ProofClassification=precheck` 表示这份报告只完成了预检，不包含 runtime probe、ONNX parse、engine build 或 inference。`DeploymentOptions` 用来保存 trtexec-like 部署参数快照。当前 `builderOptimizationLevel`、`maxAuxStreams` 和 `profilingVerbosity` 会进入 builder config；DLA、tactic source、IO format、calibration cache、sparsity、strongly typed 等参数先作为诊断记录进入 report，等待模型级 runtime 阶段补足生命周期与输出语义。`BuildEvidenceOnly=true` 和 `ProofClassification=build-only` 表示这份报告仍然只是构建证据，不是推理输出正确性的证明。`real-model-runtime` 需要具体 sample 的真实输入、labels、hash 和输出日志；`package-consumer-runtime` 只由 release proof record 记录。

如果要把 build report 和后续真实样例证据串起来，先生成 sidecar 模板：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OnnxEngineBuildEvidenceSidecarTemplate.ps1
```

然后把 `artifacts/user-acceptance/onnx-engine-build-evidence-sidecar.template.json` 复制为模型旁边的 `models/model-evidence.sidecar.json`，补 `modelSource`、`modelSha256`、`modelLicense`、`inputAssetName`、`inputAssetSha256`、`stdoutSummary` 和 `stderrSummary`。回填后运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OnnxEngineBuildEvidenceSidecar.ps1
```

这个 validator 会把模板清单中尚未创建的 sidecar 记为 `owner-action-required`，而不是 error。只有当真实模型、真实输入、hash、许可证和 sample runner 日志都齐全后，Classification 或 YoloVision 的 asset manifest 才能考虑晋级到 `real-model-runtime`。

真实 sample runner 日志建议再落一份 sample run evidence record：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-SampleRunEvidenceRecordTemplate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1
```

这份记录不关心 engine build 参数细节，而是保存 `Classification` / `YoloVision` 的真实运行命令、日志路径、日志 SHA256、stdout/stderr 摘要、预期 evidence lines 和 `canPromoteRealModelRuntime`。它最多用于样例 `real-model-runtime` 晋级，不允许声明 `package-consumer-runtime`。

## Plugin 和 Timing Cache 的当前边界

`--plugins`、`--plugin`、`--dynamicPlugins`、`--setPluginsToSerialize` 与 `--timingCacheFile` 已经被 parser 和 application 接受，并写入 diagnostics。plugin 参数可以重复出现，也可以使用逗号或分号列表；parser 会归一化成 `Plugins`。当前阶段不会加载 plugin library，也不会导入/导出 timing cache。原因是 plugin register/load/deregister、serialized plugin ownership 和 cache 生命周期都需要更清晰的 ABI 与 ownership 设计。

如果模型依赖自定义 plugin，推荐先记录：

```powershell
--plugins .\plugins\my_plugin.dll --plugin .\plugins\custom_op.dll --dynamicPlugins .\plugins\dynamic_plugin.dll
```

然后在报告里保留 diagnostics，作为后续 plugin 安全加载阶段的输入。

## 常见排查

1. ONNX 文件不存在：确认路径是否为当前工作目录下的正确路径。
2. dynamic shape 缺失：补齐 `--minShapes`、`--optShapes`、`--maxShapes`。
3. TensorRT 不支持某些算子：先用官方工具验证，再决定是否需要 plugin 或 graph rewrite。
4. CUDA/TensorRT DLL 找不到：启用 `JYPPX_ENABLE_DEVELOPMENT_PROBING=1` 并检查本机 runtime。
5. `blocked-by-cuda-driver`：这是驱动/runtime 兼容问题，不应写成 API 未完成。

## 下一步

完成 build-only 后，再根据模型类型选择 sample：

- 分类模型：`samples/Classification`
- YOLO-family：`samples/YoloVision`
- 自定义推理：基于 `TensorRtInferenceBindings` 写具体 input/output 绑定

这样 ONNX 到 engine 的转换、模型输入输出绑定、后处理和真实资产证据是分层推进的。项目可以更快定位问题，也更容易写出可信的发布材料。

## 第三批正文门禁

### 适用读者

本文适合需要把 ONNX 模型转换成 TensorRT engine 的用户，也适合维护 OnnxToEngine 与 TensorRtExec parity 的开发者。

### 解决问题

本文解决模型转换路径的分层：dry-run 只验证参数，build-only 只证明 engine 构建，synthetic runtime 只证明最小运行，真实模型 runtime proof 需要输入资产和 validator。

### 背景与场景

官方 `trtexec` 是很多用户的基准工具。TensorRtSharp 的 OnnxToEngine 和 TensorRtExec 要尽量贴近它的转换体验，但仍要用 C# wrapper 和 release proof record 明确证据边界。

### 操作路径

先用 dry-run 检查参数归一化，再用 build-only 构建 engine 和 TensorRtExec report；随后按模型类型进入 Classification、YoloVision 或自定义 runner，补真实输入、输出摘要、host metadata、hash 和 validator。OnnxToEngine report 只证明转换链路，不能替代真实模型 runtime proof。

### 代码与文件入口

- `samples/OnnxToEngine/Program.cs`
- `applications/TensorRtExec/README.md`
- `samples/OnnxToEngine/trtexec-parity-matrix.json`
- `artifacts/user-acceptance/onnx-engine-build-evidence-sidecar.template.json`

### 边界说明

build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 都不是 runtime proof。真实模型 proof 必须来自真实输入、真实输出、hash、host metadata 和 validator。

### 下一步

下一步选择一个真实 ONNX 模型，补模型下载、导出、build-only、runtime runner 和 proof validator 的完整图文链路。
