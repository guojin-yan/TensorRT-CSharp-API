# OnnxToEngine 与 TensorRtExec 如何分工

TensorRtSharp4.0 里有两个容易被混用的入口：`applications/OnnxToEngine` 和 `applications/TensorRtExec`。它们都和 ONNX 到 TensorRT engine 有关，但目标不同。前者是最小可验证样例，后者是面向用户的工具应用。

理解这条分工，可以少走很多弯路：先用 OnnxToEngine 证明最小构建链路，再用 TensorRtExec 管理外部 ONNX 的 build/report 工作流，最后由具体 sample runner 证明真实模型语义。

## OnnxToEngine：最小 round-trip 样例

`applications/OnnxToEngine` 适合回答一个基础问题：在当前 TensorRT/CUDA 运行环境下，项目能否完成最小 ONNX parser、profile、engine build、deserialize、binding 和 readback。

它默认生成内置 dynamic identity ONNX，不依赖外部模型资产：

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"

dotnet run --project .\applications\OnnxToEngine -- `
  --tensor-rt-line 10 `
  --batch 2
```

理想 evidence lines 包括：

```text
Parsed=True
ProfileIndex=0
EngineFileRoundTrip=True
BindingReport Ready=True
OutputMatch=True
```

这条路径适合写入入门教程、CI smoke 和本机环境 sanity check。它证明的是项目内置 identity 模型链路，不证明任意外部 ONNX 模型都已经具备推理语义。

## TensorRtExec：面向用户的 build/report 工具

`applications/TensorRtExec` 是用户侧 ONNX-to-engine 工具，提供 CLI 和 WinForms 双入口。它复用 `src/JYPPX.TensorRtSharp.Tools` 的参数模型：

- `TrtexecLikeParser`
- `TrtexecLikeOptions`
- `OnnxEngineBuildOptions`
- `OnnxEngineBuildService`
- `OnnxEngineBuildDiagnostics`

外部 ONNX 常见 build-only 命令：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model.plan `
  --minShapes input:1x3x640x640 `
  --optShapes input:1x3x640x640 `
  --maxShapes input:4x3x640x640 `
  --fp16 `
  --workspace 512 `
  --profilingVerbosity detailed `
  --buildOnly `
  --skipInference `
  --exportReport .\models\model-build-report.json `
  --evidenceSidecar .\models\model-evidence.sidecar.json
```

这条命令适合回答：模型文件是否存在、shape profile 是否合理、parser/builder 是否能走到构建边界、报告字段是否可用于后续证据回填。

## 为什么外部 ONNX 默认不做通用推理

一个 ONNX 能 build 成 engine，并不代表工具知道如何喂输入、读输出、解释结果。分类、检测、分割、pose、OBB、语义分割的输出语义都不同。即使同是 YOLO-family，不同版本和导出脚本也可能产生不同 tensor layout。

所以 TensorRtExec 当前把外部 ONNX 默认视作 build/report 证据：

```text
ProofClassification=build-only
BuildEvidenceOnly=True
InferenceRan=False
OutputMatch=False
IsRuntimeExecutionProof=False
```

这不是功能缺失，而是边界清晰。真实模型 runtime proof 应由 `samples/ComputerVision/01.Classification`、`applications/YoloVision` 或用户自己的 binding 应用来补足。

对 YoloVision 来说，输出语义不由 OnnxToEngine 猜测。`applications/YoloVision/yolovision-task-output-contract.json` 负责定义 det/cls/seg/obb/pose/sem 的输出角色、required metadata、TensorRtExec profile hint 和 proof boundary；OnnxToEngine 或 TensorRtExec 的 build/report 只能作为 owner backfill 的构建证据，不能替代该 contract 与 YoloVision 真实运行日志。

## dryRun 与 buildOnly 的区别

`--dryRun` 或 `--previewOnly` 是 precheck：

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model.plan `
  --minShapes input:1x3x640x640 `
  --dryRun `
  --exportReport .\models\model-precheck-report.md
```

它不读取 ONNX，不探测 TensorRT runtime，不构建 engine，不运行 inference。它只证明参数可以解析、命令可以归一化、报告可以落盘。

`--buildOnly` 则会进入 parser/builder 路径，成功时可作为模型转换和构建证据。但 build-only 不是 inference proof，也不是 package consumer runtime proof。

## Evidence sidecar 的作用

`--evidenceSidecar` 用来把模型来源、模型 SHA256、许可证、输入资产 SHA256、stdout/stderr 摘要等字段写进 build report。它是 build report 和真实资产记录之间的桥。

sidecar 不能越级。即使 sidecar 写了 `real-model-runtime` 或 `package-consumer-runtime`，build report 也不能因此变成 release proof。真实模型 runtime 由 sample runner 证明；package consumer runtime 由 `external-runtime-proof-record.json` 和 release proof record 证明。

## 与 Classification / YoloVision 的连接

推荐流程：

1. 用 TensorRtExec 对外部 ONNX 做 build-only，生成 engine 和 report。
2. 用 sidecar 记录模型来源、hash、许可证和输入资产。
3. 根据模型类型选择 sample：
   - 分类模型：`samples/ComputerVision/01.Classification`
   - YOLO-family：`applications/YoloVision`
4. 运行 sample runner，保存真实日志。
5. 回填 `sample-run-evidence-record`。
6. 运行 manifest 和 user acceptance catalog 校验。

这样 build evidence、real-model runtime evidence 和 release proof 不会混在一起。

## 常见误读

- `Parsed=True` 不等于推理正确。
- `EngineSaved=True` 不等于真实模型输出可解释。
- `OutputMatch=True` 只在内置 identity round-trip 中证明最小输出匹配。
- `build-only` 不是 runtime execution proof。
- `blocked-by-cuda-driver` is not smoke passed；它是驱动/runtime 兼容性阻塞。
- `package-consumer-runtime` belongs to release proof records，必须由 `external-runtime-proof-record.json` 等 owner proof record 和 validator 证明，不能由 TensorRtExec、OnnxToEngine、sidecar 或 sample manifest 直接声明。

## 小结

OnnxToEngine 是教学和最小验证入口，TensorRtExec 是外部 ONNX build/report 工具。它们合在一起能覆盖从入门到模型转换的大部分工作，但真实模型语义和发布 proof 仍需要更高层证据链。分工越清楚，用户越容易定位问题，发布材料也越不容易过度承诺。
