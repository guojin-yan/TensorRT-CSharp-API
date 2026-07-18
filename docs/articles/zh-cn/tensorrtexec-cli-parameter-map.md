# TensorRtExec CLI 参数对照表

`applications/TensorRtExec` 提供 trtexec-like CLI。它不是官方 `trtexec` 的完整复制，而是把 TensorRtSharp4.0 当前支持的 ONNX build/report、preflight、runtime artifact 和证据边界集中到一个 .NET 工具入口。

## 目标读者

- 熟悉 NVIDIA `trtexec`，希望在 .NET 项目中使用类似参数模型的用户。
- 需要把 CLI、WinForms 和 `src/JYPPX.TensorRtSharp.Tools` 统一到同一 service path 的维护者。
- 需要审计 build-only、parse-only、diagnostic、runtime artifact 边界的发布负责人。

## 可复制命令

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model.plan `
  --minShapes input:1x3x640x640 `
  --optShapes input:1x3x640x640 `
  --maxShapes input:4x3x640x640 `
  --fp16 `
  --workspace 512MiB `
  --buildOnly `
  --exportReport .\models\model-build-report.md
```

相关实现路径：

- `applications/TensorRtExec`
- `src/JYPPX.TensorRtSharp.Tools`
- `samples/OnnxToEngine/trtexec-parity-matrix.json`
- `artifacts/user-acceptance/trtexec-option-coverage.md`

## 基础参数

| TensorRtExec | 兼容别名 | 说明 | 状态 |
| --- | --- | --- | --- |
| `--onnx` | 无 | ONNX 输入 | implemented |
| `--saveEngine` | `--save-engine`、`--engine` | engine 输出 | implemented |
| `--loadEngine` | `--load-engine` | 已有 engine preflight | preflight |
| `--tensor-rt-line` | 无 | TRT 8/10/11 line | implemented |
| `--workspace` | 无 | workspace limit | implemented |
| `--minShapes` | 无 | dynamic profile min | implemented |
| `--optShapes` | 无 | dynamic profile opt | implemented |
| `--maxShapes` | 无 | dynamic profile max | implemented |

## 构建与部署参数

| 参数 | 说明 | 状态 |
| --- | --- | --- |
| `--fp16` | FP16 builder intent | implemented when runtime supports it |
| `--bf16` | BF16 intent | version/hardware dependent |
| `--noTF32` | TF32 policy | diagnostic |
| `--int8` | INT8 intent | boundary until calibrator proof |
| `--calib` | calibration cache path | diagnostic |
| `--builderOptimizationLevel` | builder optimization level | implemented |
| `--maxAuxStreams` | auxiliary streams | implemented |
| `--memPoolSize` | memory pool intent | diagnostic |
| `--tacticSources` | tactic source intent | diagnostic |
| `--plugins` | plugin path list | diagnostic; no load/register |

## Runtime 与输出参数

| 参数 | 说明 | 状态 |
| --- | --- | --- |
| `--iterations` | benchmark iterations | embedded synthetic runtime only |
| `--warmUp` | warmup ms | diagnostic/execution when runtime path exists |
| `--duration` | benchmark seconds | diagnostic/execution when runtime path exists |
| `--streams` | stream count | diagnostic |
| `--infStreams` | inference stream count | parse/report-only |
| `--useCudaGraph` | CUDA graph intent | boundary diagnostic |
| `--loadInputs` | input file mapping | runtime path required |
| `--dumpOutput` | console output dump | runtime path required |
| `--dumpRawBindingsToFile` | raw binding dump | runtime path required |
| `--exportOutput` | output JSON | runtime path required |
| `--exportTimes` | timing JSON | runtime path required |
| `--exportProfile` | profile JSON | runtime path required |
| `--saveProfile` | profile output | runtime path required |

## 报告参数

| 参数 | 说明 |
| --- | --- |
| `--exportReport` | 输出 JSON 或 Markdown report |
| `--evidenceSidecar` | 输出 evidence sidecar |
| `--dryRun` / `--previewOnly` | 只做参数归一化和 precheck |
| `--buildOnly` | 构建 engine，不声明推理 proof |
| `--skipInference` | 跳过 runtime inference |

## 边界说明

`TrtexecAlignmentStatus=parse-only` 表示参数进入 parser/report，不表示官方 `trtexec` 同等行为已经完全实现。CLI report 不执行发布、不上传包、不批准 public release、不关闭 release issue。build-only/precheck output cannot be promoted to package-consumer-runtime proof。Boundary keywords: not public package proof, not post-publish proof, not package push, not release close approval.

CLI 与 WinForms 都应继续走 `TensorRtExecOptions` 和 `TensorRtExecService`，避免出现一条路径支持参数、另一条路径只展示控件的分裂。`--int8/--calib`、refit、DLA、plugin library 等能力在没有真实 runtime proof 前只能保持 diagnostic、parse-only、planned 或 blocked 状态；timing cache 已完成 build-cache lifecycle，但仍不等于 runtime proof。

本参数表也是 not runtime proof、not real-model-runtime proof。要证明真实模型推理，需要样例 runner 输出真实日志、hash、stdout/stderr summary 和 validator 结果。

## 截图与图示建议

- CLI normalized command 和 report path 输出截图。
- WinForms 参数填写界面截图。
- 参数状态分层图：implemented、wrapper-ready、diagnostic、parse-only、blocked by runtime proof。

## 下一步

- 更新参数时同步修改 `TrtexecLikeParser`、`TensorRtExecOptions`、`tensor-rt-exec-feature-matrix.json` 和 `trtexec-parity-matrix.json`。
- 新增参数时同步扩展 `OnnxToEngineTrtexecLikeTests`、`TensorRtExecApplicationTests` 或单独的 parity matrix 测试。

## 第二批正文门禁

### 适用读者

本文适合需要把官方 `trtexec` 参数映射到 C# CLI/WinForms 应用的用户，也适合做发布验收的人。

### 解决问题

TensorRtExec 的目标不是只做命令行壳，而是让 CLI 和 WinForms 共享同一套 `TensorRtExecOptions`、`TensorRtExecService` 和 report schema。本文解决参数对齐、状态标记和 proof 边界问题。

### 核心思路

核心思路是让参数拥有状态，而不是只有字符串解析。`implemented` 表示已经连到 service；`parse-only` 表示只进入 report；`diagnostic` 表示只输出环境或配置；`blocked` 表示需要真实 runtime 或 owner 输入。

### 操作路径

在 CLI 中解析参数并生成 normalized command，在 WinForms 中绑定相同 options model，输出 `TensorRtExec report` 记录 parse/build/runtime/proof 状态。对 build-only 或 dry-run 结果保留非 proof 标记。

### 边界说明

build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 都不是 runtime proof。真实模型 proof 需要单独 runner、资产 hash、host metadata 和 validator。

### 下一步

下一步继续补 CLI/WinForms parity，并用真实模型 proof 输入验证参数不只停留在 parse/report 层。
