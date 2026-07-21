# TensorRtExec 部署策略：Device、DLA、Tactic 与 Strongly Typed

官方 `trtexec` 不只是模型转换器。它还负责选择 CUDA device、约束 tactic 来源、配置 DLA/GPU
fallback、设置网络边界策略，并决定是否创建 strongly typed network。参数能被 parser 接收并不代表
这些策略已经生效；可靠实现必须同时记录 requested、readback、版本 guard 和失败原因。

本章说明 TensorRtExec 与 OnnxToEngine 如何实现以下选项：

- `--device`
- `--useDLACore` 与 `--allowGPUFallback`
- `--tacticSources`
- `--directIO`
- `--sparsity`
- `--stronglyTyped`

## 快速示例

```powershell
dotnet run --project .\applications\TensorRtExec\TensorRtExec.csproj -- `
  --tensor-rt-line 10 `
  --onnx .\models\model.onnx `
  --saveEngine .\models\model.plan `
  --device 0 `
  --tacticSources=-CUDNN `
  --allowGPUFallback `
  --directIO `
  --sparsity enable `
  --stronglyTyped `
  --buildOnly `
  --exportReport .\artifacts\deployment-report.json
```

对于真实外部模型，strongly typed、DirectIO 或 tactic 约束可能改变模型可构建性。应先分别启用，再
组合验证；vendor 拒绝不能被改写成成功或 capability proof。

## CUDA Device 为什么使用专用线程

CUDA current device 是线程状态。若库代码在调用者线程直接执行 `CudaDevice.SetCurrent`，即使最终
构建成功，也会把选择结果泄漏给宿主程序后续工作。TensorRtExec 在指定 `--device` 时创建专用 host
thread，在该线程完成 device set/readback、TensorRT build/load 和 bounded runtime，资源释放后线程
结束，调用者线程不被污染。

报告中的 `TrtexecDeploymentControl Name=Device` 同时记录请求 ordinal、读回 ordinal、设备数量、
执行线程 ID 和 `ReadbackMatch`。越界 ordinal 直接失败，不回退到 device 0。

## DLA 与 GPU Fallback

`--useDLACore=N` 在写入 builder config 之前读取 `TensorRtBuilder.DlaCoreCount`。当 `N` 不在范围内时，
构建以明确错误失败。例如没有 DLA 的 RTX 主机请求 core 0，会得到：

```text
--useDLACore requested core 0, but TensorRT reports 0 DLA core(s).
```

有效 core 会设置并读回 `DefaultDeviceType=Dla` 与 `DlaCore=N`。`--allowGPUFallback` 使用 typed builder
flag set/get。TensorRT 11 还会给 ONNX parser 附加 builder config 并启用 DLA capability flags。

这些 readback 只证明配置被 TensorRT 接收，不证明 layer 真正在 DLA 执行。DLA proof 仍需要：DLA
主机、适配模型、真实 build/run log、输出校验和 owner 审核。

## Tactic Sources

语法与官方一致，每项必须用 `+` 或 `-` 表示相对默认 mask 的增删：

```text
--tacticSources=-CUDNN,+CUBLAS
```

支持 `CUBLAS`、`CUBLAS_LT`、`CUDNN`、`EDGE_MASK_CONVOLUTIONS` 和 `JIT_CONVOLUTIONS`。实现先调用
`GetTacticSources()` 获取 vendor 默认值，再解析增删项，调用 `SetTacticSources()`，最后再次读回。
未知 token 或缺少正负号会在 parser 阶段失败。

mask readback 不等于某个 tactic 被选中，也不等于性能改善。实际 tactic 和性能必须由 layer profile、
timing 与真实模型证据说明。

## DirectIO 与 Sparsity

`--directIO` 设置并读回 `TensorRtBuilderFlag.DirectIO`。该 flag 已被新版 TensorRT 标记为 deprecated，
但官方命令仍保留，因此实现保持兼容且不隐藏 vendor 错误。

`--sparsity=enable|disable` 设置并读回 `SparseWeights` flag。`--sparsity=force` 不仅启用 flag，官方工具
还会重写不满足结构化稀疏条件的模型权重；当前仓库没有这段模型变换，因此 force 明确保留
parse-only，不能只设置一个 flag 后冒充完整实现。

## Strongly Typed 的版本边界

TRT10 使用 raw bit 1（`1u << 1`）创建 strongly typed network。TRT11 的 vendor 契约是所有 network
始终 strongly typed，因此不会复用 TRT10 的 raw bit；TRT8 适配器则保持 explicit-batch network，并在
report 中把 `--stronglyTyped` 留在 `ParseOnlyOptions`。这些 guard 在 network 创建前生效，不依赖 vendor
崩溃或偶然忽略未知 bit。

strongly typed 会约束 precision 推导。它与 FP16、INT8、layer precision 或 weight streaming 的组合
必须按官方约束和真实模型逐项验证，不能从 identity model smoke 推广到任意模型。

## 如何检查报告

成功 build 应在 `OptionImplementationStatus.AppliedOptions` 中看到实际应用的选项，并在 log 中看到
对应的 `TrtexecDeploymentControl ... ReadbackMatch=True`。以下情况仍在 `ParseOnlyOptions`：

- dry-run 或 load-engine 对 builder-only 选项的请求；
- TRT8 的 `--stronglyTyped`；
- `--sparsity=force`；
- vendor setter/readback 失败的选项。

本地 TRT10.11/CUDA12.9 identity smoke 可以证明 device thread、builder config、network creation、engine
round-trip 和 enqueue 行为，但仍是 `synthetic-input-runtime`。它不是 real-model runtime proof，也不是
repository-external package consumer proof。

## 证据边界

部署策略证据回答的是“请求是否在这台主机上被应用并读回”，不回答以下问题：

- 模型输出是否正确；
- DLA 是否执行了目标 layers；
- sparse tactic 是否被选择；
- 性能是否提升；
- NuGet/GitHub package 是否能被外部消费者运行；
- 是否允许公开发布或关闭 release issue。

公开结论仍必须由真实模型、外部 clean consumer、公开包 URL/hash、CI 和 owner acceptance 共同证明。
