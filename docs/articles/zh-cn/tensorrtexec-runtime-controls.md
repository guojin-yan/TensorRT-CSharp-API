# TensorRtExec 线程、Spin Wait、CUDA Graph 与零传输实战

`TensorRtExec` 与 `OnnxToEngine` 的 bounded benchmark 不再只解析高级 runtime 参数。本篇说明
`--threads`、`--useSpinWait`、`--useCudaGraph` 和 `--noDataTransfers` 的真实执行语义、使用方式、
报告判读与证据边界。实现对照 TensorRT 10.11 官方 `trtexec` 帮助以及 v10.11 源码中的
`samples/common/sampleOptions.cpp`、`sampleInference.cpp`。

## 先理解四个开关

`--threads` 是布尔开关，不是线程数。有效并发数仍由 `--infStreams`（优先）或 `--streams`
决定；开启后，每个 execution context 由独立 host driver thread 驱动。为了兼容旧调用方，parser
仍接受 `--threads 2` 和 `--threads=2`，但数值只表示“开关出现过”，normalized command 始终输出
裸 `--threads`。

`--useSpinWait` 不等同于修改报告字段。worker 会查询 stop `CudaEvent.IsReady()` 并主动等待；未
开启时 event 使用 blocking-sync flags 并调用 `Synchronize()`。两种路径都保留 GPU event timing。

`--useCudaGraph` 会先执行一次 direct enqueue，完成 TensorRT/CUDA lazy initialization；随后按
context 捕获 enqueue、结束 capture、实例化 graph exec，并在 warmup/measurement 中调用 graph
launch。只要任一 context 捕获失败，所有已创建 graph owner 都会释放，整次 run 回退 direct
enqueue，并写入 `UseCudaGraphFallbackReason`。这种 run 不把 `--useCudaGraph` 标成 applied。

`--noDataTransfers` 只分配并绑定 device buffers，不执行 input H2D，也不执行 output D2H/readback。
它可以证明 scheduler 和 TensorRT enqueue 确实运行，却不能证明输出正确。报告必须保持
`OutputMatch=false`；`--dumpOutput`、`--exportOutput` 和 `--dumpRawBindingsToFile` 也保持
parse-only，只输出无 tensor 数据的边界 artifact。

## 线程、Spin Wait 与 CUDA Graph

以下命令使用两个 execution contexts，分别由独立线程驱动，并同时启用 event polling 和 CUDA
Graph：

```powershell
dotnet run --project .\samples\OnnxToEngine -- `
  --tensor-rt-line 10 `
  --iterations 3 --warmUp 5 --duration 0 `
  --streams 1 --infStreams 2 `
  --threads --useSpinWait --useCudaGraph `
  --avgRuns 2 --percentile 90 `
  --exportReport .\run\runtime-controls-report.json `
  --exportTimes .\run\runtime-controls-times.json
```

重点检查 `BenchmarkSummary`：

```json
{
  "ThreadsExecuted": 2,
  "UseSpinWaitApplied": true,
  "UseCudaGraphRequested": true,
  "UseCudaGraphApplied": true,
  "UseCudaGraphFallbackReason": "",
  "MeasurementRoundsPerContext": [3, 3]
}
```

然后检查 `OptionImplementationStatus`。只有真实执行成功的 control 才应位于 `AppliedOptions`。
`--infStreams 2` 覆盖 `--streams 1`，因此后者继续出现在 `ParseOnlyOptions`，这是有效配置优先级，
不是调度失败。

## 零传输基准

```powershell
dotnet run --project .\samples\OnnxToEngine -- `
  --tensor-rt-line 10 `
  --iterations 2 --warmUp 2 `
  --noDataTransfers `
  --dumpOutput `
  --exportOutput .\run\no-transfer-output.json `
  --dumpRawBindingsToFile .\run\no-transfer-output.raw `
  --exportTimes .\run\no-transfer-times.json `
  --exportReport .\run\no-transfer-report.json
```

预期结果是 `InferenceRan=true`、`NoDataTransfersApplied=true`、`TimingSampleCount>0`，同时
`OutputMatch=false`、`OutputElementCount=0`、`HasRawBindingProof=false`。times artifact 的
`HasBenchmarkExecutionEvidence=true` 只说明 enqueue/timing 已执行；它不改变 tensor correctness、
real-model 或 package-consumer proof 的门槛。

## 为什么 sleepTime 仍未实现

官方 `--sleepTime` 是 device-side stream sleep，用来形成 launch-to-compute gap。把它替换成
`Thread.Sleep` 只会暂停 host submission，无法重现相同 GPU timeline，因此当前仍是明确的
parse-only control。`--idleTime` 则本来就是 measurement rounds 之间的 host idle gap，继续使用
CPU sleep 是符合语义的。

## 证据与边界

本仓库的 TRT10.11/CUDA12.9 smoke 记录在
`artifacts/interface-coverage/trtexec-runtime-controls-runtime-evidence.json`。它证明两个 driver
threads、event polling、CUDA graph capture/launch 和 no-transfer enqueue 行为，但仍使用 embedded
identity 与 ProjectReference：

- 不是外部真实模型正确性证明；
- 不是公开 NuGet package consumer runtime proof；
- 不能授权 NuGet push、GitHub Packages、GitHub Release 或 release close；
- no-transfer run 尤其不能用 timing 样本替代 output validation。

实际模型发布前仍应由 YoloVision 或模型专属 runner 提供模型来源、输入资产、expected output、
engine/report/log SHA256，以及 clean external package consumer 记录。
