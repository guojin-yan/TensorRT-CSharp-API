# Readonly Summary 证据矩阵

`artifacts/interface-coverage/readonly-summary-evidence-matrix.json` 记录最近几批提升出来的 readonly summary、deployment summary、CUDA diagnostic summary、Plugin Registry Inventory summary 和 PluginCreatorV3 metadata design gate。它的目标是把“真实完成度推进”从口头边界变成可审查、可测试的证据矩阵。

## 定位

这些 summary 证明的是：

- 公开 C# wrapper 已经存在，并且提供 pointer-free 的托管摘要。
- smoke 或质量测试能看到对应 marker 或源码入口。
- 相关状态是 copied snapshot、copied scalar、copied string、copied array 或 design-gate result。
- deferred 记录不能仅因为 summary 存在而删除。

这些 summary 不能证明的是：

- runtime inference 已经执行。
- TensorRT callback、plugin enqueue、allocator 或 DebugListener 已经被真实调用。
- clean package consumer 已经从包源 restore、复制 native assets 并成功运行。
- post-publish verification 已经完成。

## 当前覆盖

| ID | 公开类型 | 证据类型 | smoke marker | 边界 |
| --- | --- | --- | --- | --- |
| `tensorrt-runtime-diagnostic-summary` | `TensorRtRuntimeDiagnosticSummary` | readonly diagnostics | `RuntimeDiagnosticSummary=` | 非 runtime proof |
| `tensorrt-engine-deployment-summary` | `TensorRtEngineDeploymentSummary` | deployment diagnostics | `EngineDeploymentSummary=` | 非 runtime proof |
| `tensorrt-builder-config-deployment-summary` | `TensorRtBuilderConfigDeploymentSummary` | deployment diagnostics | `BuilderConfigDeploymentSummary=` | 非 runtime proof |
| `tensorrt-execution-context-deployment-summary` | `TensorRtExecutionContextDeploymentSummary` | deployment diagnostics | `ExecutionContextDeploymentSummary=` | 非 runtime proof |
| `tensorrt-serialization-config-summary` | `TensorRtSerializationConfigSummary` | readonly config | `SerializationConfigSummary=` | 非 runtime proof |
| `tensorrt-runtime-config-summary` | `TensorRtRuntimeConfigSummary` | readonly config | `RuntimeConfigSummary=` | 非 runtime proof |
| `tensorrt-onnx-parser-diagnostic-summary` | `TensorRtOnnxParserDiagnosticSummary` | parser diagnostics | `ParserDiagnosticSummary=` | 非 runtime proof |
| `tensorrt-onnx-model-support-summary` | `TensorRtOnnxModelSupportSummary` | parser support diagnostics | `ParserModelSupport` | 非 runtime proof |
| `tensorrt-error-recorder-summary` | `TensorRtErrorRecorderSummary` | callback boundary diagnostics | `ErrorRecorderSummary=` | 非 runtime proof |
| `tensorrt-plugin-creator-v3-metadata-design-gate` | `TensorRtPluginCreatorV3MetadataDesignGateResult` | design gate only | `PluginCreatorV3MetadataDesignGate=` | 非 runtime proof |
| `tensorrt-plugin-registry-inventory-summary` | `TensorRtPluginRegistryInventoryDiagnostics` | plugin registry copied metadata | `PluginRegistryInventoryDiagnostics` | 非 runtime proof |
| `cuda-graph-diagnostic-summary` | `CudaGraphDiagnosticSummary` | CUDA diagnostics | `GraphDiagnosticSummary=` | 非 runtime proof |
| `cuda-graph-exec-diagnostic-summary` | `CudaGraphExecDiagnosticSummary` | CUDA diagnostics | `GraphExecDiagnosticSummary=` | 非 runtime proof |
| `cuda-device-graph-memory-summary` | `CudaDeviceGraphMemorySummary` | CUDA graph memory counters | `CudaGraphMemory ... Summary=[...]` | 非 runtime proof |
| `cuda-memory-range-diagnostic-summary` | `CudaMemoryRangeDiagnosticSummary` | CUDA diagnostics | `MemoryRangeSummary=` | 非 runtime proof |

## 晋级规则

Readonly summary 只允许作为 API 完成度、wrapper 可用性、pointer-free 边界和 smoke marker 的证据。它们不能替代以下任何真实 proof：

- `package-consumer-runtime`
- real-model runtime proof
- callback runtime proof
- Linux runner proof
- post-publish verification
- owner-authorized release close proof

下一步如果要把某条链路晋级到真实 proof，必须走 clean consumer restore、runtime package restore、native assets copy、runtime smoke、stdout/stderr summary、host metadata、package hash 和 strict validator，而不是引用本矩阵。
