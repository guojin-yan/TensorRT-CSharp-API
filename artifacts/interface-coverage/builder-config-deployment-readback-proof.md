# Builder Config Deployment Readback Proof

本批把已有的 `TensorRtBuilderConfig.GetDeploymentSnapshot()` 接入 `OnnxEngineBuildService`、`OnnxEngineBuildResult` 和 TensorRtExec/OnnxToEngine report。它不是新的 native ABI，而是把用户请求的 deployment options 与 TensorRT builder config 成功创建后的 copied readback 放到同一份报告中。

## 跨版本路由

| 版本 | native family | 结果 |
| --- | --- | --- |
| TRT8 | `jyppx_trt8_builder_config_*` | vendor bridge 可加载时读回；版本专属不支持项进入 diagnostics |
| TRT10 | `jyppx_trt10_builder_config_*` | vendor bridge 可加载时读回；现代 memory-pool/timing 语义保留 |
| TRT11 | `jyppx_trt11_builder_config_*` | vendor bridge 可加载时读回；DLA、tiling、tactic 等字段按已有 guard 读取 |

报告同时保留 `DeploymentOptions` 请求值和 `BuilderConfigDeploymentSnapshot` 实际复制值，便于发现归一化、版本 guard 或 vendor 不支持差异。快照只包含 managed copied values、serialized plugin path snapshot 和 diagnostics，不暴露 config pointer。

## 本机证据边界

本轮 dry-run 保持 `precheck`。随后使用仓库已有 TRT8/CUDA12.1 bridge-only 输出运行当前 OnnxToEngine：bridge 成功加载并报告 runtime/builder 可用，但 vendor runtime creation 在 builder-config 创建前触发 Windows structured exception `3228369022`。因此报告正确保持 `dependency-probe-only`，没有把真实 readback 写成成功证据。`isRuntimeExecutionProof=false`、`isRealModelRuntimeProof=false`、`isPackageConsumerRuntimeProof=false`；兼容 TensorRT/CUDA host 上的真实 builder readback 仍需后续采集。
