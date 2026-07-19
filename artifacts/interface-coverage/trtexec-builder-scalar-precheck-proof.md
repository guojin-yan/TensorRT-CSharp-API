# TensorRtExec Builder Scalar Precheck Proof

本 artifact 记录 TensorRtExec 对四个 trtexec-like builder scalar 的 dry-run 预检：`--maxNbTactics`、`--tilingOptimizationLevel`、`--l2LimitForTiling` 和 `--quantizationFlags`。

## 结果

- `ProofClassification=precheck`
- `State=dry-run-precheck`
- 四个参数均进入 normalized command、`ParsedOptions` 和 `ParseOnlyOptions`
- `AppliedOptions` 为空
- 未读取 ONNX、未创建 TensorRT builder config、未保存 engine、未执行 inference
- normalized command SHA256：`2a7a21cb48d8b1985ceb41e4c2d59e9b95743b30a76db6bb3bbed8edbe8767b7`

## 证据边界

该 artifact 只证明参数解析、归一化和报告分类。真实 TensorRT 10/11 build 中的 scalar setter/readback 由 `OnnxEngineBuildService.ApplyBuilderScalarDeploymentControls` 记录 `Requested`、`Applied`、`Readback` 与 `ReadbackMatch`；当前主机没有把 dry-run 结果升级成 builder readback 或 runtime proof。

`isRuntimeExecutionProof=false`、`isRealModelRuntimeProof=false`、`isPackageConsumerRuntimeProof=false`。旧 deferred manifest/history 保留。
