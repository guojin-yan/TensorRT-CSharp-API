# Package Consumer Runtime Proof 预检矩阵

`artifacts/final-release/package-consumer-runtime-proof-preflight-matrix.json` 是给 release owner 使用的预检合同。它把 Windows runtime package key、包 ID、native asset 预期数量、允许采集的 readonly summary marker、禁止替代项和 validator 命令集中到一个机器可读文件里。

它不是 runtime proof。它只说明下一步要怎样采集 proof，以及哪些证据不能晋级。

## 覆盖范围

当前矩阵覆盖 6 个 Windows runtime key：

| Runtime key | TensorRT | CUDA | cuDNN | 状态 |
| --- | --- | --- | --- | --- |
| `win-x64-trt8.6-cuda11.8-cudnn8.9` | 8.6.1.6 | 11.8 | 8.9.7.29 | owner runtime smoke 待回填 |
| `win-x64-trt8.6-cuda12.1-cudnn8.9` | 8.6.1.6 | 12.1 | 8.9.7.29 | owner runtime smoke 待回填 |
| `win-x64-trt10.11-cuda11.8-cudnn8.9` | 10.11.0.33 | 11.8 | 8.9.7.29 | owner runtime smoke 待回填 |
| `win-x64-trt10.11-cuda12.9-cudnn9.22` | 10.11.0.33 | 12.9 | 9.22.0 | owner runtime smoke 待回填 |
| `win-x64-trt11.0-cuda12.9-cudnn9.22` | 11.0.0.114 | 12.9 | 9.22.0 | owner runtime smoke 待回填 |
| `win-x64-trt11.0-cuda13.2-cudnn9.22` | 11.0.0.114 | 13.2 | 9.22.0 | 需要 CUDA 13 capable host |

## 允许采集但不能晋级的 marker

以下 marker 可以出现在 package consumer smoke 或日志摘要中，但只能作为 API/wrapper/diagnostic evidence：

- `EngineDeploymentSummary=`
- `BuilderConfigDeploymentSummary=`
- `ExecutionContextDeploymentSummary=`
- `SerializationConfigSummary=`
- `RuntimeConfigSummary=`
- `GraphDiagnosticSummary=`
- `GraphExecDiagnosticSummary=`
- `MemoryRangeSummary=`

这些 marker 不能替代 clean consumer runtime proof、callback runtime proof、real-model runtime proof 或 post-publish verification。

## 禁止替代项

矩阵和质量门禁要求以下证据不能被写成 `package-consumer-runtime`：

- readonly summary
- readonly diagnostics
- TensorRtExec report
- OnnxToEngine report
- YoloVision matrix
- bridge-only
- dependency probe
- local feed
- ProjectReference
- direct `.nupkg`
- build-only
- dry-run
- template
- blocked-by-cuda-driver
- CudaDeviceInitializationProofRunner local-smoke
- `Skipped=True`
- `SmokeResult=passed` 但没有 strict validator

## 晋级条件

`canPromotePackageConsumerRuntimeProof=true` 只能在以下全部满足后出现：

1. consumer 工程位于仓库外或 owner 指定的 clean proof 目录。
2. 不使用 ProjectReference。
3. managed/runtime package 均从 package source restore。
4. native assets 已复制并记录 listing/hash。
5. runtime smoke 真实执行，命令包含目标 `--runtime-package-key`。
6. 记录 stdout/stderr summary。
7. 记录 managed/runtime nupkg SHA256 和 runtime smoke log SHA256。
8. 记录 OS、GPU、driver、CUDA、TensorRT、cuDNN host metadata。
9. `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof` 通过。

当前预检矩阵所有条目保持 `canPromotePackageConsumerRuntimeProof=false`，因为它们还没有绑定真实 owner runtime smoke log 与 strict validator 结果。

## 自动同步边界

`eng/Test-PackageConsumer.ps1` 现在会读取这份矩阵，并把每个 runtime key 对应的 `RuntimeProofPreflight` 对象写入 `package-consumer-validation-summary.json`。该对象只复制：

- matrix schema version
- entry found
- owner action required
- restore source mode
- ProjectReference 边界
- native asset 预期/实际占位
- runtime smoke required
- can promote flag
- blocked reason
- validator command

这一步是为了减少报告、矩阵和 manifest 的人工漂移，不会改变 proof 等级。即使 package consumer smoke 执行成功，只要没有外部 strict validator 绑定真实日志、hash 和 host metadata，`IsPackageConsumerRuntimeProof` 仍必须保持 `false`。

`CudaDeviceInitializationProofRunner` 已有本地初始化 smoke scaffold 和机器可读分类记录，但它的 `ProofKind` 必须保持 `local-smoke-not-external-proof`。`IsPackageConsumerRuntimeProof=False`、`CanPromoteRuntimeProof=False` 与 `Skipped=True` forbidden substitute 是预检矩阵的一部分；该 runner 只证明 wrapper 调用顺序和本机可执行路径，不证明 clean external package consumer runtime。

质量测试还会把矩阵条目与 `pack/runtime/runtime-packages.manifest.json` 的 Windows runtime package 逐项对齐，校验 package id、RID、TensorRT/CUDA/cuDNN 版本、distribution tier、validation state 和 native asset 预期数量。后续新增或修改 runtime key 时，必须同步更新 manifest、preflight matrix、文档和测试。
