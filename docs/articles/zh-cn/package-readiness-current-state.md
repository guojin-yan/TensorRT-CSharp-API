# 当前 Package Readiness 状态说明

本文是当前 `win-x64-trt11.0-cuda13.2-cudnn9.22` 本地 readiness 状态的简要说明，便于和教程、排障文章一起引用。

## 当前结论

当前 package readiness 是 clean 的：

- managed package：ready。
- bridge package：ready。
- bridge consumer：ready。
- split components：ready 3/3。
- split collection package：ready。
- split collection consumer：ready。
- full vendor inputs：ready。
- full runtime package：ready。
- full consumer：ready。
- overall：ready。
- runtime proof：`blocked-by-cuda-driver`。
- readiness blockers：0。
- vendor blockers：none。

这里的 `overall=ready` 只表示 package layout、vendor asset、restore/build/native-copy 和 consumer 报告链条已经完整；真实运行证明要看单独的 `runtime proof` / `runtimeExecution.status`。当前 runtime execution smoke 是环境阻塞：

- full package consumer smoke requested：true。
- smoke result：`blocked-by-cuda-driver`。
- diagnostic：`cudaRuntimeGetVersion reported CUDA error 35`。
- package consumer evidence kind：`full-runtime-package-consumer-smoke-driver-blocked`。
- runtime smoke classification：`runtime-smoke-driver-blocked`。
- is runtime execution evidence：false。
- is dependency probe only：true。
- is real callback runtime proof：false。
- real callback runtime evidence：`blocked-by-cuda-driver`。
- proof：false。

## 它证明了什么

它证明：

- 本地 managed package 存在。
- split package 和 full runtime package 都存在。
- runtime package 能被 consumer restore/build。
- native assets 能复制到 consumer output。
- bridge 能加载并报告 TensorRT/CUDA build info。
- bridge consumer 的 high-level wrapper surface 已覆盖 `engine-rnn-readonly-diagnostics`，包含 `HasImplicitBatchDimensionCompatibility`、`SerializedPluginPathCountCompatibility` 以及 TRT8 RNNv2 的 layer count、hidden size、max sequence length、operation、direction、input mode 只读诊断。
- readiness 能把 CUDA driver/runtime compatibility 问题归类为环境阻塞。
- `packageConsumerEvidenceKind`、`runtimeSmokeClassification`、`isRuntimeExecutionEvidence`、`isDependencyProbeOnly`、`isRealCallbackRuntimeProof` 能把 package consumer/native-copy、dependency probe、runtime execution 和 callback proof 分开记录。

## 它没有证明什么

它没有证明：

- 当前机器可以执行 CUDA 13.2 runtime smoke。
- `isDependencyProbeOnly=true` 可以写成 runtime execution proof。
- 所有样例都能在当前机器跑通。
- callback runtime proof 已完成。
- `IGpuAllocator::*`、`IGpuAsyncAllocator::*`、`IOutputAllocator::*`、`IDebugListener::processDebugTensor` 已经可作为 public callback API 使用。
- NVIDIA 二进制再分发许可已完成最终复核。

## 推荐引用方式

写文章或发布说明时，可以写：

> 当前 `win-x64-trt11.0-cuda13.2-cudnn9.22` package readiness 已 clean，split/full package 和 consumer native-copy 证据齐全；本机 full package consumer runtime smoke 被 CUDA driver/runtime compatibility 阻塞为 `blocked-by-cuda-driver`，因此真实 callback runtime proof 仍为 `false`。

不要写：

> 当前 TensorRT 11 + CUDA 13.2 的所有 runtime smoke 都已经通过。

这会混淆 package readiness、runtime smoke 和 callback proof。
