# Windows API 完整化路线

当前项目节奏已切换为 Windows API 完整化优先。Linux 构建、打包和 handoff 结构继续保留，但不再作为当前主要推进目标。

## TRT11 第十七批状态

当前 manifest 总 API 记录为 `1513`，其中 CUDA=`104`、TensorRT=`1398`、Common=`11`。TensorRT 分线覆盖为 TRT8=`359`、TRT10=`358`、TRT11=`680`。

第十七批新增 43 条 TRT11 部署关键 API，重点覆盖 TensorRT 11 `Dims64` / int64 维度元数据：tensor、network I/O、layer I/O、engine tensor/profile shape、execution context tensor shape/stride、optimization profile shape range，以及 shuffle、slice、dynamic quantize、pooling、resize、fill、convolution、deconvolution、padding 等部署常用层参数维度。

C# 高层新增 `TensorRtDims64` 与 `TensorRtOptimizationProfileShapeRange64`，并在 `TensorRtTensor`、`TensorRtNetworkDefinition`、`TensorRtLayer`、`TensorRtEngine`、`TensorRtExecutionContext`、`TensorRtOptimizationProfile` 上提供 `*Shape64` / `*DimensionExtent64` 系列 API。普通用户不需要接触裸 `IntPtr`；unknown-rank tensor 通过 `TensorRtDims64.UnknownRank` 或 metadata summary 中的 `shape=unknown-rank` 表达。

已验证 `TensorRtSmokeRunner` 的 TensorRT 11 + CUDA 12.9 路径，Dims64 证据数达到 `EvidenceCount=27`；`NetworkTrt11ModernLayerMetadataRunner` 在 TensorRT 11.0 + CUDA 12.9 下 `7/7` modern-layer Dims64 metadata probe 通过。`CudaSmokeRunner`、TRT10/TRT8 `TensorRtSmokeRunner`、TRT10/TRT8 lifecycle、DocFX 0 warning、workflow contract、project quality tests、Debug/Release managed build、TRT11 native build 和 generator deterministic gate 均通过。

## TRT11 第十五批状态

当前 manifest 总 API 记录为 `1436`，其中 CUDA=`104`、TensorRT=`1321`。TensorRT 分线覆盖为 TRT8=`359`、TRT10=`358`、TRT11=`603`。

第十五批新增 8 条 TRT11 部署关键 API，覆盖 engine profile tensor-values V2 读取边界、execution context 反查 engine 名称/I/O tensor 数/layer 数/profile 数、input-consumed event 原生地址诊断值、runtime config allocation strategy 查询，以及 engine inspector error-recorder 清理。

本轮还新增 `TensorRtBuilderConfigDeploymentSnapshot`、`TensorRtEngineDeploymentSnapshot`、`TensorRtExecutionContextDeploymentSnapshot` 三个 C# 高层快照对象。它们将大量已有 native probe 聚合为部署诊断输出，并把 TensorRT 11 已移除或替换的 calibration profile、`allInputShapesSpecified` 等 API 作为非致命 diagnostics 记录，不再让高层部署链路失败。

其中 ExecutionContext 与 BuilderConfig 的 snapshot/summary 已按顶层类型分文件：snapshot 文件拥有复制值、
`ToSummary` 和自身 `ToString`，同名 `*DeploymentSummary.cs` 文件拥有 summary 构造、公开属性和诊断文本。
该归类不改变 readback 映射，也不把 readonly summary 提升为 runtime proof。

ExecutionContext runtime diagnostic 与 Engine deployment 的 snapshot/summary 也已按顶层类型分文件：
`TensorRtExecutionContextRuntimeDiagnosticSummary` 与 `TensorRtEngineDeploymentSummary` 分别进入同名源码文件。
runtime-diagnostic summary 保留 callback status 与 allocator/debug-listener 复制状态，engine summary 保留 tensor、
profile tensor value 与 memory/weight metadata 计数；两者仍是 pointer-free diagnostics，不替代真实 enqueue/readback proof。

已验证 `TensorRtSmokeRunner` 的 TensorRT 11 + CUDA 12.9 路径，输出包含 `ConfigSnapshot`、`EngineSnapshot`、`ContextSnapshot`、`EngineProfileTensorValuesV2`、`InspectorErrorRecorder=False->False`，且 `HighLevelChain11=True`。`CudaSmokeRunner`、TRT11 package consumer smoke、TRT11/TRT10/TRT8 lifecycle、DocFX 0 warning、workflow contract 和 generator deterministic gate 均通过。`buildSerializedNetwork(..., kernelText)` 仍是本机 vendor/API 边界：native/managed 已接线，但 TensorRT 11.0 + CUDA 12.9 返回空 TensorRT object，不伪造成成功。

## TRT11 第十四批状态

当前 manifest 总 API 记录为 `1428`，其中 CUDA=`104`、TensorRT=`1313`。TensorRT 分线覆盖为 TRT8=`359`、TRT10=`358`、TRT11=`595`。

第十四批 TRT11 新增 14 条部署关键 API，覆盖 direct engine build、serialized network with kernel text 边界、HostMemory data type、optimization profile shape-values V2、builder config clearFlag/plugin serialization，以及 execution context tensor address/device memory/input consumed event/aux streams 清理控制。

已验证 `TensorRtSmokeRunner` 的 TensorRT 11 + CUDA 12.9 路径，输出包含 `DirectEngineIOTensors=2`、`PluginsToSerialize=True`、`HostMemory=.../Int8`、`ProfileShapeValuesV2=-1/0`、`Clears=Any:True/Input:True/Output:True/DeviceMemory:True/AuxStreams:True`。`buildSerializedNetwork(..., kernelText)` 已完成 native/managed 接线，但当前本机 TensorRT 11.0 + CUDA 12.9 返回空 TensorRT object，因此记录为 vendor/API 边界，不伪造成成功。`CudaSmokeRunner`、TRT11 package consumer smoke、TRT10/TRT8 lifecycle、DocFX 0 warning、workflow contract 和 generator deterministic gate 均通过。

## 优先级

当前优先补齐部署、推理、内存和同步相关 API：

- TensorRT logger / runtime / builder / config / network / optimization profile
- TensorRT ONNX parser / serialized engine / engine metadata / execution context / inspector
- CUDA device / stream / event / memory / pinned memory / async copy / error mapping

不追求一次性手写全量 API。新增 API 优先进入 manifest，再由 generator 生成 native entrypoint、managed NativeMethods 和候选报告。

## TensorRT 11 状态

TensorRT 11 已进入 Windows 构建和 runtime 包矩阵。`trt11.0-cuda12.9-cudnn9.22` 已完成最小真实链路，并开始扩展部署关键 API：

- optimization profile
- builder config
- network input/output metadata
- engine tensor metadata
- execution context tensor shape/address/enqueue
- ONNX parser
- engine inspector

`trt11.0-cuda13.2-cudnn9.22` 当前可编译，但受本机驱动 CUDA 能力限制，运行 smoke 仍保持 pending。

## TRT11 第十二批状态

当前 manifest 总 API 记录为 `1397`，其中 CUDA=`104`、TensorRT=`1282`。TensorRT 分线覆盖为 TRT8=`359`、TRT10=`358`、TRT11=`564`。

第十二批 TRT11 重点继续追平部署关键 API：新增 39 条 manifest/native API，覆盖 runtime DLA / max threads / tempfile / temp directory / host-code / error-recorder 控制、engine serialization、serialization config、runtime config、execution context allocation strategy，以及 refitter async / max threads / weights validation / named weights metadata。

已验证 `TensorRtSmokeRunner` 的 TensorRT 11 + CUDA 12.9 路径：builder config runtime controls、network debug markers、engine inspector layer/context/error-recorder diagnostics、execution context address/output-allocator/temp-allocator/debug-listener/profiler/runtime-config/NVTX/aux-stream/unfused-debug diagnostics 均进入真实输出。`NetworkTrt11ModernLayerMetadataRunner` 已保持 7/7 modern layer probe，并在 cumulative probe 中验证 shape-output marker 路径。TRT11 runtime package consumer smoke 也通过，native assets 为 19/19。`NetworkTrt11AdvancedLayersSmokeRunner` 仍是 9/9 advanced metadata runner，但本轮在本机被 WDAC 策略阻止加载；文件签名有效，因此按环境策略阻塞记录，不伪造成代码失败。TensorRT 11 已移除或替换的旧 API，例如 tensor dynamic range accessor、calibration profile accessor、builder platform capability 和 `allInputShapesSpecified`，继续作为明确版本边界处理；高层 readiness 会对 `allInputShapesSpecified` 的移除做兼容诊断，不伪造原生 API 成功。

## CUDA 12.9 状态

CUDA `12.9` 已安装在：

- `%CUDA_PATH_V12_9%`

目标为 `cuda12.9` 的组合必须使用 CUDA `12.9`。之前 CUDA `12.3` 的临时 fallback 已废弃。

## 验证要求

Windows API 扩展批次应保持以下基线不回退：

- `CudaSmokeRunner`
- `TensorRtSmokeRunner trt10 + cuda11`
- `TensorRtSmokeRunner trt8 + cuda11`
- `LifecycleSmokeRunner trt10 + cuda11`
- `LifecycleSmokeRunner trt8 + cuda11`

TRT11 扩展批次还应尽量增加：

- `TensorRtSmokeRunner trt11 + cuda12.9`
- TRT11 runtime package consumer smoke
- deployment metadata / ONNX parser / execution context 相关 smoke

## Smart App Control / WDAC

如果 Windows 应用控制策略阻止未签名的 `jyppxtrtbridge.dll` 或 Debug sample/test 程序集，并返回 `0x800711C7`，不要修改测试断言绕过。当前本地验证策略是：

- 使用 `eng/Sign-WindowsBridgeBinaries.ps1` 签名 Windows bridge 输出。
- 使用 `eng/Sign-WindowsManagedBinaries.ps1 -Configuration Debug` 签名托管 sample/test 输出。
- 如果 Debug 输出仍被策略阻止，构建 Release sample 并使用 `Invoke-WindowsLifecycleSmoke.ps1 -Configuration Release` 记录验证结果。

这些签名脚本只用于本地开发验证，不等价于正式公开发布签名策略。
## 当前最新状态补充（2026-06-08）

当前开发节奏继续以 Windows API 完整化为主，Linux 打包与 runner handoff 继续保留结构但暂不深挖。本轮 TRT11 第十二批将 manifest API 总数推进到 `1397`，其中 Common=`11`、CUDA=`104`、TensorRT=`1282`；TensorRT 分线覆盖为 TRT8=`359`、TRT10=`358`、TRT11=`564`。

本轮在上一批运行期诊断基础上继续大规模追平 TRT11 部署控制 API，新增 runtime DLA/thread/tempfile/temp-directory/host-code/error-recorder 控制、engine serialization/runtime-config/context-allocation、serialization-config flags、runtime-config allocation strategy、refitter async/thread/weights-validation/named-weights metadata 等 39 条 API。`TensorRtSmokeRunner` 已在 TensorRT 11.0 + CUDA 12.9 下跑通新增高层探针；CUDA PowerShell smoke、TRT10/TRT8 lifecycle smoke、DocFX 0 warning、workflow contract 和 generator deterministic gate 也继续通过。直接运行 `CudaSmokeRunner.dll` 被本机 WDAC 策略以 `0x800711C7` 阻止，本轮记录为环境策略阻塞。

本轮扩展 `NetworkTrt11AdvancedLayersSmokeRunner`，覆盖 cast、non-zero、ragged softmax、NMS、reverse sequence、einsum、loop control-flow、if conditional、fill int64 九类部署相关 probe，当前 9/9 通过。新增的 C# public API 已放在 `TensorRtNetworkDefinition`、`TensorRtLayer`、`TensorRtEngine`、`TensorRtExecutionContext`、`TensorRtBuilderConfig` 等 partial 文件中，并补充中英文 XML 注释。
## TRT11 第十六批状态

当前 manifest 总 API 记录为 `1470`，其中 CUDA=`104`、TensorRT=`1355`、Common=`11`。TensorRT 分线覆盖为 TRT8=`359`、TRT10=`358`、TRT11=`637`。

第十六批新增 34 条 TRT11 部署关键 API，覆盖 direct tensor network input/output 角色查询，以及 layer input/output tensor 槽位元数据：tensor 存在性、名称、数据类型、shape、rank、维度 extent、维度名、location、allowed formats、shape/execution tensor 角色、network input/output 角色、动态维度标记和 native compact summary。

C# 高层新增 `TensorRtLayerTensorMetadata`，并提供 `TensorRtLayer.GetInputTensorMetadata`、`TensorRtLayer.GetOutputTensorMetadata`、`TensorRtLayer.GetInputTensorSummary`、`TensorRtLayer.GetOutputTensorSummary`、`TensorRtTensor.IsNetworkInput`、`TensorRtTensor.IsNetworkOutput`。这些 API 都不向普通用户暴露裸 `IntPtr`。当 TensorRT 11 现代层在构建期返回 unknown-rank tensor 时，`Shape` 会被置为 `null`，summary 中记录 `shape=unknown-rank`，以免元数据查询误伤部署诊断链路。

已验证 `TensorRtSmokeRunner` 的 TensorRT 11 + CUDA 12.9 路径，输出包含 `LayerTensorMetadata`、native slot summaries 和 direct tensor roles；`NetworkTrt11ModernLayerMetadataRunner` 在 TensorRT 11.0 + CUDA 12.9 下 7/7 modern-layer metadata probe 通过。`CudaSmokeRunner`、TRT10/TRT8 lifecycle、DocFX 0 warning、workflow contract、project quality tests 和 generator deterministic gate 均通过。
# 2026-06-08 CUDA 部署关键 API 批量补齐

当前路线仍以 Windows API 完整化优先。Linux 构建、打包和 handoff 结构继续保留，但不再作为当前节奏主线。

本轮将 manifest 总数推进到 `1534`，其中 `common=11`、`cuda=125`、`tensorrt=1398`。接口覆盖矩阵已刷新，CUDA 11.6、11.8、12.1、12.3、12.9、13.2 的启发式命中数从此前的 `64` 提升到 `75-78`。

新增 CUDA 部署关键能力包括：device cache/shared-memory config、PCI bus id 查询与反查、device runtime flags、P2P attribute、stream id、stream attribute copy、thread capture mode exchange、event record-with-flags、CUDA graph create/clone/node/root/edge count、graph exec flags 和 graph exec upload。

`cudaStreamGetDevice` 在当前 Windows CUDA 12.9 头文件中可见，但本机 cudart import library 不提供可链接符号，因此桥接层保留入口并返回明确的 `NotSupported`，不伪造成已支持。

C# 高层已同步扩展 `CudaDevice`、`CudaStream`、`CudaEvent`、`CudaGraph`、`CudaGraphExec`，普通用户仍不需要直接接触裸 `IntPtr`。`CudaSmokeRunner` 和 `CudaGraphSmokeRunner` 已输出新增 API 的运行证据。

