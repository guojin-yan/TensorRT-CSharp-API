# 源码模块化组织

Windows API 完整化阶段已经把源码模块化作为质量门禁，而不是单纯的可读性优化。

## 原生桥接层

过大的原生桥接文件需要按部署功能区拆分，同时保持 C ABI 导出名和行为不变。

- CUDA 模块放在 `native/src/cuda/modules`。
- TensorRT 8 模块放在 `native/src/tensorrt/v8/modules`。
- TensorRT 10 模块放在 `native/src/tensorrt/v10/modules`。

当前第一批已抽出的原生分组：

- CUDA pitched memory 分配与 2D copy helper。
- TensorRT convolution / scale / padding layer helper。
- TensorRT execution-context 部署 helper。

这些模块当前仍由原来的 `api.cpp` include 进同一个翻译单元，而不是直接拆成独立 `.cpp`。这样可以继续使用既有匿名命名空间 helper，避免因为移动文件导致 ABI 或行为变化。只有当校验、所有权、版本守卫等 helper 都提升为可复用头文件后，模块才适合进一步变成独立 `.cpp`。

## 托管 interop 层

托管 interop 在 wrapper 分组稳定后按功能区拆分。

当前第一批托管拆分是 CUDA pitched-memory interop：

- `src/JYPPX.CudaSharp/Internal/Interop/Memory/NativeCudaApi.PitchedMemory.cs`

其余手写 CUDA partial interop 也按公开 owner 的 feature area 归类：

- `Internal/Interop/Devices`：device resource 与 primary execution-context 操作。
- `Internal/Interop/Diagnostics`：复制型 runtime logs。
- `Internal/Interop/Drivers`：可选 Driver capability、module 与 typed launch 操作。
- `Internal/Interop/IPC`：owner-safe export/import token 操作。
- `Internal/Interop/Kernels`：Runtime kernel-library owner 与 launch 操作。
- `Internal/Interop/RuntimeCompilation`：可选 NVRTC program 操作。

`NativeCudaApi.Deployment.cs` 仍保留在 interop 根目录，因为它同时跨越 error、PCI、stream/event、pinned-memory、atomic capability 与 device selection；应在单独的行为拆分批次中处理，不能错误归入某一个 feature。

TensorRT 高层 wrapper 也开始按 layer feature 拆分：

- `src/JYPPX.TensorRtSharp/Network/TensorRtNetworkDefinition.Deconvolution.cs`
- `src/JYPPX.TensorRtSharp/Network/TensorRtNetworkDefinition.Lrn.cs`
- `src/JYPPX.TensorRtSharp/Network/TensorRtNetworkDefinition.Quantization.cs`
- `src/JYPPX.TensorRtSharp/Layers/TensorRtLayer.Deconvolution.cs`
- `src/JYPPX.TensorRtSharp/Layers/TensorRtLayer.Lrn.cs`
- `src/JYPPX.TensorRtSharp/Layers/TensorRtLayer.Quantization.cs`

手写 TensorRT partial interop 也开始按相同职责模块归类：

- `Internal/Interop/Builder`：build outputs、builder boundary controls、timing-cache、builder-config diagnostics、
  plugin serialization 与 runtime controls；`ControlFlow`：loop/conditional 操作。
- `Internal/Interop/Callbacks`：allocator dry-run、callback interface/state 复制，以及 logger/profiler/progress-monitor
  delegate 签名。
- `Internal/Interop/Diagnostics`：复制型 error-code metadata 与仅由 environment probe 使用的 TRT11 build probes；
  `Internal/Interop/Interfaces`：owner-scoped versioned-interface metadata 复制。
- `Internal/Interop/Engine`：engine/inspector boundary controls、engine/tensor/profile copied metadata 与 values、Dims64、
  engine-inspector diagnostics/error-recorder controls 与 weight-streaming/stat runtime controls；`Execution`：
  execution-context address/aux-stream controls、boundary controls、copied engine metadata、runtime-config 创建、
  allocation-strategy、deployment metadata、Dims64、diagnostics 与 allocator/event presence controls。
- `Internal/Interop/Inference`：同步 execute/enqueue 操作；`Weights`：复制型 layer-weight metadata。
- `Internal/Interop/Layers`：quantization、attention、fill-int64、兼容/部署型 layer attributes、Dims64、tensor metadata、
  transformer 与 RNNv2 操作；`Network`：network boundary controls、部署型 network layer 创建、tensor/network Dims64、
  debug/shape diagnostics、refittable-weight 标记与 safe network-v2 操作。
- `Internal/Interop/Parsing`：global ONNX parser version、legacy parser diagnostics、ONNX config/model buffer/support、
  builder-config attachment、layer-output metadata、weight-descriptor parsing 与 parser-refitter diagnostics。
- `Internal/Interop/Plugins`：plugin initialization、global/builder/runtime registry inventories，以及复制型 V2/V3 layer
  metadata/query snapshot。
- `Internal/Interop/Profiles`：optimization-profile Dims64 与 shape-value 查询。
- `Internal/Interop/Runtime`：global runtime version/logger probes、runtime deployment controls 与复制型 diagnostics；
  `Serialization`：engine serialization、serialization-config flags 与 host-memory metadata；`Refit`：async refit、weights/dynamic-range、
  entry metadata 与 refitter diagnostics。

`NativeBridgeApi.GlobalProbeShared.cs` 仅为拆分后的 global Runtime、Parsing、Plugins partial 保存共用的 unsupported-line
异常 helper，因此继续保留在 interop 根目录；它不包含公开方法。
`NativeBridgeApi.OwnerErrorRecorderSnapshotShared.cs` 仅为 Builder、EngineInspector、Execution 与 Network owner 保存
复制型 error-recorder snapshot mapper，因此继续保留根目录；它不包含公开方法。

只有当 version-prefixed 文件的方法集合仍跨 owner/feature 且需要单独行为拆分时才保留根目录；不会仅依据文件名前缀分类。
原 `Trt11DeploymentAdditions` 已按方法 owner 拆为 `Network/NativeBridgeApi.DeploymentNetworkLayers.cs` 与
`Layers/NativeBridgeApi.DeploymentLayerAttributes.cs`；两部分重组后的 Git blob 必须与拆分前原文件一致。
原 `Trt11RuntimeSerializationRefit` 也已拆入 `Runtime`、`Serialization`、`Execution` 与 `Refit`；四部分按原片段顺序
重组后必须恢复拆分前 Git blob。
原 `DeploymentMetadata` 已拆入 `Engine`、`Execution`、`Layers`、`Refit`；仅 delegate 与跨 owner 私有 helper 保留为
根目录 `NativeBridgeApi.DeploymentMetadataShared.cs`，五部分按原片段顺序重组后必须恢复拆分前 Git blob。
原 `Trt11Dims64` 已拆入 `Network`、`Engine`、`Execution`、`Profiles` 与 `Layers`；layer getter delegate/helper 随
`Layers` 移动，五部分按原片段顺序重组后必须恢复拆分前 Git blob。
原 `Trt11RuntimeControls` 已按连续 owner 区段拆入 `Engine`、`Execution` 与 `Builder`；三部分按原片段顺序重组后
必须恢复拆分前 Git blob。
原 `Trt11Diagnostics` 已拆入 `Builder`、`Network`、`Engine` 与 `Execution`；四部分按原片段顺序重组后必须恢复
拆分前 Git blob。
原 `GlobalRuntimePluginProbe` 已拆入 `Runtime`、`Parsing`、`Plugins` 与仅含 helper 的根目录 Shared partial；四部分
按原片段顺序重组后必须恢复拆分前 Git blob。
原 `SafeDeferredUplift` 已拆入 `Plugins` 与 `Parsing`；两部分按原片段顺序重组后必须恢复拆分前 Git blob。文件归类
不会改变既有 deferred 历史，也不等于 plugin/parser runtime 与 lifetime proof。
原 `Trt11BoundaryControls` 已拆入 `Builder`、`Engine`、`Execution`、`Network` 与仅含 helper 的根目录 Shared partial；
五部分按原片段顺序重组后必须恢复拆分前 Git blob。
原 `Trt11FourteenthBatch` 已拆入两个 `Builder` feature 文件以及 `Serialization`、`Profiles`、`Execution`；serialized-network
结果 struct 随 Builder build-output 方法移动，五部分按原片段顺序重组后必须恢复拆分前 Git blob。
原 `Trt11FifteenthBatch` 已拆入两个 `Engine` feature 文件与 `Execution`；profile-value validation helper 随 Engine profile
方法移动，三部分按原片段顺序重组后必须恢复拆分前 Git blob。

生成文件继续保留在各自 `Generated` 文件夹下，不手工拆分。若需要调整生成文件布局，必须通过 generator 本身完成，并通过生成器确定性门禁。

## 托管公开 API 目录

公开 API 文件按职责放入模块目录，文件移动不改变现有 namespace、类型名或 public API。SDK 风格项目会递归编译这些目录，因此项目文件不需要维护逐文件 `Compile` 清单。

| 项目 | 模块目录 |
| --- | --- |
| `JYPPX.CudaSharp` | `Core`、`Devices`、`Diagnostics`、`Drivers`、`Events`、`Graphs`、`IPC`、`Kernels`、`Memory`、`RuntimeCompilation`、`Streams` |
| `JYPPX.TensorRtSharp` | `Builder`、`ControlFlow`、`Core`、`Diagnostics`、`Engine`、`Execution`、`Inference`、`Interfaces`、`Layers`、`Network`、`Parsing`、`Plugins`、`Profiles`、`Refit`、`Runtime`、`Serialization`、`Weights` |
| `JYPPX.TensorRtSharp/Callbacks` | `Core`、`Debugging`、`MemoryAllocation`、`Monitoring` |
| `JYPPX.TensorRtSharp.Tools` | `Artifacts`、`Build`、`Core`、`Refit`、`Runtime`、`Trtexec` |

当前整理覆盖三个项目原根目录中的 294 个 `.cs` 文件。`ManagedSourceModuleLayoutTests` 会验证项目根目录不再堆放公开 API 源文件，并检查所有约定模块至少包含一个源码文件。

CUDA Driver capability 入口、复制型 module owner 和 typed launch owner 统一放入 `JYPPX.CudaSharp/Drivers`。`Kernels` 则继续负责 CUDA Runtime kernel-library owner，以及 typed argument/configuration 值对象。

TensorRT 复制型 versioned-interface metadata 与 owner-scoped metadata query 统一放入 `JYPPX.TensorRtSharp/Interfaces`。托管 weights payload、复制型 weights metadata 与 refit weights role 放入 `JYPPX.TensorRtSharp/Weights`；`Core` 只保留共享异常、维度、枚举与库信息。

ONNX stripped-plan refit 生命周期与持久化 plan 重载 snapshot 统一放入 `JYPPX.TensorRtSharp.Tools/Refit`。`Build` 模块继续负责 build options、service、diagnostics 与 result，并消费这些复制型证据模型。

## 规则

- 模块化时不得改名 C ABI 导出入口。
- 不得为了移动代码改变 manifest 语义。
- 不得在源码整理过程中向普通 C# 用户暴露裸 `IntPtr`。
- 新增公开 API 时应选择已有职责模块；只有形成独立职责边界时才增加新目录。
- 每次拆分后都必须回归原生构建、托管构建和 binding generator 确定性验证。
