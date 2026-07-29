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

剩余两份高层大 wrapper 现已沿用同一 feature 边界。`Layers/TensorRtLayer.cs` 只保留构造、通用 layer metadata、
owner lease、output index 校验、释放与共享 Dims 校验；Shuffle、MatrixMultiply、Reduce、SoftMax、Unary、TopK、
Gather、ElementWise、Activation、Pooling、Convolution、Scale、Padding、Resize、Concatenation、Slice、Fill 的
118 个方法进入 17 份 feature partial。`Network/TensorRtNetworkDefinition.cs` 只保留 input/output/layer ownership、
output mark/unmark、释放与共享 tensor 校验；对应 21 个 `Add*` 方法进入 21 份 feature partial。
`ManagedWrapperFeatureLayoutTests` 固定每份文件的精确方法集合，并按原顺序重组规范化源码；拆分前两份 Git blob
分别为 `faa5fa87fb24247a86f9d166073cf3858bad9ac2` 与 `933e7b253ecfefa00034da2544912b6e20c353ae`。
这次仅整理源码，不改变 public 签名、SafeHandle/owner-lease 行为、line 路由、校验顺序、native entrypoint、
generated binding、manifest 或 ABI 证据。

通用 Engine 与 ONNX Parser wrapper 也沿用同一模式。`Engine/TensorRtEngine.cs` 现为 124 行 handle、标量属性与
Dispose core；36 个 tensor/profile metadata、binding report、execution-context、refit 与 inspection 方法进入 5 份
feature partial，两个 binding-report 私有 helper 随 `TensorRtEngine.BindingReports.cs` 迁移。
`Parsing/TensorRtOnnxParser.cs` 现为 198 行构造、logger/config/initializer lifetime、标量属性、Dispose 与共享校验
core；32 个 model parsing/loading、TryParse、diagnostics、operator-support 与 flags 方法进入 6 份 feature partial。
Parser flag 校验与 model segment/stream copy helper 被多个 partial 共同消费，因此继续留在 core。
`ManagedEngineParserFeatureLayoutTests` 固定方法/helper 归属，并重组拆分前 Git blob
`fd6907a9e03eab3b6f9a1b5820eea9e6e1e82ea9` 与 `d8e3f135b71f9b2fd893146776da7d538db5d020`。

BuilderConfig 与 ExecutionContext 的通用 wrapper 也已拆分，且不与既有 TRT11 partial 重叠。
`Builder/TensorRtBuilderConfig.cs` 现为 102 行 handle、progress-monitor lifetime、共享校验与 Dispose core；
34 个 profile/flag/compatibility/layer-device/memory-pool/scalar/tactic/timing-cache 方法和 7 个 feature 属性进入
8 份 partial。`ValidateLayer`、disposed-state 与 progress-monitor helper 仍被既有 diagnostics partial 消费，因此留 core。
`Execution/TensorRtExecutionContext.cs` 现为 176 行 metadata、profiler/aux-stream lifetime、cleanup 与 Dispose core；
20 个 shape/address/device-memory/event/enqueue 方法进入 5 份 feature partial。布局门禁可重组拆分前 Git blob
`592ce09c4da5fb4f7a376800cd5a6b309a22481b` 与 `1614e46a4b0175ff9889302f977a0520be404b29`。

ParserRefitter 与高层 InferenceBindings 也按调用阶段拆分。`Parsing/TensorRtOnnxParserRefitter.cs` 现为 114 行
native owner、refitter/logger borrower lifetime、initializer pin lifetime、共享 model segment/stream copy helper 与 Dispose core；
21 个 refit/model-proto/initializer/diagnostic 方法进入 4 份 feature partial。`Inference/TensorRtInferenceBindings.cs`
现为 124 行 engine/context 引用、buffer owner、共享 tensor lookup/report refresh、Dispose 与 disposed-state core；
14 个 geometry/buffer/host-transfer/address-binding/execution/diagnostic 方法进入 6 份 feature partial，feature 专属的
size estimation、buffer replacement 与 readiness helper 跟随各自 owner。布局门禁可重组拆分前 Git blob
`07975ca6274fb64fcce06ca6e967515f07aa734c` 与 `6257b4c8ad3c0f8c4ec349208e8583b670359d0a`。

环境探测与 plugin inventory 也已消除单文件多职责。`Diagnostics/TensorRtEnvironmentProbe.cs` 从 1,497 行降为
62 行跨 feature 异常分类、诊断格式化与 stage helper core；43 个公开静态入口进入 PluginInitialization、
RuntimeMetadata、GlobalPluginRegistry、BuilderPluginRegistry、DependencyProbes、ObjectCreation、BuildChains 七份 partial。
`Plugins/TensorRtPluginRegistryInventory.cs` 只保留主 inventory 聚合类；两个 enum、field/creator copied metadata、
creator/field summary 与 inventory diagnostics 分别进入 6 份独立文件。布局门禁可重组拆分前 Git blob
`bcd1301777f81d9de32625b4c7be952de3126d92` 与 `c39d9ec853987f50f452b0dfebc2658120aa95d2`。

CUDA 高层 graph 与 device-memory wrapper 也按相同的 feature owner 规则归类。`Graphs/CudaGraph.cs` 从 1,630 行降为
206 行 graph handle、capture/conditional/allocation owner 生命周期、共享校验与 Dispose core；conditional handle、
graph composition、node creation、topology diagnostics、node inspection、node mutation、node relations 与 instantiation
进入 8 份 feature partial。`Memory/CudaMemory.cs` 从 951 行降为 114 行 allocation handle、IPC import metadata、共享
range 校验与 Dispose core；IPC、range diagnostics、async allocation、fill、prefetch/advice、host transfer、device
transfer、async free 与 array conversion 进入 9 份 feature partial，并继续保留既有 registered-host partial。布局门禁
可重组拆分前 Git blob `5b7e06e4e59d6961c9c848f88d1f9ace6a9c0450` 与
`1d2085d8ffc527cfd280c2ba68836d14bb2c6334`。

CUDA device 级 helper 与跨模块 enum 也不再集中在两个宽泛文件中。`Devices/CudaDevice.cs` 从 793 行降为 144 行
runtime/driver/device identity 与 property snapshot core；graph resources、runtime configuration、initialization/selection、
peer capabilities、memory pools、cache/RDMA、synchronization/error diagnostics 进入 7 份 feature partial。原 912 行
`Core/CudaFlags.cs` 已删除，其中 24 个 public enum 进入 Streams、Events、Memory、Devices、Graphs 下 10 个模块文件；
enum 名称、底层类型、数值与 XML 注释保持不变。布局门禁可重组拆分前 Git blob
`df16e51427f82c9b99fa867819015539dd65ac0c` 与 `dca1aa594002506ce47bd247f47141201af6591d`。

Pitched 与 CUDA array memory owner 也已按传输维度拆分，同时把共享校验保留在 owner core。
`Memory/CudaPitchedMemory.cs` 从 678 行降为 181 行 allocation/metadata、共享 pitch/extent/pinned-buffer validation 与
Dispose core；fill、2D transfer、3D transfer、array conversion 进入 4 份 feature partial。`Memory/CudaArray.cs` 从
609 行降为 208 行 allocation/metadata、共享 validation 与 Dispose core；复制型 requirements/sparse diagnostics、
1D/2D/3D transfer、array conversion 进入 5 份 feature partial。布局门禁可重组拆分前 Git blob
`e30004cce7cc55c7b62e19478de913d68be3c591` 与 `6de4bc82180a8539e3b6d6fa610c485c042d043e`。

CUDA stream 与 memory-pool owner 也按调用阶段和职责边界拆分。`Streams/CudaStream.cs` 从 439 行降为 126 行
handle/property、capture owner-count 与 Dispose core；一般 diagnostics、capture diagnostics、capture dependencies、
synchronization/event 和 capture lifecycle 进入 5 份 feature partial。`Memory/CudaMemoryPool.cs` 从 372 行降为
31 行 handle/value core；factory、allocation、access 与 attributes 进入 4 份 feature partial，独立 owner
`CudaOwnedMemoryPool` 与两个 pool enum 各自移入专用文件。布局门禁可重组拆分前 Git blob
`379fd60f08beca36711507a6c83a2192d20b6562` 与 `7028220b628b3d2af9c3f007dda75ec4ff2f5fd3`。

Tools 的 ONNX engine 构建服务也按构建阶段归类。`Build/OnnxEngineBuildService.cs` 从 3,214 行降为 540 行
selected-device thread、dry-run/build orchestration core；builder configuration、timing cache、result creation、refit、
diagnostics、runtime execution、benchmarking、runtime inputs、reference validation 与 deployment configuration 进入
10 份 feature partial，private lease/worker/runtime state type 跟随各自 feature。原 970 行
`Build/OnnxEngineBuildResult.cs` 降为 304 行主 result；timing-cache artifact、capability probe、loaded-engine diagnostics、
preflight metadata、model evidence 与 benchmark summary 六个独立 public 类型各自成文件。布局门禁可重组拆分前 Git blob
`31f2c170c9c74b7278b4bc266eca76405dc33067` 与 `97f6e992fe582513fcf77b10ce05de4de32af1a5`。

手写 TensorRT partial interop 也开始按相同职责模块归类：

- `Internal/Interop/Builder`：builder creation/capabilities、serialized build outputs、builder boundary controls、
  timing-cache lifecycle/TRT11 controls、builder-config core/scalar controls、diagnostics、plugin serialization 与 runtime controls；
  `ControlFlow`：loop/conditional 操作。
- `Internal/Interop/Callbacks`：allocator dry-run、callback interface/state 复制、logger/profiler/progress-monitor delegate
  签名与 managed callback diagnostics。
- `Internal/Interop/Diagnostics`：复制型 error-code metadata 与仅由 environment probe 使用的跨版本/TRT11 build-chain probes；
  `Internal/Interop/Interfaces`：owner-scoped versioned-interface metadata 复制。
- `Internal/Interop/Engine`：core/deployment engine metadata、inspector lifecycle/information/boundary/diagnostics、
  tensor/profile copied values、Dims64、error-recorder controls 与 weight-streaming/stat runtime controls；`Execution`：
  execution-context core creation、binding/address/enqueue/aux-stream controls、boundary controls、copied engine metadata、runtime-config 创建、
  allocation-strategy、deployment metadata、Dims64、diagnostics 与 allocator/event presence controls。
- `Internal/Interop/Inference`：同步 execute/enqueue 操作；`Weights`：复制型 layer-weight metadata。
- `Internal/Interop/Layers`：identity/constant/convolution/deconvolution/scale/padding/element-wise/matrix-multiply/shuffle/reduce、
  softmax、unary、TopK、gather feature、
  activation、pooling、LRN feature、共用 optional-weight helper、quantization、attention、fill-int64、
  resize、concatenation、slice feature、兼容/部署型 layer attributes、Dims64、tensor metadata、
  shape、select、fill feature、core layer metadata、transformer 与 RNNv2 操作；
  `Network`：core definition input/output/name/flags、network boundary controls、部署型 network
  layer 创建、core tensor name/type/shape/range metadata、tensor/network Dims64、debug/shape diagnostics、
  refittable-weight 标记与 safe network-v2 操作。
- `Internal/Interop/Parsing`：global ONNX parser version、parser lifecycle/input/diagnostics/flags、legacy parser diagnostics、
  ONNX config/model buffer/support、builder-config attachment、layer-output metadata、weight-descriptor parsing、
  parser-refitter diagnostics 与共用复制字符串 helper。
- `Internal/Interop/Plugins`：plugin initialization、global/builder/runtime registry inventories，以及复制型 V2/V3 layer
  metadata/query snapshot。
- `Internal/Interop/Profiles`：optimization-profile core creation/shape controls、Dims64 与 shape-value 查询。
- `Internal/Interop/Runtime`：adapter information、runtime creation diagnostics、跨版本 line-binding helper、
  global runtime version/logger probes、runtime deployment controls 与复制型 diagnostics；
  `Serialization`：engine deserialization/serialization、serialization-config flags 与 host-memory buffer/metadata；`Refit`：async refit、weights/dynamic-range、
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
根 `NativeBridgeApi.cs` 开始按小批 owner/feature 区段持续瘦身：跨版本 timing-cache create/set/serialize 片段已移入
`Builder/NativeBridgeApi.TimingCacheLifecycle.cs`，将该片段插回原位置后必须恢复拆分前的根文件 Git blob。
根文件的 ONNX parser core 已拆为 Parsing lifecycle/input、diagnostics、flags/operator-support 与仅含字符串读取 helper
的 Shared partial。parser 专属 diagnostic helper 随 diagnostics 移动，delegate 与复制字符串 allocator 继续由 parser、
parser-refitter、support 共用；按原片段顺序重组后必须恢复拆分前的根文件 Git blob。
根文件尾部 owner 区段已拆为 Engine inspector core、ExecutionContext binding/enqueue、Serialization host-memory buffer 与
Engine core metadata partial。engine-information 与 IO-tensor-name getter helper 随 owner 移动。BuilderConfig bit-flag helper
最初继续留在根文件；Tensor/Layer name helper 随后与 core owner partial 一起移动，且仍要求按原顺序重组。
跨版本 line-binding delegate、私有 bindings class 与版本路由已移入 helper-only Runtime partial；六个 environment-probe
操作及 minimal build-chain helper 已移入 Diagnostics。生成的 bindings/helper 仍消费同一 partial 私有类型，按原顺序
重组后必须恢复拆分前根文件 blob。
根文件的 Network definition core 已移入 `Network/NativeBridgeApi.NetworkCore.cs`，包含 input/output ownership、layer
lookup、name/flags metadata 与专属 name getter。Layer creation 最初留在根文件等待按 feature 拆分；optional-weight helper
继续随 weighted-layer 消费方法保留。
Identity/constant、convolution、deconvolution、scale creation 已分别移入 Layers feature partial。Scale 保留仅供自身使用
的 data-type selector；convolution/deconvolution/scale 共用的 pin/validation helper 进入 helper-only Layers Shared partial。
重组时必须保持原 helper 顺序。
Padding、element-wise、matrix-multiply creation/attributes 已移入三个独立 Layers feature partial，且这些区段没有私有
helper。Shuffle creation、reshape/transpose attributes 与 zero-placeholder controls 已移入第四个 partial，该区段同样没有
私有 helper，且仍要求按原顺序直接重组。Reduce creation 与 readonly operation/axes/keep-dimensions attributes 也已移入
另一个无 helper 的 partial。
SoftMax、unary、TopK、gather creation/attributes 已移入四个独立的无 helper partial。Activation、pooling、LRN
creation/attributes 已移入另外三个独立的无 helper partial。Resize、concatenation、slice creation/attributes 已移入
另外三个无 helper partial，resize 的数组 pinning 留在 feature 文件内。Shape、select、fill creation/attributes 已移入
三个无 helper feature partial；根文件随后从 `GetLayerOutput` 进入通用 Layer metadata 区段。
通用 Layer metadata 已连同 `MapLayerType` 和专属 name getter 移入 `Layers/NativeBridgeApi.LayerCoreMetadata.cs`；通用
Tensor metadata 与专属 name getter 移入 `Network/NativeBridgeApi.TensorCoreMetadata.cs`。
剩余 adapter/callback/runtime/builder/network/serialization/execution/profile/config 实现已移入十份 owner partial，其中
`Builder/NativeBridgeApi.BuilderConfigCore.cs` 接管 bit-flag helper。根 `NativeBridgeApi.cs` 现为 13 行声明 shell，不含
public/private static 实现；将十个主体插回 shell header 与 close brace 之间即可恢复拆分前 root blob。

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
