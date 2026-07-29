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

- `Internal/Interop/Builder`：timing-cache 操作；`ControlFlow`：loop/conditional 操作。
- `Internal/Interop/Callbacks`：allocator dry-run、callback interface/state 复制，以及 logger/profiler/progress-monitor
  delegate 签名。
- `Internal/Interop/Diagnostics`：复制型 error-code metadata；`Internal/Interop/Interfaces`：owner-scoped
  versioned-interface metadata 复制。
- `Internal/Interop/Inference`：同步 execute/enqueue 操作；`Weights`：复制型 layer-weight metadata。
- `Internal/Interop/Layers`：quantization、attention、fill-int64、tensor metadata、transformer 与 RNNv2 操作；
  `Network`：safe network-v2 layer 创建操作。
- `Internal/Interop/Parsing`：legacy parser diagnostics、ONNX config/model buffer/support、builder-config attachment、
  layer-output metadata 与 parser-refitter diagnostics。
- `Internal/Interop/Plugins`：builder capability/runtime registry inventories，以及复制型 V2/V3 layer metadata/query snapshot。

`NativeBridgeApi.GlobalRuntimePluginProbe.cs` 仍保留在 interop 根目录，因为它混合了 global runtime version、logger、
ONNX parser version 与 plugin-registry 操作；应在单独的行为拆分批次中处理，不能标记为纯 plugin 文件。

`NativeBridgeApi.SafeDeferredUplift.cs` 也继续保留根目录，因为它混合 plugin initialization 与 ONNX weight-descriptor
parsing。callback 文件归类不等于 callback trampoline、lifetime 或 runtime proof。

当 version-prefixed 文件的方法集合跨越 builder、engine、execution-context、network 与 layer owner 时，仍保留根目录。
`Trt11Diagnostics`、`Trt11Dims64` 与 `Trt11RuntimeControls` 不会仅依据文件名前缀分类。

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
