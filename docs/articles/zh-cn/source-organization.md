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

TensorRT 高层 wrapper 也开始按 layer feature 拆分：

- `src/JYPPX.TensorRtSharp/Network/TensorRtNetworkDefinition.Deconvolution.cs`
- `src/JYPPX.TensorRtSharp/Network/TensorRtNetworkDefinition.Lrn.cs`
- `src/JYPPX.TensorRtSharp/Network/TensorRtNetworkDefinition.Quantization.cs`
- `src/JYPPX.TensorRtSharp/Layers/TensorRtLayer.Deconvolution.cs`
- `src/JYPPX.TensorRtSharp/Layers/TensorRtLayer.Lrn.cs`
- `src/JYPPX.TensorRtSharp/Layers/TensorRtLayer.Quantization.cs`

生成文件继续保留在各自 `Generated` 文件夹下，不手工拆分。若需要调整生成文件布局，必须通过 generator 本身完成，并通过生成器确定性门禁。

## 托管公开 API 目录

公开 API 文件按职责放入模块目录，文件移动不改变现有 namespace、类型名或 public API。SDK 风格项目会递归编译这些目录，因此项目文件不需要维护逐文件 `Compile` 清单。

| 项目 | 模块目录 |
| --- | --- |
| `JYPPX.CudaSharp` | `Core`、`Devices`、`Diagnostics`、`Events`、`Graphs`、`IPC`、`Kernels`、`Memory`、`RuntimeCompilation`、`Streams` |
| `JYPPX.TensorRtSharp` | `Builder`、`ControlFlow`、`Core`、`Diagnostics`、`Engine`、`Execution`、`Inference`、`Layers`、`Network`、`Parsing`、`Plugins`、`Profiles`、`Refit`、`Runtime`、`Serialization` |
| `JYPPX.TensorRtSharp/Callbacks` | `Core`、`Debugging`、`MemoryAllocation`、`Monitoring` |
| `JYPPX.TensorRtSharp.Tools` | `Artifacts`、`Build`、`Core`、`Runtime`、`Trtexec` |

当前整理覆盖三个项目原根目录中的 294 个 `.cs` 文件。`ManagedSourceModuleLayoutTests` 会验证项目根目录不再堆放公开 API 源文件，并检查所有约定模块至少包含一个源码文件。

## 规则

- 模块化时不得改名 C ABI 导出入口。
- 不得为了移动代码改变 manifest 语义。
- 不得在源码整理过程中向普通 C# 用户暴露裸 `IntPtr`。
- 新增公开 API 时应选择已有职责模块；只有形成独立职责边界时才增加新目录。
- 每次拆分后都必须回归原生构建、托管构建和 binding generator 确定性验证。
