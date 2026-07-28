# PluginCreatorV3 Metadata 安全设计门

## 目标

本设计门用于把 `IPluginCreatorV3One` 与 `IVersionedInterface` 的只读 metadata 从“直接 borrowed pointer 访问”收口到“Plugin Registry Inventory copied snapshot”安全路径。它不是 runtime proof，也不能替代 package-consumer-runtime 或 post-publish verification。

## 已覆盖接口

本阶段固定 6 个候选：

- `IPluginCreatorV3One::getPluginName`
- `IPluginCreatorV3One::getPluginVersion`
- `IPluginCreatorV3One::getPluginNamespace`
- `IPluginCreatorV3One::getFieldNames`
- `IPluginCreatorV3One::getInterfaceInfo`
- `IVersionedInterface::getInterfaceInfo`

这些接口的用户价值在于列出 plugin creator、核对 plugin 版本和命名空间、展示字段 schema、诊断 TensorRT plugin 可用性。安全 public surface 不返回 creator pointer，而是通过 `TensorRtPluginRegistryInventory`、`TensorRtPluginCreatorInfo`、`TensorRtPluginCreatorSummary` 和 `TensorRtPluginFieldSummary` 返回 managed snapshot。

## 安全输出模式

输出模式固定为：

`copied string, copied field metadata, and managed interface metadata snapshot`

边界要求：

- name/version/namespace 必须复制到 managed string。
- field name/type/length/hasData 必须复制到 managed record。
- interface metadata 只能作为 copied metadata/snapshot 进入 public API。
- public API 不暴露 `IntPtr`、`nint`、`IPluginCreator*`、`IPluginCreatorV3One*`。
- `createPlugin`、plugin clone/enqueue、plugin resource acquire/release 继续 deferred。

## 当前证据链

- native：`native/src/tensorrt/common/plugin_registry_inventory.inc`
- interop：`src/JYPPX.TensorRtSharp/Internal/Interop/NativeBridgeApi.PluginRegistryInventory.cs`
- high-level wrapper：`src/JYPPX.TensorRtSharp/Plugins/TensorRtPluginRegistryInventory.cs`
- design gate：`src/JYPPX.TensorRtSharp/Plugins/TensorRtPluginCreatorV3MetadataDesignGate.cs`
- smoke：`smoke/PluginRegistryInventorySmokeRunner/Program.cs`
- quality：`tests/JYPPX.ProjectQuality.Tests/PluginCreatorV3MetadataDesignGateTests.cs`

## 不可晋级项

下列内容不能因为本设计门存在而被视为完成：

- Plugin instance create/clone/enqueue。
- Plugin library load/deregister。
- Plugin resource acquire/release。
- Borrowed creator pointer public handle。
- Build-only、dry-run、readonly diagnostics、YoloVision matrix、OnnxToEngine report、TensorRtExec report。

## 下一步

下一阶段如果继续提升 plugin 相关能力，应优先补更强的 runtime smoke 与 package consumer proof：在真实 TensorRT 环境中加载 runtime/builder registry，读取 creator count、creator name/version/namespace、field metadata，并确认 public API 仍无裸指针暴露。不要把 `createPlugin` 或 callback trampoline 放入同一批。
