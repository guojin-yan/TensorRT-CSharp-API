# Plugin Inventory 只读 API：从插件注册表看到什么

这篇文章面向想确认 TensorRT 插件环境是否可用的使用者。它解释 `TensorRtPluginRegistryInventory` 如何读取 registry exists、creator count、creator name/version/namespace 与 field metadata，并说明为什么这些 API 只返回复制后的字符串和结构，不暴露 borrowed pointer。

## 适合

- 想检查目标机器 TensorRT plugin registry 是否可见的 .NET 使用者。
- 需要比较 TRT10 / TRT11 插件 creator inventory 的维护者。
- 想理解 `src/JYPPX.TensorRtSharp/TensorRtPluginRegistryInventory.cs` 与 native plugin registry inventory bridge 的发布候选能力边界的人。

## 关键路径

- 高层入口：`src/JYPPX.TensorRtSharp/TensorRtPluginRegistryInventory.cs`。
- Native bridge：`native/src/tensorrt/common/plugin_registry_inventory.inc`。
- 覆盖矩阵：`artifacts/interface-coverage/tensorrt-interface-comparison.csv`。
- 质量测试：plugin registry inventory smoke / project quality tests。

## proof 边界

Plugin inventory 是只读诊断 API，不是 package-consumer-runtime proof。creator count、creator name/version/namespace 和 field metadata 能证明 bridge 可读取 registry metadata，但不能证明插件实例 create/clone/enqueue 成功，也不能替代真实模型 runtime proof。

禁止把以下内容写成 runtime proof：

- plugin registry exists。
- creator metadata readback。
- field metadata readback。
- TensorRtExec build-only 或 readonly diagnostics report。
- owner input template 或 local feed consumer。

## 配图建议

- 一张三层图：TensorRT registry -> native copy bridge -> C# immutable wrapper。
- 一张 checklist 图：registry exists、creator count、creator name/version/namespace、field metadata，全都标成 readonly diagnostics。

## 下一步

继续推进安全的只读 metadata wrapper；plugin resource acquire/release、plugin instance create/clone/enqueue、Plugin V2/V3 callback trampoline 仍应保持 deferred，直到 ownership 和 callback lifetime 有独立 runtime proof。
