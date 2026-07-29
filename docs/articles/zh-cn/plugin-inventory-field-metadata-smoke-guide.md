# Plugin Inventory Field Metadata Smoke 指南

## 适用读者

本文面向验证 TensorRT plugin registry inventory 只读 API 的维护者，重点关注 creator field metadata 是否能安全复制到 C# 层。

## 解决问题

Plugin creator metadata 常见风险是 borrowed pointer 生命周期不清楚。本文说明 smoke 应覆盖 registry exists、creator count、creator name/version/namespace、field count 和 field metadata，同时避免 public API 暴露裸 `IntPtr`。build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 也不能被误判为 runtime proof。

## 背景与场景

Plugin inventory 属于 deferred readonly API 提升中较安全的区域：它只读取 registry 与 creator 的复制快照，不创建 plugin instance，不 register/deregister，不 load library，不进入 callback trampoline。Field metadata smoke 能证明 wrapper 能读取字段结构，但不证明 plugin enqueue 或模型运行成功。

该路径必须保持 pointer-free：native creator、field collection、field name 和 field metadata 都只能复制为托管 snapshot、record、string 或 array，不允许把 borrowed pointer 作为 public API 结果暴露出去。

## 操作路径

1. 在 native 层使用 count/copy 或 caller buffer 模式读取 creator 和 field 信息。
2. 在 C# interop 层将字符串和数组复制到托管 record，不暴露 creator pointer。
3. 在高层 API 中返回不可变 inventory snapshot。
4. 在 smoke 中至少断言 registry exists 和 creator count 路径可执行。
5. 如果当前环境无 plugin creator，也应返回空 inventory 而不是错误晋级。

## 代码与文件入口

- `src/JYPPX.TensorRtSharp/Plugins/TensorRtPluginRegistryInventory.cs`：高层只读 inventory 聚合。
- `src/JYPPX.TensorRtSharp/Plugins/TensorRtPluginCreatorInfo.cs` 与 `TensorRtPluginFieldInfo.cs`：复制型 creator/field metadata。
- `src/JYPPX.TensorRtSharp/Plugins/TensorRtPluginRegistryInventoryDiagnostics.cs`：pointer-free inventory 诊断。
- `src/JYPPX.TensorRtSharp/Builder/TensorRtBuilder.PluginRegistryInventory.cs`：builder 侧入口。
- `native/src/tensorrt/common`：跨版本 native adapter。
- `native/src/tensorrt/v10` 与 `native/src/tensorrt/v11`：版本实现。
- `docs/articles/zh-cn/plugin-registry-inventory-user-guide.md`：用户指南。

## 图示建议

建议用快照图展示 `RegistrySnapshot -> CreatorSnapshot[] -> FieldMetadata[]`，并用边框标出所有数据都已复制到托管内存。

## 边界说明

Plugin inventory smoke 是 readonly diagnostics，不是 runtime proof。TensorRtExec report、OnnxToEngine report、YoloVision matrix、build-only、dry-run、template、local feed、ProjectReference 和 direct `.nupkg` 不能因为 plugin metadata 可读而晋级。

## 下一步

下一轮应继续补齐 field metadata 的跨版本一致性测试，并将 plugin resource acquire/release、plugin instance create/clone/enqueue 继续保持 deferred。
