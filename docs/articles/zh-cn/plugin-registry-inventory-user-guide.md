# Plugin Registry Inventory 用户指南

## 适用读者

本文面向需要查看 TensorRT plugin registry 和 plugin creator inventory 的 C# 用户，尤其是希望在不创建 plugin、不接管 borrowed pointer 的前提下读取 creator name、version、namespace 和 field metadata 的开发者。

## 解决问题

Plugin registry 常见误区是把 native creator 指针暴露给 public API。本文说明只读 inventory 的使用边界：可以查询 registry 是否存在、creator count、creator name/version/namespace、field count 和 field metadata，但不做 register、deregister、load library、create、clone、enqueue 或 callback trampoline。

## 背景与场景

当前主线已经从 missing 接口清零转向 deferred 边界提升。Plugin Inventory 属于高价值低 ownership 风险区域，因为它能帮助用户判断 TensorRT 当前进程里有哪些 plugin creator 可见，同时避免 plugin instance 生命周期和回调 ABI 风险。

## 操作路径

1. 调用高层 C# wrapper 查询 registry exists，若 registry 不存在则返回空 inventory。
2. 查询 creator count，按 index 读取 creator identity。
3. 对每个 creator 读取 name、version、namespace，使用 caller buffer 或 copy 模式避免悬空字符串。
4. 读取 field count 和 field metadata，记录 field name、type、length 和是否可解释。
5. 将结果导出为只读 diagnostics，供用户检查模型所需 plugin 是否可见。

## 代码与文件入口

- `src/JYPPX.TensorRtSharp/Plugins/TensorRtPluginRegistryInventory.cs`：高层只读聚合 wrapper。
- `src/JYPPX.TensorRtSharp/Plugins/TensorRtPluginCreatorInfo.cs`、`TensorRtPluginFieldInfo.cs`：复制型 metadata owner。
- `native/src/tensorrt/common`：跨版本 native adapter。
- `native/src/tensorrt/v10` 与 `native/src/tensorrt/v11`：版本特定实现。
- `native/manifests/tensorrt/v10` 与 `native/manifests/tensorrt/v11`：manifest 声明。
- `docs/articles/zh-cn/plugin-inventory-readonly-api.md`：已有只读 API 说明。

## 图示建议

建议画一个树状图：`registry -> creators[] -> fields[]`。creator 节点只展示复制出的字符串和 metadata，不展示 native pointer。

## 边界说明

Plugin inventory 输出是 readonly diagnostics，不是 runtime proof。它不能替代 TensorRtExec report、OnnxToEngine report、YoloVision matrix 或真实 package consumer 运行记录。build-only、dry-run、template、local feed、ProjectReference 和 direct `.nupkg` 也不能因为 inventory 成功而晋级为 proof。

## 下一步

下一轮应继续补齐 TRT10/TRT11 的 creator field metadata 一致封装，并增加 smoke 覆盖 registry exists、creator count 和 creator identity 的可执行路径。
