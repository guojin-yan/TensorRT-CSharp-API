# Plugin Ownership Boundary

`plugin-ownership-boundary` 用来区分已经安全落地的 Plugin Registry copied metadata inventory，以及仍必须 deferred 的 plugin instance、resource、library 和 callback 生命周期接口。这个边界不是 “plugin API 未完成” 的笼统标签，而是防止把 borrowed pointer 或 callback 生命周期暴露给 C# public API 的发布门禁。

## 已完成的安全面

当前已完成并由质量测试覆盖的是只读、复制式 metadata surface：

- builder-owned plugin registry inventory：registry exists、creator count、creator metadata、field metadata、creator lookup。
- global runtime plugin registry probe：global registry exists、creator lookup、interface info、field metadata。
- runtime-local plugin registry inventory：runtime registry exists、creator count、recursive creator count、creator metadata、lookup metadata。
- TRT8 / TRT10 / TRT11 跨版本 wrapper：通过 count/copy 或 caller-buffer 模式复制字符串和字段信息，不返回 TensorRT 内部 borrowed pointer。
- C# public API：`TensorRtPluginRegistryInventory`、`TensorRtPluginCreatorInfo`、`TensorRtPluginFieldInfo`、`TensorRtPluginRegistryInventoryDiagnostics` 等均为 managed value snapshot，不要求用户持有 `IPluginCreator*`。

这些能力可以用于诊断部署环境、确认 plugin creator 是否存在、读取 creator name/version/namespace，以及复制 field name/type/length/hasData metadata。它们不能创建 plugin，也不能证明 plugin enqueue 或 callback trampoline 已经可用。

`TensorRtPluginRegistryInventory.GetDiagnostics()` 属于同一个安全面：它只检查已经复制到 C# 对象中的 creator count、summary count、recursive count、field count、空 name/version 和异常 field length，不调用 TensorRT、不返回 borrowed pointer，也不复制 field data payload。

## 继续 deferred 的高风险面

以下接口仍属于 high-risk ownership boundary，不能从 candidate plan 中机械提升：

- registry register / deregister。
- load library / deregister library。
- plugin resource acquire / release。
- plugin instance create / clone / destroy。
- plugin enqueue / configure / attach / detach。
- Plugin V2/V3 callback trampoline。
- borrowed plugin creator、plugin resource、plugin registry 或 plugin instance pointer。

这些接口需要先完成 owner model、no-throw ABI boundary、callback exception/status mapping、resource lifetime ledger、real runtime smoke 和 package-consumer proof。没有这些证据时，任何 “参数补齐” 都只是把风险前移到用户侧。

## Public API 边界

Plugin public API 必须满足：

- 不暴露 `public IntPtr` / `public nint` plugin creator、resource、registry 或 instance。
- 字符串与数组输出必须通过 count/copy、caller-buffer 或 managed snapshot。
- 诊断 API 只能消费 managed snapshot；不能把 native registry、creator 或 plugin instance pointer 作为 public surface。
- unsupported TensorRT line 通过 diagnostic / `Try*` API 表达，不跨 ABI 抛异常。
- TRT8 / TRT10 / TRT11 version guard 必须在 manifest、native source、interop routing 和 wrapper 中保持一致。

## 下一步

后续如果要继续推进 plugin 能力，应先拆成两个阶段：

1. 继续只读 metadata：只允许已有 owner object 下的 copied snapshot，并且必须有 smoke 或 quality test。
2. plugin instance/resource/callback：必须先完成 owner ledger、no-throw callback bridge、runtime proof precheck 和真实 runtime smoke，再考虑解除 deferred。

直到这些条件满足前，`plugin-ownership-boundary` 只能作为边界规划输入，不能作为 release proof 或低风险提升清单。
