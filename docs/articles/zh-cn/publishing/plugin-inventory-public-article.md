# Plugin Inventory 只读 API：从插件注册表看到什么

Plugin inventory 是 TensorRtSharp4.0 里用来排查 TensorRT 插件环境的只读诊断能力。它回答的问题很具体：当前进程能不能看到 TensorRT plugin registry，registry 里有多少 creator，每个 creator 的 name、version、namespace、interface kind、API language、TensorRT version 与 field metadata 是什么。

它不回答另一个问题：某个 plugin instance 能否被 create、clone、serialize、deserialize 或 enqueue。`TensorRtPluginRegistryInventory` 和 `TensorRtBuilder.PluginRegistryInventory.cs` / `TensorRtRuntime.PluginRegistryInventory.cs` 都只返回 copied metadata，不暴露 borrowed pointer，不接管 creator ownership，也不加载 plugin library。因此这类证据是 readonly diagnostics，不是 real-model-runtime proof，也不是 package-consumer-runtime proof。

## 适合

- 想检查目标机器 TensorRT plugin registry 是否可见的 .NET 使用者。
- 需要比较 TRT8、TRT10、TRT11 builder/global/runtime/capability registry 差异的维护者。
- 想理解 `src/JYPPX.TensorRtSharp/TensorRtPluginRegistryInventory.cs` 与 native plugin registry inventory bridge 的发布候选能力边界的人。
- 正在为公开包准备 local feed package consumer、direct `.nupkg` install、clean external consumer 与 post-publish verification 分层证据的发布负责人。

## 关键路径

- 高层模型：`src/JYPPX.TensorRtSharp/TensorRtPluginRegistryInventory.cs`。
- Builder-owned registry：`src/JYPPX.TensorRtSharp/TensorRtBuilder.PluginRegistryInventory.cs`。
- Runtime-local registry：`src/JYPPX.TensorRtSharp/TensorRtRuntime.PluginRegistryInventory.cs`。
- 环境级 global/capability registry：`src/JYPPX.TensorRtSharp/TensorRtEnvironmentProbe.PluginRegistryInventory.cs`。
- Native bridge：`native/src/tensorrt/common/plugin_registry_inventory.inc`。
- Managed interop：`src/JYPPX.TensorRtSharp/Internal/Interop/NativeBridgeApi.PluginRegistryInventory.cs` 与 `src/JYPPX.TensorRtSharp/Internal/Interop/NativeBridgeApi.RuntimePluginRegistryInventory.cs`。
- Smoke runner：`smoke/PluginRegistryInventorySmokeRunner/Program.cs`。
- 覆盖矩阵：`artifacts/interface-coverage/tensorrt-interface-comparison.csv`。
- 专项质量门：`PluginRegistryInventoryTests`、`PluginInventorySourceOnlySmokeTests`、`PluginCreatorApiLanguageReadonlyTests`、`PluginCreatorV3MetadataDesignGateTests`。

## 能读取什么

`TensorRtPluginRegistryInventory` 的公开读取面是托管对象快照：

```text
Line
Source
HasErrorRecorder
ParentSearchEnabled
CreatorCount
RecursiveCreatorCount
Creators
FindCreator(name, version, namespace)
TryFindCreator(name, version, namespace, out creator)
GetCreatorSummaries(maxCreators)
GetFieldSummaries(maxCreators, maxFieldsPerCreator)
GetDiagnostics()
```

creator 层复制这些字段：

```text
Index
Name
Version
Namespace
InterfaceKind
InterfaceMajor
InterfaceMinor
ApiLanguage
TensorRtVersion
Fields
```

field 层复制这些字段：

```text
Name
FieldType
Length
HasData
```

`HasData` 只说明 TensorRT 报告了非空 field data pointer；指针值本身不会进入 C#。`TensorRtPluginRegistryInventoryDiagnostics` 还会生成 `EmptyNameCount`、`EmptyVersionCount`、`EmptyNamespaceCount`、`CreatorWithFieldCount`、`TotalFieldCount`、`EmptyFieldNameCount`、`NegativeFieldLengthCount` 和 summary count/match 状态，用来判断 snapshot 是否像一个可读的 registry metadata 结果。

## Registry source

当前文章要区分四种 registry source：

```text
TensorRtPluginRegistrySource.Builder
TensorRtPluginRegistrySource.Global
TensorRtPluginRegistrySource.BuilderCapability
TensorRtPluginRegistrySource.Runtime
```

Builder source 来自 builder-owned registry；Runtime source 来自 runtime-local registry；Global 和 BuilderCapability source 由 `TensorRtEnvironmentProbe` 读取。TRT8 对 recursive creator count 的支持不同，文章和测试都不能把 TRT10/11 的 readback 行为机械套到 TRT8 上。

`ParentSearchEnabled` 和 `HasErrorRecorder` 也是状态读取，不是 ownership proof。即使 smoke runner 能打印 `GlobalPluginRegistryParentSearch Original/Requested/Readback/Restored`，也只说明 parent-search flag 可以 round-trip；它不能证明 plugin lifecycle 已经完成。

## Native bridge 边界

`native/src/tensorrt/common/plugin_registry_inventory.inc` 负责把 TensorRT registry 和 creator metadata 复制出来。它包含 SEH guard、vendor mismatch/missing 报告、registry availability probe、`getAllCreators` creator count probe、`getAllCreatorsRecursive` recursive creator count probe、error recorder presence probe 与 parent-search readback。

桥接层必须保持以下边界：

- 可以读 `getAllCreators` 返回的 creator list 并立即复制 metadata。
- 可以读 creator name/version/namespace/interface kind/API language/TensorRT version。
- 可以读 field name/type/length/hasData。
- 不返回 `IPluginRegistry*`、`IPluginCreatorInterface*`、field data pointer 或 error recorder pointer 给 C#。
- 不调用 `createPlugin`、`clone`、`serialize`、`deserializePlugin`、`attachToContext`、`enqueue` 或 `destroy`。
- 不把 `registerCreator`、`deregisterCreator`、`loadLibrary`、plugin resource acquire/release 或 callback trampoline 当成低风险只读 API。

这就是为什么文章只能把它归类为 copied/read-only diagnostics。

## Smoke runner 怎么读

`smoke/PluginRegistryInventorySmokeRunner/Program.cs` 是发布候选里最适合展示 inventory 的入口。它会先做 dependency probe，然后初始化 built-in plugins，再按 line 读取 global、builder capability、runtime-local 和 builder-owned registry：

```powershell
dotnet run --project .\smoke\PluginRegistryInventorySmokeRunner\PluginRegistryInventorySmokeRunner.csproj -- `
  --tensor-rt-line auto
```

在缺少 TensorRT、CUDA DLL、匹配 adapter 或本机 registry 时，smoke runner 会输出 `Skipped=True` 与 `Reason=...`。这不是失败伪装，也不是 runtime proof；它只是把环境 blocker 分类清楚，方便 owner 后续在真实机器上补证据。

一个健康的输出通常会包含：

```text
PluginRegistryInventorySmokeRunner
BuiltInPluginInitialization
GlobalPluginRegistry Exists=
BuilderCapabilityPluginRegistry Exists=
RuntimePluginRegistry Exists=
BuilderPluginRegistry Exists=
Inventory Source=
CreatorCount=
RecursiveCreatorCount=
FieldSummary Creator=
PluginRegistryInventoryDiagnostics
PluginRegistryInventorySmokeRunner Passed=True
```

其中 `FindCreator`、`TryFindCreator`、`GetCreatorSummaries` 和 `GetFieldSummaries` 仍然只在 copied managed snapshot 上工作，不会重新拿 native pointer。

## proof 边界

Plugin inventory 是只读诊断 API，不是 package-consumer-runtime proof。creator count、creator name/version/namespace、API language、TensorRT version 和 field metadata 能证明 bridge 可读取 registry metadata，但不能证明插件实例 create/clone/enqueue 成功，也不能替代真实模型 runtime proof。

禁止把以下内容写成 runtime proof：

- plugin registry exists。
- creator metadata readback。
- field metadata readback。
- `HasErrorRecorder` presence probe。
- `ParentSearchEnabled` round-trip。
- `Skipped=True Reason=DependencyProbeOnly`。
- TensorRtExec build-only 或 readonly diagnostics report。
- owner input template、local feed package consumer、ProjectReference consumer 或 direct `.nupkg` install。
- GitHub Actions dry-run、dashboard、GUI screenshot 或 release checklist。

能推动 package-consumer-runtime proof 的证据必须来自干净外部 consumer：安装公开包或候选包、加载匹配 runtime package、执行真实模型 inference、验证输出、记录 stdout/stderr/hash，并在 owner input validator 中通过 forbidden substitute scan。Plugin inventory 可以作为 context evidence，但不能单独晋级。

## 常见排障

`DllNotFoundException`、`BadImageFormatException`、`EntryPointNotFoundException`、`SEHException` 和 `AccessViolationException` 会被 `TryGetPluginRegistryInventory` 系列方法转换为 diagnostic 字符串，避免只读探针把发布门跑崩。文章里可以展示这些 blocker，但不能把 blocker 数量为零写成 release close。

TRT11 的 internal creators 在字段钩子上可能与 TRT8/TRT10 不同，所以 smoke runner 会输出 `CreatorFieldCollection Included=` 与原因。维护者看到 field count 变化时，应先核对 TensorRT line、creator interface kind 和 API language，不要立即认定 bridge regression。

如果 registry exists 为 false，优先检查 TensorRT DLL、CUDA runtime、plugin library 路径和 built-in plugin initialization。不要让教程建议把 DLL 复制到 C 盘临时目录；大文件、runtime 包和本地验证资产应放在 E 盘固定 workspace。

## 配图建议

- 一张三层图：TensorRT registry -> native copy bridge -> C# immutable wrapper。
- 一张 source 对照图：Builder / Global / BuilderCapability / Runtime 四个 source，全部标成 readonly diagnostics。
- 一张 checklist 图：registry exists、creator count、creator name/version/namespace、API language、field metadata、diagnostics，全都标成 not package-consumer-runtime proof。

## 下一步

继续推进安全的只读 metadata wrapper；plugin resource acquire/release、plugin instance create/clone/enqueue、Plugin V2/V3 callback trampoline、custom allocator、external resource 与 borrowed pointer API 仍应保持 deferred，直到 ownership、callback lifetime 和真实 runtime proof 有独立证据。
