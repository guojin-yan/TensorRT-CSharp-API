# Plugin Registry Inventory：安全只读 API 如何避免 Borrowed Pointer 泄露

## 适用读者

这篇文章适合 TensorRT plugin 用户、需要审查 native ownership 的维护者，以及正在关注 deferred 接口提升策略的贡献者。

## 解决问题

Plugin registry 是高价值 API：用户想知道 registry 是否存在、creator 数量、creator name/version/namespace、field metadata 等信息。但 plugin creator、field collection 和 plugin instance 都涉及 borrowed pointer、库加载、注册/反注册、clone、enqueue 和 callback。直接把 native pointer 暴露给 C# public API，会把生命周期风险转嫁给用户。

## 只读 Inventory 的安全路线

当前项目优先提升安全只读查询：

- registry exists / creator count。
- creator name / version / namespace。
- creator field count / field name / field metadata。
- creator lookup 的只读结果。
- TRT10/TRT11 已有能力的跨版本一致封装。

这些 API 的实现原则是 count/copy 或 caller buffer：native 层把字符串和数组复制出来，托管层只看到不可变 snapshot。这样用户可以诊断 plugin 环境，而不会持有 plugin creator 的悬空地址。

## 暂缓的能力

register、deregister、load library、deregister library、resource acquire/release、plugin instance create/clone/enqueue、Plugin V2/V3 callback trampoline 都应继续谨慎处理。它们不是“补一个入口”就能完成的接口，而是需要 no-throw ABI、ownership ledger、dispose 顺序、错误码、测试和 package-consumer/runtime proof 的组合。

## 示例检查路径

```powershell
rg -n "Plugin|plugin|deferred" native src artifacts/interface-coverage
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --filter "FullyQualifiedName~Plugin"
```

这些命令适合维护者检查接口状态，不应被解释为 plugin runtime proof。

## 边界说明

Plugin inventory 是 readonly diagnostics，不是 runtime proof。`build-only`、`dry-run`、`template`、local feed、ProjectReference、direct `.nupkg`、GUI screenshot、TensorRtExec report、YoloVision matrix、OnnxToEngine report、sidecar、readonly diagnostics 都不能替代 plugin load/register/enqueue proof，也不能替代 package-consumer-runtime proof。

## 下一步

下一步可以继续扩展只读 plugin field metadata 的高层 wrapper 和 smoke；涉及注册、资源和 callback 的 API 应等待 ownership bridge 设计和真实 runtime proof 再开放。
