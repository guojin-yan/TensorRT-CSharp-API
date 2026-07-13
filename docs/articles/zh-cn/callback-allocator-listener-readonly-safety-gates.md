# 下一批只读安全门：Callback、Allocator 与 Listener 的发布前边界

Callback、Allocator、OutputAllocator、DebugListener 等接口是 TensorRT 高级集成的重要能力，但它们跨越 native lifetime、C ABI、C# GC、callback vtable 和 borrowed pointer 边界。下一批提升必须先做只读安全门、metadata、preflight 和 owner ledger，不能直接进入 callback trampoline。

## 适用读者

- 需要 allocator / debug listener 能力的高级用户。
- 维护 TensorRT native bridge 与 C# wrapper 的开发者。
- 审核 deferred 边界是否可提升的项目负责人。

## 解决问题

当前项目已经有大量 callback/allocator 相关 design gate 和 precheck 文档，但真实完成度不能按“文件存在”判断。下一批工作应聚焦：

- pointer-free metadata。
- attach/detach preflight。
- owner ledger。
- copied status / diagnostic。
- no-throw native boundary。
- managed readiness snapshot。

不要在 ownership 不清楚时暴露裸 `IntPtr`、`nint` 或 borrowed pointer，也不要让 C++ 异常跨 ABI 抛出。

## 推荐优先级

1. DebugListener attach native preflight：只返回 copied diagnostic，不触发真实 callback。
2. OutputAllocator owner ledger：记录 attach/detach/release 顺序，不交出 native owner。
3. Allocator callback owner design：确认 lifetime、dispose/drain、failure status mapping。
4. Calibrator metadata / readonly state：先做只读状态，不做数据回调。
5. PluginCreatorV3 metadata runtime precheck：继续保持 creator metadata copy-out，不创建 plugin instance。

## 边界说明

以下内容都不是 runtime proof：

- build-only
- dry-run
- template
- local feed
- ProjectReference
- direct `.nupkg`
- TensorRtExec report
- YoloVision matrix
- OnnxToEngine report
- readonly diagnostics
- design gate
- blocked-by-cuda-driver

`readonly diagnostics` 和 `design gate` 可以降低发布风险，但不能替代真实 callback invocation proof。真实 proof 至少需要 compatible host、真实 attach、真实 invocation、`InvocationCount>0`、`FailureCount=0`、`InFlightCallbackCount=0` 和 full package consumer report。

## 可复制命令

建议先跑只读和设计门质量测试：

```powershell
dotnet test tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~DeferredReadonly|FullyQualifiedName~Callback|FullyQualifiedName~Allocator|FullyQualifiedName~DebugListener|FullyQualifiedName~Calibrator"
```

如果后续补 native bridge，再单独跑对应 smoke 和 package-consumer-runtime 验证。不要把 precheck 或 skipped proof 写成 passed proof。

## 截图与图示建议

- Callback lifetime / drain 状态机。
- owner ledger 和 attach/detach 顺序图。
- “design gate -> managed readiness -> real callback runtime proof” 三层分解图。

## 下一步

1. 每批只选 2-4 个低风险设计门或 metadata/preflight。
2. 更新 `artifacts/interface-coverage/deferred-readonly-candidate-list.json`，但不删除 deferred 记录制造完成度。
3. 补 C# wrapper / test / 中文文档，并保持 public API pointer-free。
4. 等真实兼容主机和 full package consumer proof 完成后，再考虑 callback trampoline。
