# Allocator Owner 与 Callback Trampoline 设计门禁

> 状态：设计门禁
> 适用范围：TensorRT `IGpuAllocator`、`IGpuAsyncAllocator`、`IOutputAllocator`、`IDebugListener` callback 边界
> 当前策略：只读与清理类 safe controls 可用；allocator/debug-listener callback trampoline 继续 deferred，直到本门禁全部满足。

## 背景

当前项目已经把一部分 callback/allocator 边界提升为可调用 API：

- presence query：例如 `HasOutputAllocator`、`HasTemporaryStorageAllocator`、`HasDebugListener`。
- borrowed pointer clear：例如 `ClearOutputAllocator`、`ClearTemporaryStorageAllocator`、`ClearDebugListener`、`ClearGpuAllocator`。
- copied metadata：例如 `TryGetOutputAllocatorInterfaceInfo`、`TryGetTemporaryStorageAllocatorInterfaceInfo`、`TryGetDebugListenerInterfaceInfo`。

这些 API 都遵守同一个原则：不把 TensorRT 借出的 callback 指针暴露给 C#，不让 C# 接管 borrowed pointer 生命周期，也不让异常跨 ABI 抛出。

真正的 allocator/debug-listener callback trampoline 仍然不同。TensorRT 会长期保存 callback 对象，并在 build/runtime/enqueue 路径中反向调用用户代码。这个边界同时涉及 native vtable、托管 delegate、生存期、device pointer ownership、stream、alignment、OOM 和异常映射。只要其中一项不明确，就必须保持 deferred。

## 当前 deferred 边界

必须继续 deferred 的 direct callback 面：

| Interface | Methods | 当前原因 |
| --- | --- | --- |
| `IGpuAllocator` | `allocate`、`free`/`deallocate`、`reallocate` | TensorRT 反向调用用户 allocator；device pointer ownership 与释放配对必须可追踪。 |
| `IGpuAsyncAllocator` | `allocateAsync`、`deallocateAsync`、`getInterfaceInfo` | async stream lifetime 与 allocator owner 尚未建模；当前没有安全 owner getter。 |
| `IOutputAllocator` | `reallocateOutput`、`notifyShape` | TensorRT 在 execution context 中保存 callback；输出 buffer lifetime、shape 变化与释放责任必须明确。 |
| `IDebugListener` | `processDebugTensor` | TensorRT 向用户 callback 传递 debug tensor；tensor buffer lifetime 和异常策略必须明确。 |

已经有安全替代能力的 direct deferred 行仍应保留。例如 `IGpuAllocator::getInterfaceInfo` 可通过 execution context 的 temporary-storage allocator copied metadata 查询覆盖，但 direct borrowed allocator entry 仍保留 deferred 历史。

当前必须继续 deferred 的完整签名包括：

- `IGpuAllocator::allocate`
- `IGpuAllocator::deallocate`
- `IGpuAllocator::free`
- `IGpuAllocator::reallocate`
- `IGpuAsyncAllocator::allocateAsync`
- `IGpuAsyncAllocator::deallocateAsync`
- `IOutputAllocator::notifyShape`
- `IOutputAllocator::reallocateOutput`
- `IDebugListener::processDebugTensor`

## Dry-run owner skeleton 进展

当前已新增 `TensorRtAllocatorCallbackOwner`、`TensorRtAllocatorDryRunRequest`、`TensorRtAllocatorDryRunResult` 和 `TensorRtAllocatorDryRunHandler` 作为托管 dry-run 诊断骨架。该骨架验证 managed handler keep-alive、invocation/failure 计数、`LastCallbackException`、`LastDiagnostic` 和异常吞吐，不绑定 TensorRT，不返回 device pointer，也不解除任何 direct allocator/debug-listener callback deferred。

该能力的 readiness marker 是 `allocator-owner-dry-run-diagnostics`。它只能证明 package consumer 可以强类型引用 owner skeleton 和 dry-run diagnostic；不能作为 `IGpuAllocator::allocate/free/deallocate/reallocate`、`IGpuAsyncAllocator::*`、`IOutputAllocator::*` 或 `IDebugListener::processDebugTensor` 已可用的证据。

当前还新增了 `TensorRtAllocatorNativeDryRunResult` 与 `TensorRtAllocatorCallbackOwner.RunNativeDryRunDiagnostic`，用于创建短生命周期 native diagnostic owner、写入 copied counters/status/message 并立即释放 native handle。该能力的 readiness marker 是 `allocator-owner-native-dry-run-controls`。它只证明 native owner 生命周期和 copied diagnostic C ABI 已经可探测；仍不调用 `setGpuAllocator`、不绑定 `IOutputAllocator`、不实现 TensorRT callback trampoline，也不能作为真实 allocator callback 已启用的证据。

当前进一步新增了 `TensorRtAllocatorOwnerStateDryRunResult` 与 `TensorRtAllocatorCallbackOwner.RunNativeStateLedgerDryRunDiagnostic`，用于在短生命周期 native owner 内记录 synthetic attach/allocation/release/detach intent，并复制 owner id、state transition count、ledger allocation/release/failure count、last operation 与 diagnostic。该能力的 readiness marker 是 `allocator-owner-state-ledger-dry-run-controls`。它只证明状态机和 ledger intent 的 C ABI、C# wrapper、smoke/package-consumer/readiness 证据已闭环；仍不保存真实 device pointer，不调用 `setGpuAllocator` / `setOutputAllocator`，也不解除任何 callback deferred 行。

当前还新增了 allocator owner lifecycle snapshot 诊断：public `TensorRtAllocatorCallbackOwner.RunLifecycleDiagnostic`、`GetSnapshot` 和 `TensorRtAllocatorCallbackOwnerSnapshot` 会包装底层 `RunInternalSyncAllocatorRuntimePrototype` / `GetInternalRuntimePrototypeSnapshot`，输出 `EvidenceKind=allocator-owner-internal-runtime-prototype`、`RuntimeEvidenceKind=not-present`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`CallbackKind=sync-allocator-prototype`、`InFlightCallbackCount`、`ReleaseHookCount`、`CallbackStatePinned`、`DelegatePinned`、`DisposeRequested`、`DevicePointerExposed=False`、`DevicePointerProduced=False`、`BorrowedPointerEscaped=False`、`ManagedKeepAliveReady`、`DisposeReleaseReady`、`PointerFreeSurfaceReady`、`LastStatus` 和 `LastDiagnostic`。该 prototype 验证托管 `GCHandle` 与 delegate keep-alive 配对、no-throw/exception-to-status、in-flight callback counter 和 dispose release hook 诊断；它不注册到 TensorRT，不调用 `setGpuAllocator`，不产生 device pointer，不是 `real-callback-runtime`，也不是 proof that TensorRT callbacks are enabled。

当前还新增了 [Allocator Owner Ledger Safety Gate](allocator-owner-ledger-safety-gate.md)：`TensorRtAllocatorLedgerSafetyGate` / `TensorRtAllocatorLedgerSafetyGateResult` 将 internal prototype 与 native owner state ledger dry-run 汇总为 `EvidenceKind=allocator-owner-ledger-safety-gate`、`RuntimeEvidenceKind=ledger-safety-gate`、`RealCallbackRuntime=False`、`IsRealCallbackRuntimeProof=False`、`ManagedKeepAliveReady`、`DisposeReleaseReady`、`NativeLedgerDesignReady`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`。该 gate 只复制诊断和阻塞项，不调用 `setGpuAllocator`，not proof。

## Native owner 与 ledger 设计进展

当前已新增 [Native Allocator Owner 与 Device Pointer Ledger 设计门禁](allocator-owner-ledger-design.md)，将 native owner shape、device pointer ledger、失败/OOM/异常映射和 TRT8/TRT10/TRT11 route 拆成独立可审计门禁。

该能力的 readiness marker 是 `allocator-owner-ledger-design-gate`。它只表示设计门禁已被文档、quality tests 和 readiness 报告审计；不能作为真实 TensorRT allocator callback 已启用的证据。

下一层门禁是 [真实 Callback Trampoline 门禁复审](real-callback-trampoline-gate.md)。该门禁使用 `real-callback-trampoline-gate` marker，要求 package-consumer/runtime evidence 明确区分 `dry-run`、`copied-state`、`internal-runtime-gate` 与 `real-callback-runtime`，并继续确认 `IGpuAllocator::*`、`IGpuAsyncAllocator::*`、`IOutputAllocator::*` 和 `IDebugListener::processDebugTensor` direct callback rows 未被误删。[OutputAllocator Runtime Gate](output-allocator-runtime-gate.md) 的 `output-allocator-internal-runtime-gate` 只复制 shape/size/alignment diagnostics，`RealCallbackRuntime=False`，not proof。

## 解除 deferred 的最低条件

任何 allocator/debug-listener callback API 解除 deferred 前，必须同时满足以下条件。

### 1. Native owner 明确

必须有 native owner 对象负责实现 TensorRT callback interface，并且只能通过 bridge C ABI 创建、绑定、解绑、销毁。

要求：

- native owner 析构不抛异常。
- native owner 持有的 managed callback state 只能通过显式 release hook 释放。
- TensorRT 保存 callback pointer 时，native owner 必须至少活到对应 builder/runtime/context/config 的解绑或销毁。
- owner 不能被复制；只能 move 或以 stable handle 管理。
- 每个 TRT line 的 owner 绑定必须有 version guard。

### 2. Managed owner 明确

C# public API 必须提供托管 owner wrapper，而不是让用户传入裸 `IntPtr`。

要求：

- wrapper 必须执行 delegate pinning，固定托管 delegate 和 callback state。
- wrapper 暴露 `IsAttached`、`CallbackInvocationCount`、`CallbackFailureCount`、`LastCallbackException` 一类诊断。
- wrapper `Dispose` 必须先 detach 或进入不可回调状态，再释放 native owner。
- wrapper 不允许把 TensorRT 借出的 callback pointer、device pointer ownership 或 native vtable pointer 作为 public API 返回。
- wrapper 需要 XML 注释说明 lifetime：谁持有、谁释放、何时可 Dispose。

### 3. Device pointer ownership 明确

allocator callback 的返回值不能被当成普通 `IntPtr` 交给用户自由管理。

要求：

- 每次 allocation 必须记录 size、alignment、stream、owner、pointer、状态。
- deallocate/free 必须验证 pointer 来自同一个 owner，避免跨 allocator 释放。
- OOM 必须可诊断，不能通过跨 ABI exception 表达。
- callback 返回 null / failure 时，必须映射为 TensorRT 可接受的失败语义，并记录 last error。
- 如果需要暴露诊断，只能暴露 pointer value 的只读数值或 allocation id，不暴露可解引用 ownership。

### 4. ABI no-throw 明确

native 和 managed callback 都不能让异常跨 ABI 逃逸。

要求：

- managed callback 体内异常必须捕获，写入 owner diagnostic。
- native trampoline 必须捕获 C++ exception 和 Windows SEH。
- C ABI 只返回 `JYPPX_StatusCode`、bool 或 TensorRT callback 约定允许的失败值。
- last error 中必须包含 feature name、line、callback kind 和失败分类。

### 5. Stream 与 async 边界明确

`IGpuAsyncAllocator` 不应复用同步 allocator 的简单模型。

要求：

- stream handle 只能作为 borrowed execution context 参数处理。
- async allocation 的完成/可释放时机必须由设计文档说明。
- 不能让托管层在 stream 未同步时释放仍被 TensorRT 使用的 memory。
- 在没有 stream-lifetime 设计和 smoke 前，`IGpuAsyncAllocator::*` 继续 deferred。

### 6. 跨版本策略明确

TRT8、TRT10、TRT11 的 callback API 不完全一致，不能用一个 public wrapper 模糊处理。

要求：

- TRT8 使用 `free` / `getInterfaceVersion` 等旧接口时，必须有独立 route。
- TRT10/TRT11 使用 `InterfaceInfo` 时，必须保持 copied metadata 模式。
- 每个 line 的 manifest、native header、source、C# interop 和 wrapper 都要同步。
- quality tests 必须检查 version guard 和 deferred rows 未被误删。

## 建议实现顺序

1. 只做 owner skeleton，不接入 TensorRT callback：
   - native owner create/destroy。
   - managed wrapper pin/unpin。
   - diagnostic counters。
   - no public pointer exposure tests。
2. 增加 dry-run diagnostic callback：
   - 不返回 device pointer。
   - 只验证 managed-to-native-to-managed callback 路径和异常记录。
3. 选择最小同步 allocator 场景：
   - 只允许 `IGpuAllocator::allocate/deallocate`。
   - 暂不处理 `reallocate`、async allocator、output allocator。
   - 使用真实 TensorRT smoke 前必须先有 package-consumer 强类型证据。
4. 最后再评估 output allocator 和 debug listener：
   - 它们绑定在 execution context 生命周期上，必须先解决 detach 与 context dispose 顺序。

## 质量门

每个阶段至少保留以下验证：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
dotnet build .\TensorRtSharp.sln -c Debug --no-restore /p:UseSharedCompilation=false
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build /p:UseSharedCompilation=false
```

如果修改 native callback owner 或 TensorRT binding，还必须追加：

```powershell
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
cmake --preset win-x64-trt10-cuda12-release
cmake --build --preset win-x64-trt10-cuda12-release --parallel
```

如果修改 package consumer / readiness，还必须追加：

```powershell
dotnet pack .\pack\JYPPX.TensorRT.CSharp.API\JYPPX.TensorRT.CSharp.API.csproj -c Debug -o .\artifacts\managed -p:JYPPXPackageVersion=4.0.0 /p:UseSharedCompilation=false
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-BridgePackageConsumer.ps1 -SourceRuntimeKey win-x64-trt11.0-cuda13.2-cudnn9.22 -BridgePackageDirectory .\artifacts\runtime-split-nupkg\win-x64-trt11.0-cuda13.2-cudnn9.22 -SkipProbe
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RuntimePackageReadiness.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22
```

## 当前结论

在 native owner、managed owner、device pointer ledger、stream lifetime 和 destroy hook 都未实现前，allocator/debug-listener callback trampoline 不应解除 deferred。当前可继续提升的范围应优先选择只读查询、copied metadata、presence/clear controls、package-consumer/readiness 证据和 owner skeleton，并在 `real-callback-trampoline-gate` 中保持真实 callback runtime 的 go/no-go 复审。
