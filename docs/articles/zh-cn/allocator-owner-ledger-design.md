# Native Allocator Owner 与 Device Pointer Ledger 设计门禁

> 状态：设计门禁
> 适用范围：TensorRT `IGpuAllocator`、`IGpuAsyncAllocator`、`IOutputAllocator`、`IDebugListener` callback 边界
> readiness marker：`allocator-owner-ledger-design-gate`
> 当前结论：只建立 owner/ledger/status 设计证据，不解锁真实 TensorRT allocator callback。
> 下一层门禁：[真实 Callback Trampoline 门禁复审](real-callback-trampoline-gate.md)，readiness marker 为 `real-callback-trampoline-gate`。

## 目标

本门禁用于把上一阶段的纯托管 `TensorRtAllocatorCallbackOwner` dry-run skeleton 推进到可审计的 native owner 与 device pointer ledger 设计。它不是 allocator callback 的实现完成声明，也不允许删除 deferred rows。

当前仍必须 deferred：

- `IGpuAllocator::allocate`
- `IGpuAllocator::deallocate`
- `IGpuAllocator::free`
- `IGpuAllocator::reallocate`
- `IGpuAsyncAllocator::allocateAsync`
- `IGpuAsyncAllocator::deallocateAsync`
- `IOutputAllocator::notifyShape`
- `IOutputAllocator::reallocateOutput`
- `IDebugListener::processDebugTensor`

## Native Owner 形状

真实 callback 解除 deferred 前，native owner 必须满足以下约束：

- owner 只能通过 C ABI create/destroy/release hook 管理。
- owner 不可复制，必须保持 stable address。
- owner 析构必须 no-throw。
- owner 内部可以保存 native callback vtable state，但 public C# 不能获得 native owner pointer。
- owner 必须有显式 attach/detach 状态，且 detach 前不能释放 TensorRT 仍可能回调的对象。
- owner 必须保存 callback kind、TensorRT API line、last status、last diagnostic、invocation count、failure count。
- owner release hook 必须能释放托管 GCHandle/delegate state，且重复 release 安全。
- owner 的每个 ABI entry 必须捕获 C++ exception，并在 Windows 上捕获 SEH。

建议 native 结构只作为私有实现细节，类似：

```cpp
struct AllocatorOwner final
{
    AllocatorOwner(AllocatorOwner const&) = delete;
    AllocatorOwner& operator=(AllocatorOwner const&) = delete;
    ~AllocatorOwner() noexcept;

    JYPPX_TensorRtApiLine line;
    JYPPX_AllocatorCallbackKind callback_kind;
    std::atomic<uint64_t> invocation_count;
    std::atomic<uint64_t> failure_count;
    std::atomic<JYPPX_StatusCode> last_status;
    char last_diagnostic[1024];
};
```

该示例只描述 shape，不要求当前阶段新增 native source。

## Device Pointer Ledger

allocator callback 返回值不能被当成普通 `IntPtr` 交给用户自由管理。真实 allocation 前必须有 ledger。

ledger entry 至少包含：

| 字段 | 含义 |
| --- | --- |
| `allocationId` | owner 内单调递增 ID，用于诊断和释放配对 |
| `ownerId` | 产生该 allocation 的 owner |
| `line` | TRT8/TRT10/TRT11 route |
| `kind` | sync allocator、async allocator、output allocator 等 |
| `pointerValue` | 只读数值，不表示 public ownership |
| `size` | 请求或实际字节数 |
| `alignment` | 对齐要求 |
| `streamValue` | async stream 的 borrowed handle 数值，仅诊断 |
| `state` | allocated、released、failed、unknown |
| `lastStatus` | 最近一次分配/释放状态 |
| `diagnostic` | OOM、跨 owner 释放、重复释放等诊断 |

约束：

- deallocate/free 必须验证 pointer 来自同一个 owner。
- 重复释放必须进入 diagnostic，不允许静默成功。
- 跨 owner 释放必须失败并记录。
- OOM 不能通过跨 ABI exception 表达。
- public API 只能暴露 allocation id、size、alignment、state、diagnostic 和 pointer value 的只读数值；不能暴露可释放、可解引用或可转交 ownership 的 raw pointer。

## 失败与异常映射

所有 callback 边界必须 no-throw。

| 来源 | 映射 |
| --- | --- |
| managed handler exception | `CallbackFailureCount++`、`LastCallbackException`、owner last diagnostic、native status 非 OK |
| native C++ exception | `JYPPX_STATUS_RUNTIME_ERROR` 或更具体 status，记录 feature/line/callback kind |
| Windows SEH | `JYPPX_STATUS_RUNTIME_ERROR`，diagnostic 必须包含 structured exception code |
| OOM | TensorRT callback 允许的 failure/null 语义，ledger entry 标记 failed |
| invalid pointer | status 非 OK，ledger diagnostic 标记 foreign pointer 或 released pointer |
| unsupported API line | `JYPPX_STATUS_NOT_SUPPORTED`，保留跨版本 route 信息 |

错误信息必须包含：

- feature：例如 `allocator-owner-ledger`。
- TensorRT line：TRT8/TRT10/TRT11。
- callback kind：sync/async/output/debug-listener。
- operation：allocate/deallocate/free/reallocate/notify/process。
- 失败分类：managed-exception/native-exception/seh/oom/invalid-state/unsupported。

## 跨版本 Route

TRT8、TRT10、TRT11 不能共用一个模糊 route。

- TRT8 使用 `IGpuAllocator::free`，必须独立于 TRT10/TRT11 的 `deallocate`。
- TRT10/TRT11 的 `IGpuAllocator::deallocate` route 必须分开 guard。
- TRT10/TRT11 `InterfaceInfo` 继续使用 copied metadata 模式。
- `IGpuAsyncAllocator::*` 必须等待 stream lifetime 设计，不得复用同步 allocator 简化模型。
- `IOutputAllocator::*` 必须等待 execution context detach/dispose 顺序设计。
- `IDebugListener::processDebugTensor` 必须等待 debug tensor buffer lifetime 设计。

## Public C# 边界

public wrapper 必须继续隐藏 ownership：

- 不公开 native owner pointer。
- 不公开 callback vtable pointer。
- 不公开可释放的 device pointer。
- 不接受用户传入裸 `IntPtr` 作为 allocator ownership。
- 诊断类型可暴露只读 allocation id、size、alignment、state、diagnostic。
- `TensorRtAllocatorCallbackOwner` 继续是当前唯一 public owner 形状，但仍是 dry-run only。

在真实 callback 解锁前，XML 注释必须明确说明：

- 谁持有 owner。
- 何时 attach/detach。
- `Dispose` 在 attached 时如何处理。
- 异常如何映射。
- device pointer ownership 不属于 public C#。

## 解锁前检查清单

真实 allocator callback 解除 deferred 前，至少需要：

- native owner create/destroy/release C ABI。
- owner no-copy/no-throw 测试或源码审计。
- managed delegate/GCHandle pinning 与 release hook。
- device pointer ledger 结构和释放配对测试。
- TRT8/TRT10/TRT11 独立 manifest、native source、header、interop、wrapper route。
- smoke 覆盖 dry-run native owner。
- package consumer 强类型引用。
- readiness marker 区分设计门禁、dry-run、真实 callback。
- coverage matrix 仍保留 direct deferred 历史，只有真实替代 API 可以标记 implemented-with-deferred-history。

## 当前阶段边界

本阶段只确认 `allocator-owner-ledger-design-gate`，代表 native owner 与 ledger 的最低设计门禁已经写入并被 tests/readiness/docfx 审计。它不代表以下接口可调用：

- `IGpuAllocator::allocate/free/deallocate/reallocate`
- `IGpuAsyncAllocator::allocateAsync/deallocateAsync`
- `IOutputAllocator::notifyShape/reallocateOutput`
- `IDebugListener::processDebugTensor`

进入真实 callback trampoline 前，还必须通过 [真实 Callback Trampoline 门禁复审](real-callback-trampoline-gate.md)，确保 native owner 生命周期、dispose 顺序、`GCHandle`/delegate pinning、`C ABI no-throw`、Windows SEH、`exception-to-status`、device pointer ledger、`stream/async` 和 package-consumer `real-callback-runtime` 证据都可审计。`dry-run` 和 `copied-state` 仍不能作为真实 TensorRT callback 已启用的证据。

## Dry-run 状态机进展

当前已新增 `allocator-owner-state-ledger-dry-run-controls`，把上一节的设计门禁推进到可调用但仍 synthetic 的 native owner 状态机：

- `allocator_owner_dry_run_get_state`
- `allocator_owner_dry_run_attach_intent`
- `allocator_owner_dry_run_detach_intent`
- `allocator_owner_dry_run_ledger_record_allocation_intent`
- `allocator_owner_dry_run_ledger_record_release_intent`

这些 entrypoint 已在 TRT8、TRT10、TRT11 manifest/header/source 中保持一致 version guard，并由 C# `TensorRtAllocatorCallbackOwner.RunNativeStateLedgerDryRunDiagnostic` 包装成 copied result。该能力只记录 owner id、状态转移次数、synthetic allocation id、ledger 计数、last operation 和 diagnostic，不保存真实 pointer value，不返回 device pointer，也不调用 TensorRT callback setter。
