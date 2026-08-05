# Callback Owner Closure Matrix

状态：closure-matrix / non-proof

readiness marker：`callback-owner-closure-matrix`

公开 API：

- `TensorRtCallbackOwnerClosureMatrix`
- `TensorRtCallbackOwnerClosureMatrix.Evaluate`
- `TensorRtCallbackOwnerClosureMatrixResult`
- `TensorRtCallbackOwnerClosureMatrixRow`

该矩阵用于把 callback owner 的设计门和历史 deferred 状态按 family 汇总到一个 pointer-free 结果中。它消费已有的 copied gate / precheck evidence，不调用 TensorRT，不安装 native vtable，不 attach non-null callback owner，不运行 allocator / debug listener / stream callback，也不返回 `IntPtr`、`nint`、native owner pointer、vtable pointer、device pointer、stream pointer 或 borrowed tensor pointer。各 family 后续捕获的独立 package-consumer runtime evidence 不会自动回写这份旧聚合器，因此必须同时查看对应的真实运行记录。

因此它是 owner 闭环矩阵，not proof。当前必须保持：

- `RuntimeEvidenceKind=closure-matrix`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `RuntimeProofBlocked=True`
- `DeferredRowsStillRequired=True`

## Matrix Families

当前 `TensorRtCallbackOwnerClosureMatrixResult` 覆盖 5 个 family：

| OwnerFamily | 代表 callback | 当前状态 |
| --- | --- | --- |
| `GpuAllocator` | `IGpuAllocator::allocate/free/deallocate/reallocate` | ledger / managed owner dry-run 可审计，但 line-specific attach、no-throw vtable、device pointer ledger、package-consumer runtime proof 未完成。 |
| `GpuAsyncAllocator` | `IGpuAsyncAllocator::allocateAsync/deallocateAsync` | 复用 allocator ledger evidence，但 CUDA stream lifetime 与 async ordering 未闭合。 |
| `OutputAllocator` | `IOutputAllocator::notifyShape/reallocateOutput` | owner design、detach clear、borrowed pointer blocker 已有 copied gate；non-null attach、native vtable、device pointer ownership、真实 runtime invocation 未完成。 |
| `DebugListener` | `IDebugListener::processDebugTensor` | native owner / no-copy / no-throw destructor / borrowed pointer blocker 等 gate 已接入矩阵；non-null attach 仍 disabled，package-consumer invocation proof 未完成。 |
| `StreamReaderWriter` | `IStreamReader::read`、`IStreamReaderV2::read/seek`、`IStreamWriter::write` | `IStreamReaderV2::read/seek` 已有 immutable native owner、no-throw vtable、pointer-free snapshot、borrower ledger 和 TensorRT 10.11 仓库外双包实机证明；legacy `IStreamReader`、`IStreamWriter` 与 TRT11 实机证明仍未完成，所以合并 family 继续 blocked。 |

每行都包含以下闭环列：

- managed owner state
- SafeHandle / GCHandle keep-alive
- native noncopyable owner storage
- native create/destroy symmetry
- attach/detach/clear control
- detach-before-release ordering
- no-throw destructor
- no-throw vtable
- managed exception capture
- exception-to-status mapping
- in-flight callback accounting
- borrowed pointer escape blocker
- opt-in runtime smoke readiness
- `PackageConsumerRuntimeProofRequired`
- `PackageConsumerRuntimeProofReady`

`PackageConsumerRuntimeProofRequired=True` 只是说明提升真实 callback runtime proof 前必须有 package-consumer 证据。聚合器中的 `PackageConsumerRuntimeProofReady=False` 是 family 级旧门禁值，不得用它否定后来独立捕获的子接口证据，也不得用单个子接口证据把整个 family 晋级完成。`IStreamReaderV2` 的当前证据入口是 `stream-reader-local-package-consumer-tutorial.md`；该证据仍是本地包结果，不是公开包 proof。

## Smoke Marker

`CallbackAllocatorSafeControlsSmokeRunner` 输出：

```text
CallbackOwnerClosureMatrix=callback-owner-closure-matrix;EvidenceKind=callback-owner-closure-matrix;RuntimeEvidenceKind=closure-matrix;RealCallbackRuntime=False;IsRealCallbackRuntimeProof=False;FamilyCount=5;...
```

该 smoke marker 可以帮助下一阶段快速定位 owner family 的缺口，但不能作为真实 TensorRT callback runtime proof。以下 direct callback deferred rows 作为历史覆盖记录继续保留；保留记录不表示对应接口没有独立的新实现或实机证据：

- `IGpuAllocator::allocate`
- `IGpuAllocator::free`
- `IGpuAllocator::deallocate`
- `IGpuAllocator::reallocate`
- `IGpuAsyncAllocator::allocateAsync`
- `IGpuAsyncAllocator::deallocateAsync`
- `IOutputAllocator::notifyShape`
- `IOutputAllocator::reallocateOutput`
- `IDebugListener::processDebugTensor`
- `IStreamReader::read`
- `IStreamReaderV2::read`
- `IStreamReaderV2::seek`
- `IStreamWriter::write`

## PackageConsumer / Readiness

本阶段已把矩阵接入 package-consumer/readiness 证据链：

- `eng\Test-BridgePackageConsumer.ps1` 的临时消费项目会引用 `TensorRtCallbackOwnerClosureMatrix.Evaluate(...)`，并把 `TensorRtCallbackOwnerClosureMatrix`、`TensorRtCallbackOwnerClosureMatrixResult`、`TensorRtCallbackOwnerClosureMatrixRow`、`FamilyCount`、`DesignGateReadyFamilyCount`、`ClosureReadyFamilyCount`、`RuntimeProofAttemptReadyFamilyCount`、`PackageConsumerRuntimeProofReadyFamilyCount`、`RuntimeProofBlocked` 和 `DeferredRowsStillRequired` 写入 high-level wrapper surface。
- `eng\Test-RuntimePackageReadiness.ps1` 输出 `callbackOwnerClosureMatrix` 证据对象，分类固定为 `evidenceKind=closure-matrix` / `runtimeEvidenceKind=closure-matrix`，并继续保持 `isRealCallbackRuntimeProof=false`。
- readiness 只用该对象确认 owner 缺口矩阵、文档、smoke、bridge surface 与 deferred row evidence 是否齐备；它不能作为发布 runtime proof，也不能删除 direct callback deferred 记录。

## 下一步

下一阶段应以矩阵中的 `NextWorkItem` 为入口，优先选择一个 family 做真实闭环：

1. DebugListener：先把 non-null attach 的 enable guard、native vtable install、真实 `processDebugTensor` invocation、detach/release 顺序和 package-consumer proof 串成一个 opt-in 路径。
2. OutputAllocator：补 native stable owner、no-throw vtable、device pointer ledger、CUDA stream / current memory policy，再做真实 `notifyShape/reallocateOutput` proof。
3. GpuAllocator / GpuAsyncAllocator：先补 line-specific attach/detach 和 device pointer / stream lifetime ledger，不直接启用分配接管。
4. StreamReaderWriter：`IStreamReaderV2` 已完成 TensorRT 10.11 owner-safe runtime 与本地双包证明；下一步只处理 TRT11 实机验证，并在能建立真实使用路径后分别评估 legacy `IStreamReader` 和 `IStreamWriter`，不得由 v2 结果外推。
