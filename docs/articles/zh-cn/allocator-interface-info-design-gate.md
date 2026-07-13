# Allocator InterfaceInfo 安全设计门

## 目标

本设计门覆盖 `IGpuAllocator::getInterfaceInfo`、`IGpuAsyncAllocator::getInterfaceInfo` 和 `IOutputAllocator::getInterfaceInfo`。这些接口涉及应用侧 allocator callback 和 device memory ownership，因此只允许 copied metadata / owner-scoped diagnostics。

## 安全边界

- 不暴露 allocator handle。
- 不暴露 device memory pointer。
- 不启用 allocate/deallocate/reallocate/notifyShape callback。
- async stream lifetime、device memory release policy 和 output buffer ownership 继续 deferred。
- 本设计门不是 runtime proof。

## 当前证据

- 设计门：`src/JYPPX.TensorRtSharp/TensorRtAllocatorInterfaceInfoDesignGate.cs`
- 相关门：`src/JYPPX.TensorRtSharp/TensorRtOutputAllocatorAttachDetachDesignGate.cs`
- 测试：`tests/JYPPX.ProjectQuality.Tests/AllocatorInterfaceInfoDesignGateTests.cs`
- 机器清单：`artifacts/interface-coverage/deferred-readonly-candidate-list.json`

## 下一步

后续只有在 allocator owner、device memory ledger、stream lifetime、detach-before-release 和 runtime proof 都闭环后，才能考虑真实 allocator callback bridge。
