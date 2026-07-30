# Runtime Deserialization Boundary Precheck

`runtime-deserialization-boundary-precheck` 用来收口 `IRuntime::deserializeCudaEngine`、`IRuntime::deserializeCudaEngineV2` 和 `IRuntime::loadRuntime` 周围的安全边界。它不是 runtime execution proof；它记录当前 C# 高层 `TensorRtRuntime.Deserialize(...)` 的安全形态，并区分已经由 scoped-buffer native bridge 覆盖的 `deserializeCudaEngine` 与仍需 callback/ownership 设计的 V2、`loadRuntime` 行。

源码职责分为 `TensorRtRuntimeDeserializationBoundaryPrecheck.cs` 与
`TensorRtRuntimeDeserializationBoundaryPrecheckResult.cs`：前者只拥有两组 evaluation 入口，后者拥有 result
构造、公开属性、阻塞项、诊断和 `ToString`。该归类不改变 runtime-precheck/non-proof 边界。

## 当前可用边界

- `TensorRtRuntime.Deserialize(byte[])` 通过托管 `byte[]` 调用 native `jyppx_trt*_runtime_deserialize_engine`。
- `TensorRtRuntime.Deserialize(ArraySegment<byte>)` 会先复制成精确长度的托管数组。
- `TensorRtRuntime.Deserialize(ReadOnlySpan<byte>)` 会先复制到托管数组。
- `TensorRtRuntime.Deserialize(Stream)` 会先复制到 `MemoryStream`，不是 `IStreamReader` callback bridge。
- `TensorRtRuntime.DeserializeFromFile(string)` 读取托管字节后复用 byte-array 入口。
- `TensorRtRuntime.Deserialize(TensorRtHostMemory)` 使用 bridge-owned host memory handle，不向 public API 暴露 `IHostMemory*`。

预检输出 `RuntimeEvidenceKind=runtime-precheck`、`ManagedByteArrayDeserializeReady=True`、`ManagedStreamDeserializeReady=True`、`HostMemoryDeserializeReady=True`、`SerializedBufferCopiedBeforeInterop=True`、`PinnedBufferScopedToInteropCall=True`、`BorrowedSerializedBufferEscaped=False`、`EngineHandleOwnedByWrapper=True`、`EnginePointerExposed=False`、`DirectDeserializeCudaEngineRowsDeferred=False`、`DirectDeserializeCudaEngineRowsImplemented=True`、`DirectDeserializeCudaEngineV2RowsDeferred=True`、`LoadRuntimeDeferred=True` 和 `RuntimeProofBlocked=True`。

## 为什么仍是 not proof

这个 precheck 不会运行 TensorRT engine，不会证明 plugin library dependency 已完整，不会证明 CUDA driver 与当前 runtime package 匹配，也不会解除 `loadRuntime` 返回 runtime 的 ownership 问题。它只是证明当前 public C# surface 对 serialized buffer 和 returned engine handle 没有裸指针逃逸。

因此：

- direct `IRuntime::deserializeCudaEngine` 由调用期间 pinned buffer 和 bridge-owned engine handle 覆盖，不再是 deferred-only。
- direct `IRuntime::deserializeCudaEngineV2` 仍保留 deferred 行。
- direct `IRuntime::loadRuntime` 仍保留 deferred 行。
- package-consumer runtime proof 仍必须来自真实 full runtime consumer smoke。
- `blocked-by-cuda-driver` 不是 smoke passed。

## 下一步

下一阶段可以在这个 precheck 之上继续推进：

1. 继续完善 [Runtime Deserialization Dependency Diagnostics](runtime-deserialization-dependency-diagnostics.md)，让反序列化失败时能区分缺 plugin、缺 host code、缺 runtime package asset、dependency-probe-only 和 CUDA driver 阻塞。
2. 为 `loadRuntime` 建立 returned runtime ownership 模型，不能暴露 borrowed runtime pointer。
3. 在 full package consumer smoke 里使用真实 serialized engine 文件验证 `Deserialize(byte[])` 或 `DeserializeFromFile`。
4. 只有当 full package consumer report 输出 promotable `package-consumer-runtime` proof 后，才讨论提升 direct deferred 行。
