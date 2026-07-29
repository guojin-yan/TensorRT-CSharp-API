# Stream IO InterfaceInfo 安全设计门

## 目标

本设计门覆盖 `IStreamReader::getInterfaceInfo`、`IStreamReaderV2::getInterfaceInfo` 和 `IStreamWriter::getInterfaceInfo`。这些接口来自应用侧 stream reader/writer callback，不能直接暴露 native handle。

## 安全边界

- 只允许 copied interface metadata。
- 不暴露 `IStreamReader*`、`IStreamReaderV2*`、`IStreamWriter*`、`IntPtr` 或 `nint`。
- 不调用 read/write callback。
- `IStreamReader::read`、`IStreamReaderV2::read`、`IStreamReaderV2::seek` 和 `IStreamWriter::write` 继续 deferred。
- 对应 `getAPILanguage` 只进入 owner-scoped 候选清单，不能绕过 managed-owned stream owner 生命周期直接提升。
- read/write buffer ownership、seek/tell state、stream owner lifetime 继续 deferred。
- 本设计门不是 runtime proof。

## Owner 台账

未来要实现真实 stream bridge，至少需要先闭合这些条件：

- 使用 `SafeHandle` 表达 `TensorRtStreamReaderOwnerHandle` / `TensorRtStreamWriterOwnerHandle`，并由 native 非复制 owner storage 承载 vtable。
- managed owner state 通过 `GCHandle` pin 住，且只能在 native detach 之后释放。
- native create/destroy 必须对称，析构和 vtable callback 必须 no-throw。
- managed callback exception 必须映射为 status/diagnostic，不能跨 ABI 抛出。
- read/write buffer 必须为 caller-owned 临时缓冲，不能被 native 或 managed owner 保留。
- seek/tell 状态必须 owner-scoped，并明确 thread-safety / reentrancy 规则。
- release 前必须先 detach；detach/release 顺序需要质量测试和 smoke 输出证明。

## 当前证据

- 设计门 evaluator：`src/JYPPX.TensorRtSharp/Callbacks/Monitoring/TensorRtStreamIoInterfaceInfoDesignGate.cs`
- 设计门 result：`src/JYPPX.TensorRtSharp/Callbacks/Monitoring/TensorRtStreamIoInterfaceInfoDesignGateResult.cs`
- 测试：`tests/JYPPX.ProjectQuality.Tests/StreamIoInterfaceInfoDesignGateTests.cs`
- 机器清单：`artifacts/interface-coverage/deferred-readonly-candidate-list.json`

## 下一步

只有在 stream owner lifetime、buffer ownership、seek/tell 状态和跨 ABI no-throw 语义完成后，才能考虑真实 callback bridge。当前阶段继续保留 direct deferred history。
