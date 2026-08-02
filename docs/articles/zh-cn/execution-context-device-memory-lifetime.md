# Execution Context Device Memory 生命周期

TensorRT execution context 可以使用调用方分配的 activation device memory。TensorRtSharp4.0 保留既有 `SetDeviceMemory(CudaMemory)` 签名，并为 TensorRT 10/11 增加显式 `SetDeviceMemoryV2(CudaMemory)`。两条 API 都只接收 `CudaMemory` owner，不暴露 `IntPtr`、CUDA pointer 或 borrowed `SafeHandle`。

## 版本行为

| TensorRT | 兼容 API | 显式 V2 API | native 调用 |
| --- | --- | --- | --- |
| 8.6 | `SetDeviceMemory` | 不支持 | `IExecutionContext::setDeviceMemory(pointer)` |
| 10.11 | `SetDeviceMemory` | `SetDeviceMemoryV2` | `IExecutionContext::setDeviceMemoryV2(pointer,size)` |
| 11.0 | `SetDeviceMemory` | `SetDeviceMemoryV2` | `IExecutionContext::setDeviceMemoryV2(pointer,size)` |

TRT10/11 的 `ClearDeviceMemory()` 调用 `setDeviceMemoryV2(nullptr,0)`。TRT8 没有 size-aware clear contract，因此会明确返回 `NotSupportedException`，不会把一个缺少 size 语义的调用伪装成跨版本一致行为。

## Owner-Safe 用法

```csharp
using TensorRtExecutionContext context = engine.CreateExecutionContextWithoutDeviceMemory();

int requiredBytes = checked((int)engine.DeviceMemorySizeInBytes);
CudaMemory activationMemory = new CudaMemory(requiredBytes);
context.SetDeviceMemoryV2(activationMemory);

// Context 已取得 SafeHandle lease；这里只释放调用方 wrapper。
activationMemory.Dispose();

context.SetTensorAddress("input", inputMemory);
context.SetTensorAddress("output", outputMemory);
context.EnqueueAsync(stream);
stream.Synchronize();
```

`SetDeviceMemory` 或 `SetDeviceMemoryV2` 会先执行 `DangerousAddRef`，native set 成功后才提交新 lease。native 调用失败时，新 lease 会回滚，不会覆盖当前 binding。

## 重绑与清理

TensorRT 文档允许 enqueue 后异步使用 execution-context memory。托管层无法从任意 `ClearDeviceMemory` 或重绑调用推断之前的 stream 是否已经完成，因此不能在重绑成功时立即释放旧 allocation。

本实现采用保守生命周期：

- 当前 binding 由 `_deviceMemoryLease` 保持；
- rebind 或 clear 后，旧 lease 进入 retired 集合；
- retired lease 不再作为当前 binding，但继续阻止 native allocation 提前释放；
- `TensorRtExecutionContext.Dispose()` 先销毁 native context，再释放当前和 retired lease；
- lease 自带 finalizer 回退，避免未正确 dispose context 时永久保留 `DangerousAddRef`。

这会让频繁重绑的 context 在释放前保留历史 allocation。它是有意的安全取舍：调用方应按 engine/profile 复用固定 activation memory，避免在长生命周期 context 上无界重绑。

可以通过以下 pointer-free 状态检查当前托管 ownership：

- `HasBoundDeviceMemory`
- `BoundDeviceMemorySizeInBytes`
- `RetainedDeviceMemoryLeaseCount`

这些属性不查询或返回 native pointer，也不是 inference 已完成的证明。

## 真实 Smoke

`NetworkConvolutionScaleSmokeRunner` 使用 `CreateExecutionContextWithoutDeviceMemory()`，依次执行：

1. 通过兼容 API 绑定第一块 memory，并立即 dispose 调用方 wrapper；
2. TRT10/11 通过 V2 API 重绑第二块 memory，再次立即 dispose wrapper；
3. 绑定 input/output，执行真实 convolution/scale/padding enqueue；
4. 同步并校验输出；
5. TRT10/11 clear 当前 binding，同时验证 retired lease 数量未下降。

```powershell
$env:JYPPX_NATIVE_BRIDGE_PATH = '<jyppxtrtbridge.dll>'
dotnet run --project .\smoke\NetworkConvolutionScaleSmokeRunner -- --tensor-rt-line 10
```

成功标记包括 `CallerWrappersDisposed=True`、`RetainedLeases=2` 和 `ConvolutionScalePaddingOutputMatch=True`。

### 当前本机验证矩阵

| TensorRT / CUDA | native bridge | 真实运行结果 | 证明范围 |
| --- | --- | --- | --- |
| 8.6.1 / 12.1 | 复用已构建 bridge，TRT8 native 路径本批未修改 | 通过；兼容入口连续重绑，调用方 wrapper 释放后 enqueue 与输出比较成功 | TRT8 兼容入口和 owner lease |
| 10.11.0 / 12.9 | 本批 Release 构建通过 | 通过；兼容入口、显式 V2、clear、enqueue 与输出比较均成功 | TRT10 owner lease、V2 和 clear |
| 11.0.0 / 13.2 | 本批 Release 构建通过，新增导出已进入 DLL | 受阻；`createInferRuntime returned a null TensorRT object`，尚未进入 device-memory 调用 | 只能证明头文件、ABI 和 bridge 编译，不能声称 TRT11 runtime 已通过 |

结构化记录位于
`artifacts/interface-coverage/execution-context-device-memory-owner-local-runtime-evidence.json`。记录保存 bridge 与 smoke assembly 的 SHA256，但不保存 device pointer 或完整运行日志。

这个 smoke 直接用 TensorRT API 构造 convolution/scale/padding 网络，不读取深度学习模型，因此没有模型获取、转换或 ONNX 暂存步骤。需要模型的其他演示仍必须把转换后的 ONNX 放在仓库外层 `models` 目录，并在对应文章中写明获取与转换方式。

## 证据边界

这条 smoke 可以证明当前 source-tree bridge、CUDA memory owner 和 TensorRT context 在指定主机、且仅限通过的 TRT8/TRT10 版本线上的生命周期与真实推理结果。它不能替代 TRT11 runtime、公开包 clean consumer、Linux runner、post-publish 或 Owner release acceptance；这些 gate 继续保持独立。
