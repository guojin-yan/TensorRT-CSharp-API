# CUDA IPC 导出与 Owner-Safe Import

`CudaEvent.ExportIpcToken()` 与 `CudaMemory.ExportIpcToken()` 会把 CUDA 的 opaque IPC
导出 handle 复制成不可变托管值。公开 API 不返回 CUDA event handle、device pointer、
`IntPtr`、`UIntPtr` 或 `SafeHandle`。

## 导出 Event Token

CUDA 要求 IPC event 同时带有两个标志：

```csharp
using CudaEvent cudaEvent = new CudaEvent(
    CudaEventCreationFlags.Interprocess | CudaEventCreationFlags.DisableTiming);

CudaIpcExportToken token = cudaEvent.ExportIpcToken();
byte[] serialized = token.ToArray();
Console.WriteLine($"Kind={token.Kind} Length={token.Length}");
```

另一个进程使用 token 期间，源 `CudaEvent` 必须保持存活。CUDA 文档明确说明：先销毁
exported event，再操作 imported event，行为未定义。

## 导出 Device Memory Token

```csharp
using CudaMemory memory = new CudaMemory(4096);
CudaIpcMemoryExportDescriptor descriptor = memory.ExportIpcDescriptor();
byte[] serialized = descriptor.Token.ToArray();
int allocationSize = descriptor.SizeInBytes;

// 建立 share handle 后，再提交需要由 importer 观察的写入。
memory.Fill(0x2A);
```

只有 `CudaMemory(int)` 同步路径得到的 `cudaMalloc` 基址可以导出。managed memory、
`cudaMallocAsync` 与 memory-pool allocation 会 fail closed。所有 imported mapping 停止使用前，
源 `CudaMemory` 必须保持存活。

token 字节与精确 allocation size 必须作为一个整体在可信通道中传输。接收进程重建不可变
descriptor，并获得明确拥有 mapping 的 wrapper：

```csharp
CudaIpcMemoryExportDescriptor received =
    CudaIpcMemoryExportDescriptor.FromBytes(serialized, allocationSize);
using CudaMemory imported = CudaMemory.ImportIpcDescriptor(received);
Console.WriteLine($"Imported={imported.IsIpcImported} Size={imported.SizeInBytes}");
```

imported `CudaMemory.Dispose()` 固定路由到 `cudaIpcCloseMemHandle`。CUDA 没有异步 IPC close，
所以 `FreeAsync` 会 fail closed。普通 allocation 继续使用 `cudaFree`，两条释放路径不能混用。

## 导入 Event

接收进程从传输字节重建 event token，并拥有进程内 event wrapper：

```csharp
CudaIpcExportToken received =
    CudaIpcExportToken.FromBytes(CudaIpcExportTokenKind.Event, serializedEventToken);
using CudaEvent importedEvent = CudaEvent.ImportIpcToken(received);
importedEvent.Synchronize();
```

imported event 使用 `cudaEventDestroy` 释放。其整个生命周期内，导出进程必须保持源 event 存活。

Windows/WDDM 上建议固定采用以下顺序：先导出 memory descriptor 与 event token，再提交 producer
写入并在写入后 record 已导出的 event，最后由 importer 等待该 event。仓库 smoke 会锁定此顺序，
因为在已完成写入之后才建立 Windows 兼容 share handle 时，验证机的 imported mapping 无法观察
此前内容。

`ToArray()` 每次返回新副本。`ToString()` 只显示 kind 与 length；日志中不打印 token
内容，也不要把 token 当成普通诊断字段上传。

## Ownership 与 Proof 边界

bridge 现在通过 owner-safe wrapper 实现 `cudaIpcOpenEventHandle`、
`cudaIpcOpenMemHandle` 与 `cudaIpcCloseMemHandle`。memory open flags 固定为
`cudaIpcMemLazyEnablePeerAccess`；native 校验 64 字节 token，托管 descriptor 保留精确长度，
普通 free 路径遇到 imported mapping 会 fail closed。token/size 的传输通道、进程信任、device
选择和 exporter 生命周期仍由应用负责。

仓库内跨进程 smoke 属于真实本地 runtime evidence，但不是 packed-package consumer proof、
公开部署证明、发布批准或 post-publish verification。

NVIDIA 将 Windows CUDA IPC 定义为兼容用途支持，但不建议用于性能敏感设计。实际部署前应
检查设备 IPC 能力，并在目标机器上验证真实多进程流程。
