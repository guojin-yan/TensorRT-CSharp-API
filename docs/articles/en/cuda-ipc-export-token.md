# CUDA IPC Export And Owner-Safe Import

`CudaEvent.ExportIpcToken()` and `CudaMemory.ExportIpcToken()` copy CUDA's opaque IPC export
handles into immutable managed values. They never expose a CUDA event handle, device pointer,
`IntPtr`, `UIntPtr`, or `SafeHandle`.

## Export an event

CUDA requires an IPC event to use both flags:

```csharp
using CudaEvent cudaEvent = new CudaEvent(
    CudaEventCreationFlags.Interprocess | CudaEventCreationFlags.DisableTiming);

CudaIpcExportToken token = cudaEvent.ExportIpcToken();
byte[] serialized = token.ToArray();
Console.WriteLine($"Kind={token.Kind} Length={token.Length}");
```

Keep the source owner alive while another process uses the exported token. Destroying the source
event first makes subsequent imported-event operations undefined according to CUDA.

## Export device memory

```csharp
using CudaMemory memory = new CudaMemory(4096);
CudaIpcMemoryExportDescriptor descriptor = memory.ExportIpcDescriptor();
byte[] serialized = descriptor.Token.ToArray();
int allocationSize = descriptor.SizeInBytes;

// Submit writes that the importer must observe after creating the share handle.
memory.Fill(0x2A);
```

Only the base allocation returned by the synchronous `CudaMemory(int)` path is accepted. Managed
memory, `cudaMallocAsync`, and pool allocations fail closed. Keep the source owner alive until every
imported mapping has stopped using it.

Transport the token bytes and exact allocation size as one authenticated descriptor. The receiving
process reconstructs an immutable descriptor and receives an owner wrapper:

```csharp
CudaIpcMemoryExportDescriptor received =
    CudaIpcMemoryExportDescriptor.FromBytes(serialized, allocationSize);
using CudaMemory imported = CudaMemory.ImportIpcDescriptor(received);
Console.WriteLine($"Imported={imported.IsIpcImported} Size={imported.SizeInBytes}");
```

`CudaMemory.Dispose()` routes imported mappings to `cudaIpcCloseMemHandle`. `FreeAsync` is rejected
because CUDA does not define asynchronous IPC close. A normal allocation still uses `cudaFree`; the
two release paths cannot be mixed.

## Import an event

The receiving process reconstructs the event token and owns the process-local event wrapper:

```csharp
CudaIpcExportToken received =
    CudaIpcExportToken.FromBytes(CudaIpcExportTokenKind.Event, serializedEventToken);
using CudaEvent importedEvent = CudaEvent.ImportIpcToken(received);
importedEvent.Synchronize();
```

The imported event is released with `cudaEventDestroy`. The exporting event must remain alive for
the entire imported-event lifetime.

For the most reliable Windows/WDDM sequence, export the memory descriptor and event token before
submitting producer writes, record the exported event after those writes, and let the importer wait
on that event. The repository smoke locks this order because creating the Windows compatibility
share after an already-completed write did not expose that earlier content on the validated WDDM
host.

`ToArray()` returns a new copy. `ToString()` deliberately reports only kind and length; application
logs should not print the token bytes.

## Ownership and proof boundary

The bridge now implements `cudaIpcOpenEventHandle`, `cudaIpcOpenMemHandle`, and
`cudaIpcCloseMemHandle` through owner-safe wrappers. It fixes memory-open flags to
`cudaIpcMemLazyEnablePeerAccess`, validates the 64-byte token shape, keeps allocation length in an
immutable transport descriptor, and fails closed when a normal free path is used for an imported
mapping. The token/size transport channel, process trust, device selection, and exporter lifetime
remain application responsibilities.

The repository cross-process smoke is real local runtime evidence. It is not packed-package
consumer proof, public deployment proof, publication approval, or post-publish verification.

On Windows, NVIDIA documents CUDA IPC as supported for compatibility but not recommended for
performance-sensitive designs. Check device IPC capability and validate the real multi-process
workflow on the deployment host.
