# CUDA IPC Export Tokens

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
CudaIpcExportToken token = memory.ExportIpcToken();
byte[] serialized = token.ToArray();
```

Only the base allocation returned by the synchronous `CudaMemory(int)` path is accepted. Managed
memory, `cudaMallocAsync`, and pool allocations fail closed. Keep the source owner alive until every
imported mapping has stopped using it.

`ToArray()` returns a new copy. `ToString()` deliberately reports only kind and length; application
logs should not print the token bytes.

## Deliberate boundary

This batch does not implement `cudaIpcOpenEventHandle`, `cudaIpcOpenMemHandle`, or
`cudaIpcCloseMemHandle`. Importing creates process-local resources and requires a separate design for
device affinity, peer access, reference counts, recovery, and cleanup. Export success is not
cross-process runtime proof or package-consumer proof.

On Windows, NVIDIA documents CUDA IPC as supported for compatibility but not recommended for
performance-sensitive designs. Check device IPC capability and validate the real multi-process
workflow on the deployment host.
