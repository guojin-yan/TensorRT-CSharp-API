using System;
using JYPPX.CudaSharp.Internal.Handles;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static partial class NativeCudaApi
{
    public static CudaIpcExportToken ExportEventIpcToken(SafeCudaEventHandle eventHandle)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_ipc_export_event_token_safe(
            eventHandle,
            Array.Empty<byte>(),
            UIntPtr.Zero,
            out UIntPtr requiredSize));
        byte[] bytes = AllocateIpcTokenBuffer(requiredSize);
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_ipc_export_event_token_safe(
            eventHandle,
            bytes,
            new UIntPtr((uint)bytes.Length),
            out UIntPtr copiedSize));
        ValidateCopiedIpcTokenSize(bytes, copiedSize);
        return new CudaIpcExportToken(CudaIpcExportTokenKind.Event, bytes);
    }

    public static CudaIpcExportToken ExportMemoryIpcToken(SafeCudaMemoryHandle memory)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_ipc_export_memory_token_safe(
            memory,
            Array.Empty<byte>(),
            UIntPtr.Zero,
            out UIntPtr requiredSize));
        byte[] bytes = AllocateIpcTokenBuffer(requiredSize);
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_ipc_export_memory_token_safe(
            memory,
            bytes,
            new UIntPtr((uint)bytes.Length),
            out UIntPtr copiedSize));
        ValidateCopiedIpcTokenSize(bytes, copiedSize);
        return new CudaIpcExportToken(CudaIpcExportTokenKind.Memory, bytes);
    }

    private static byte[] AllocateIpcTokenBuffer(UIntPtr requiredSize)
    {
        ulong size = requiredSize.ToUInt64();
        if (size == 0 || size > 4096)
        {
            throw new InvalidOperationException($"CUDA reported an invalid IPC export token size: {size}.");
        }

        return new byte[checked((int)size)];
    }

    private static void ValidateCopiedIpcTokenSize(byte[] bytes, UIntPtr copiedSize)
    {
        if (copiedSize.ToUInt64() != (ulong)bytes.Length)
        {
            throw new InvalidOperationException(
                $"CUDA IPC export token size changed between query and copy: expected {bytes.Length}, copied {copiedSize.ToUInt64()}.");
        }
    }
}
