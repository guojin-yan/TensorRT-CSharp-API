using System;
using JYPPX.CudaSharp.Internal.Handles;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static partial class NativeCudaApi
{
    public static SafeCudaEventHandle ImportEventIpcToken(CudaIpcExportToken token)
    {
        byte[] bytes = GetIpcTokenBytes(token, CudaIpcExportTokenKind.Event, nameof(token));
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_ipc_import_event_safe(
            bytes,
            new UIntPtr((uint)bytes.Length),
            out SafeCudaEventHandle eventHandle));
        return eventHandle;
    }

    public static SafeCudaMemoryHandle ImportMemoryIpcDescriptor(CudaIpcMemoryExportDescriptor descriptor)
    {
        if (descriptor == null)
        {
            throw new ArgumentNullException(nameof(descriptor));
        }

        byte[] bytes = GetIpcTokenBytes(
            descriptor.Token,
            CudaIpcExportTokenKind.Memory,
            nameof(descriptor));
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_ipc_import_memory_safe(
            bytes,
            new UIntPtr((uint)bytes.Length),
            new UIntPtr((uint)descriptor.SizeInBytes),
            out SafeCudaMemoryHandle memoryHandle));
        memoryHandle.MarkIpcImported();
        return memoryHandle;
    }

    private static byte[] GetIpcTokenBytes(
        CudaIpcExportToken token,
        CudaIpcExportTokenKind requiredKind,
        string parameterName)
    {
        if (token == null)
        {
            throw new ArgumentNullException(parameterName);
        }

        if (token.Kind != requiredKind)
        {
            throw new ArgumentException(
                $"CUDA IPC token kind must be {requiredKind}, but was {token.Kind}.",
                parameterName);
        }

        return token.ToArray();
    }
}
