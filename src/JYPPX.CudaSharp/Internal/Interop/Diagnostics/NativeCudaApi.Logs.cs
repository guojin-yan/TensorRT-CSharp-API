using System;
using System.Text;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static partial class NativeCudaApi
{
    public static CudaLogCursor GetCudaLogCursor()
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_logs_get_current_cursor_safe(out uint cursor));
        return new CudaLogCursor(cursor);
    }

    public static CudaLogSnapshot DumpCudaLogsToMemory(bool useCursor, CudaLogCursor cursor, int bufferSize)
    {
        byte[] buffer = new byte[bufferSize];
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_logs_dump_to_memory_safe(
            useCursor ? 1 : 0,
            cursor.Value,
            buffer,
            new UIntPtr((uint)buffer.Length),
            out UIntPtr writtenSize,
            out uint nextCursor));

        ulong written = writtenSize.ToUInt64();
        if (written > (ulong)buffer.Length || written > int.MaxValue)
        {
            throw new InvalidOperationException("CUDA returned an invalid log byte count.");
        }

        int byteCount = (int)written;
        string text = byteCount == 0 ? string.Empty : Encoding.UTF8.GetString(buffer, 0, byteCount);
        return new CudaLogSnapshot(text, byteCount, useCursor ? new CudaLogCursor(nextCursor) : (CudaLogCursor?)null);
    }

    public static CudaLogCursor? DumpCudaLogsToFile(bool useCursor, CudaLogCursor cursor, string path)
    {
        using Utf8Interop.Utf8StringScope pathUtf8 = Utf8Interop.ToNativeString(path);
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_logs_dump_to_file_safe(
            useCursor ? 1 : 0,
            cursor.Value,
            pathUtf8.Pointer,
            out uint nextCursor));
        return useCursor ? new CudaLogCursor(nextCursor) : (CudaLogCursor?)null;
    }
}
