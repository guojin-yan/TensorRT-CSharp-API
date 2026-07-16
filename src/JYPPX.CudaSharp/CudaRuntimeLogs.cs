using System;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Provides copied access to CUDA runtime logs without exposing callbacks or native buffers.
/// 提供 CUDA runtime 日志的复制型访问，不暴露 callback 或 native buffer。
/// </summary>
public static class CudaRuntimeLogs
{
    /// <summary>The maximum buffer size accepted by CUDA's log dump API. CUDA 日志 dump API 接受的最大缓冲区大小。</summary>
    public const int MaximumBufferSize = 25600;

    /// <summary>Gets a cursor positioned at the current tail of the CUDA log buffer. 获取位于 CUDA 日志缓冲区当前末尾的 cursor。</summary>
    public static CudaLogCursor GetCurrentCursor()
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetCudaLogCursor();
    }

    /// <summary>Dumps available CUDA logs into a managed UTF-8 snapshot. 将可用 CUDA 日志 dump 到托管 UTF-8 快照。</summary>
    public static CudaLogSnapshot DumpToMemory(int bufferSize = MaximumBufferSize)
    {
        ValidateBufferSize(bufferSize);
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.DumpCudaLogsToMemory(false, default, bufferSize);
    }

    /// <summary>Dumps CUDA logs starting at a cursor and returns the advanced cursor. 从 cursor 开始 dump CUDA 日志并返回推进后的 cursor。</summary>
    public static CudaLogSnapshot DumpToMemory(CudaLogCursor cursor, int bufferSize = MaximumBufferSize)
    {
        ValidateBufferSize(bufferSize);
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.DumpCudaLogsToMemory(true, cursor, bufferSize);
    }

    /// <summary>Dumps all available CUDA logs to a UTF-8 file path. 将全部可用 CUDA 日志 dump 到 UTF-8 文件路径。</summary>
    public static void DumpToFile(string path)
    {
        ValidatePath(path);
        NativeBridgeLoader.EnsureInitialized();
        _ = NativeCudaApi.DumpCudaLogsToFile(false, default, path);
    }

    /// <summary>Dumps CUDA logs from a cursor to a UTF-8 file path and returns the advanced cursor. 从 cursor 开始将 CUDA 日志 dump 到 UTF-8 文件路径，并返回推进后的 cursor。</summary>
    public static CudaLogCursor DumpToFile(CudaLogCursor cursor, string path)
    {
        ValidatePath(path);
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.DumpCudaLogsToFile(true, cursor, path)!.Value;
    }

    private static void ValidateBufferSize(int bufferSize)
    {
        if (bufferSize <= 0 || bufferSize > MaximumBufferSize)
        {
            throw new ArgumentOutOfRangeException(nameof(bufferSize));
        }
    }

    private static void ValidatePath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("CUDA log output path must not be null or empty.", nameof(path));
        }
    }
}
