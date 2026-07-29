using System;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Identifies a CUDA device memory pool without exposing the raw native handle to ordinary users.
/// 标识一个 CUDA 设备内存池，同时避免向普通用户暴露原生句柄。
/// </summary>
public readonly partial struct CudaMemoryPool
{
    private readonly ulong _handle;

    internal CudaMemoryPool(ulong handle, int deviceOrdinal)
    {
        if (handle == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(handle));
        }

        _handle = handle;
        DeviceOrdinal = deviceOrdinal;
    }

    /// <summary>
    /// Gets the CUDA device ordinal that owns this memory pool.
    /// 获取拥有当前内存池的 CUDA 设备序号。
    /// </summary>
    public int DeviceOrdinal { get; }

    internal ulong Handle => _handle;

}
