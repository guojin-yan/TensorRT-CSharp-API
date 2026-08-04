using System;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Managed wrapper around a TensorRT builder configuration.
/// TensorRT builder 配置的托管封装。
/// </summary>
public sealed partial class TensorRtBuilderConfig
{
    /// <summary>
    /// Sets a TensorRT memory-pool size limit.
    /// 设置一个 TensorRT memory pool 大小上限。
    /// </summary>
    /// <param name="pool">The TensorRT memory pool. TensorRT 内存池。</param>
    /// <param name="bytes">The size limit in bytes. 大小上限，单位为字节。</param>
    public void SetMemoryPoolLimit(TensorRtMemoryPoolType pool, ulong bytes)
    {
        NativeBridgeApi.SetMemoryPoolLimit(Line, _handle, pool, bytes);
    }

    /// <summary>
    /// Gets a TensorRT memory-pool size limit.
    /// 获取一个 TensorRT memory pool 大小上限。
    /// </summary>
    /// <param name="pool">The TensorRT memory pool. TensorRT 内存池。</param>
    /// <returns>The size limit in bytes. 大小上限，单位为字节。</returns>
    public ulong GetMemoryPoolLimit(TensorRtMemoryPoolType pool)
    {
        return NativeBridgeApi.GetMemoryPoolLimit(Line, _handle, pool);
    }

}
