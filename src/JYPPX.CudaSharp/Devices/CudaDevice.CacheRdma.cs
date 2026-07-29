using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public static partial class CudaDevice
{
    /// <summary>
    /// Gets the maximum linear texture width supported for the specified descriptor.
    /// 获取指定通道描述符支持的最大 linear texture 宽度。
    /// </summary>
    /// <param name="descriptor">The CUDA channel descriptor. CUDA 通道描述符。</param>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <returns>The maximum width in bytes or CUDA-defined units. 以字节或 CUDA 定义单位表示的最大宽度。</returns>
    public static ulong GetTexture1DLinearMaxWidth(CudaChannelFormatDescriptor descriptor, int ordinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetTexture1DLinearMaxWidth(descriptor, ordinal);
    }

    /// <summary>
    /// Resets the persisting L2 cache state for the current device.
    /// 重置当前设备的 persisting L2 cache 状态。
    /// </summary>
    public static void ResetPersistingL2Cache()
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.ResetPersistingL2Cache();
    }

    /// <summary>
    /// Flushes GPU Direct RDMA writes for the selected target and scope.
    /// 为指定目标与范围刷新 GPU Direct RDMA 写入。
    /// </summary>
    /// <param name="target">The RDMA target to flush. 要刷新的 RDMA 目标。</param>
    /// <param name="scope">The RDMA visibility scope. RDMA 可见性范围。</param>
    public static void FlushGpuDirectRdmaWrites(CudaGpuDirectRdmaWritesTarget target, CudaGpuDirectRdmaWritesScope scope)
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.FlushGpuDirectRdmaWrites(target, scope);
    }

}
