using System;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public readonly partial struct CudaMemoryPool
{
    /// <summary>
    /// Sets access permissions for this memory pool from another CUDA device.
    /// 设置另一个 CUDA 设备访问当前 memory pool 的权限。
    /// </summary>
    /// <param name="deviceOrdinal">The device ordinal whose access should be updated. 需要更新访问权限的设备序号。</param>
    /// <param name="flags">The requested access flags. 目标访问权限标志。</param>
    /// <remarks>
    /// This maps to CUDA's <c>cudaMemPoolSetAccess</c> safe subset with a device-location descriptor.
    /// 该方法映射到 CUDA <c>cudaMemPoolSetAccess</c> 的设备位置描述符安全子集。
    /// </remarks>
    public void SetAccess(int deviceOrdinal, CudaMemoryPoolAccessFlags flags)
    {
        NativeCudaApi.SetMemoryPoolAccess(_handle, deviceOrdinal, (uint)flags);
    }

    /// <summary>
    /// Gets access permissions for this memory pool from a CUDA device.
    /// 查询指定 CUDA 设备访问当前 memory pool 的权限。
    /// </summary>
    /// <param name="deviceOrdinal">The device ordinal to query. 要查询的设备序号。</param>
    /// <returns>The access flags reported by CUDA. CUDA 返回的访问权限标志。</returns>
    public CudaMemoryPoolAccessFlags GetAccess(int deviceOrdinal)
    {
        return (CudaMemoryPoolAccessFlags)NativeCudaApi.GetMemoryPoolAccess(_handle, deviceOrdinal);
    }

}
