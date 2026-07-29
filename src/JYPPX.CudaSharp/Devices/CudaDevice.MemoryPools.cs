using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public static partial class CudaDevice
{
    /// <summary>
    /// Gets free and total memory information for the current CUDA device.
    /// 获取当前 CUDA 设备的空闲与总显存信息。
    /// </summary>
    /// <returns>The current CUDA memory information snapshot. 当前 CUDA 显存信息快照。</returns>
    public static CudaMemoryInfo GetMemoryInfo()
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaMemoryInfo info = NativeCudaApi.GetMemoryInfo();
        return new CudaMemoryInfo(info.FreeBytes, info.TotalBytes);
    }

    /// <summary>
    /// Tries to read CUDA memory information without throwing when the runtime is unavailable.
    /// 在 CUDA runtime 不可用时以诊断字符串返回失败，而不是抛出异常。
    /// </summary>
    /// <param name="memoryInfo">The memory information when the query succeeds. 查询成功时返回的显存信息。</param>
    /// <param name="diagnostic">An empty string on success, or the CUDA bridge diagnostic on failure. 成功时为空字符串，失败时为 CUDA 桥接诊断。</param>
    /// <returns><c>true</c> when memory information was read successfully. 成功读取显存信息时返回 <c>true</c>。</returns>
    public static bool TryGetMemoryInfo(out CudaMemoryInfo? memoryInfo, out string diagnostic)
    {
        try
        {
            memoryInfo = GetMemoryInfo();
            diagnostic = string.Empty;
            return true;
        }
        catch (CudaException exception)
        {
            memoryInfo = null;
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Gets a CUDA memory pressure snapshot for the current device.
    /// 获取当前 CUDA 设备的显存压力快照。
    /// </summary>
    /// <returns>A memory pressure snapshot for deployment diagnostics. 用于部署诊断的显存压力快照。</returns>
    public static CudaMemoryPressureSnapshot GetMemoryPressureSnapshot()
    {
        int ordinal = Current;
        return new CudaMemoryPressureSnapshot(ordinal, GetMemoryInfo(), GetProperties(ordinal));
    }

    /// <summary>
    /// Gets the default CUDA memory pool for a device.
    /// 获取指定设备的默认 CUDA memory pool。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <returns>The default memory pool wrapper. 默认 memory pool 封装。</returns>
    public static CudaMemoryPool GetDefaultMemoryPool(int ordinal)
    {
        return CudaMemoryPool.GetDefault(ordinal);
    }

    /// <summary>
    /// Gets the current CUDA memory pool for a device.
    /// 获取指定设备当前使用的 CUDA memory pool。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <returns>The current memory pool wrapper. 当前 memory pool 封装。</returns>
    public static CudaMemoryPool GetCurrentMemoryPool(int ordinal)
    {
        return CudaMemoryPool.GetCurrent(ordinal);
    }

    /// <summary>
    /// Sets the current CUDA memory pool for the memory pool's device.
    /// 为该 memory pool 所属设备设置当前 CUDA memory pool。
    /// </summary>
    /// <param name="memoryPool">The memory pool to make current. 要设置为当前池的 memory pool。</param>
    public static void SetCurrentMemoryPool(CudaMemoryPool memoryPool)
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.SetCurrentMemoryPoolHandle(memoryPool.DeviceOrdinal, memoryPool.Handle);
    }

}
