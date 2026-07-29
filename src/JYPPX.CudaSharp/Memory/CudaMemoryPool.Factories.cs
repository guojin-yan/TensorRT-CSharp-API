using System;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public readonly partial struct CudaMemoryPool
{
    /// <summary>
    /// Creates a standalone CUDA memory pool owned by the returned managed object.
    /// 创建一个由返回托管对象拥有并负责释放的独立 CUDA 内存池。
    /// </summary>
    /// <param name="deviceOrdinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <returns>An owned memory-pool wrapper that must be disposed. 需要释放的 owned 内存池封装。</returns>
    public static CudaOwnedMemoryPool Create(int deviceOrdinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        ulong handle = NativeCudaApi.CreateMemoryPoolHandle(deviceOrdinal);
        return new CudaOwnedMemoryPool(new CudaMemoryPool(handle, deviceOrdinal));
    }

    /// <summary>
    /// Gets the default CUDA memory pool for a device.
    /// 获取指定设备的默认 CUDA 内存池。
    /// </summary>
    /// <param name="deviceOrdinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <returns>The default memory pool wrapper. 默认内存池封装。</returns>
    public static CudaMemoryPool GetDefault(int deviceOrdinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        ulong handle = NativeCudaApi.GetDefaultMemoryPoolHandle(deviceOrdinal);
        return new CudaMemoryPool(handle, deviceOrdinal);
    }

    /// <summary>
    /// Gets the current CUDA memory pool for a device.
    /// 获取指定设备当前使用的 CUDA 内存池。
    /// </summary>
    /// <param name="deviceOrdinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <returns>The current memory pool wrapper. 当前内存池封装。</returns>
    public static CudaMemoryPool GetCurrent(int deviceOrdinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        ulong handle = NativeCudaApi.GetCurrentMemoryPoolHandle(deviceOrdinal);
        return new CudaMemoryPool(handle, deviceOrdinal);
    }

    /// <summary>
    /// Makes this memory pool the current pool for its owning device.
    /// 将当前内存池设置为其所属 CUDA 设备的当前内存池。
    /// </summary>
    public void MakeCurrent()
    {
        NativeCudaApi.SetCurrentMemoryPoolHandle(DeviceOrdinal, _handle);
    }

}
