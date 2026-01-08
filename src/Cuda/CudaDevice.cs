using JYPPX.TensorRtSharp.Cuda.Enum;
using JYPPX.TensorRtSharp.Cuda.Struct;
using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using System;
using System.Text;

namespace JYPPX.TensorRtSharp.Cuda
{
    /// <summary>
    /// 提供 CUDA 设备管理的功能，包括设备枚举、属性查询、缓存配置以及点对点 (P2P) 访问。
    /// Provides CUDA device management functionalities, including device enumeration, property queries, cache configuration, and peer-to-peer (P2P) access.
    /// </summary>
    public static class CudaDevice
    {
        /// <summary>
        /// 销毁当前进程中的所有分配并重置所有设备上的所有状态。
        /// Destroys all allocations and resets all state on all devices in the current process.
        /// </summary>
        public static void Reset()
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaDeviceReset());
        }

        /// <summary>
        /// 阻塞 CPU 线程，直到设备完成所有请求的任务。
        /// Blocks the CPU thread until the device has completed all requested tasks.
        /// </summary>
        public static void Synchronize()
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaDeviceSynchronize());
        }

        /// <summary>
        /// 设置设备执行的限制。
        /// Sets the limit on the device's execution.
        /// </summary>
        /// <param name="limit">要设置的限制。/ The limit to set.</param>
        /// <param name="value">限制的新值。/ The new value of the limit.</param>
        public static void SetLimit(CudaLimit limit, ulong value)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaDeviceSetLimit(limit, value));
        }

        /// <summary>
        /// 获取设备执行限制的当前值。
        /// Gets the current value of a device execution limit.
        /// </summary>
        /// <param name="limit">要查询的限制。/ The limit to query.</param>
        /// <returns>限制的当前值。/ The current value of the limit.</returns>
        public static ulong GetLimit(CudaLimit limit)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaDeviceGetLimit(out ulong value, limit));
            return value;
        }

        /// <summary>
        /// 获取给定设备上使用特定纹理格式时一维纹理的最大宽度。
        /// Gets the maximum 1D texture width for a given device using a specific texture format.
        /// </summary>
        /// <param name="fmtDesc">纹理的通道格式描述符。/ The channel format descriptor of the texture.</param>
        /// <param name="device">设备索引。/ The device index.</param>
        /// <returns>最大宽度（以元素为单位）。/ The maximum width in elements.</returns>
        public static ulong GetTexture1DLinearMaxWidth(CudaChannelFormatDesc fmtDesc, int device)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaDeviceGetTexture1DLinearMaxWidth(out ulong maxWidthInElements, ref fmtDesc, device));
            return maxWidthInElements;
        }

        /// <summary>
        /// 获取当前设备的首选缓存配置。
        /// Gets the preferred cache configuration for the current device.
        /// </summary>
        /// <returns>缓存配置。/ The cache configuration.</returns>
        public static CudaFuncCache GetCacheConfig()
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaDeviceGetCacheConfig(out CudaFuncCache pCacheConfig));
            return pCacheConfig;
        }

        /// <summary>
        /// 设置当前设备的首选缓存配置。
        /// Sets the preferred cache configuration for the current device.
        /// </summary>
        /// <param name="cacheConfig">缓存配置。/ The cache configuration.</param>
        public static void SetCacheConfig(CudaFuncCache cacheConfig)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaDeviceSetCacheConfig(cacheConfig));
        }

        /// <summary>
        /// 获取当前设备的共享内存配置。
        /// Gets the shared memory configuration for the current device.
        /// </summary>
        /// <returns>共享内存配置。/ The shared memory configuration.</returns>
        public static CudaSharedMemConfig GetSharedMemConfig()
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaDeviceGetSharedMemConfig(out CudaSharedMemConfig pConfig));
            return pConfig;
        }

        /// <summary>
        /// 设置当前设备的共享内存配置。
        /// Sets the shared memory configuration for the current device.
        /// </summary>
        /// <param name="config">共享内存配置。/ The shared memory configuration.</param>
        public static void SetSharedMemConfig(CudaSharedMemConfig config)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaDeviceSetSharedMemConfig(config));
        }

        /// <summary>
        /// 根据 PCI 总线 ID 返回设备的编号。
        /// Returns the device number based on the PCI bus ID.
        /// </summary>
        /// <param name="pciBusId">PCI 总线 ID 字符串。/ The PCI bus ID string.</param>
        /// <returns>设备编号。/ The device number.</returns>
        public static int GetByPCIBusId(string pciBusId)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaDeviceGetByPCIBusId(out int device, pciBusId));
            return device;
        }

        /// <summary>
        /// 返回设备的 PCI 总线 ID 字符串。
        /// Returns the PCI bus ID string for a device.
        /// </summary>
        /// <param name="device">设备索引。/ The device index.</param>
        /// <returns>PCI 总线 ID 字符串。/ The PCI bus ID string.</returns>
        public static string GetPCIBusId(int device)
        {
            // CUDA_MAX_PCIBUS_ID_LEN is typically 256 or similar. 
            // Using a sufficiently large buffer.
            StringBuilder sb = new StringBuilder(256);
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaDeviceGetPCIBusId(sb, sb.Capacity, device));
            return sb.ToString();
        }

        /// <summary>
        /// 获取系统中启用的 CUDA 兼容设备的数量。
        /// Gets the number of CUDA-capable devices enabled in the system.
        /// </summary>
        /// <returns>设备数量。/ The number of devices.</returns>
        public static int GetDeviceCount()
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaGetDeviceCount(out int count));
            return count;
        }

        /// <summary>
        /// 获取指定设备的属性。
        /// Gets the properties for the specified device.
        /// </summary>
        /// <param name="device">设备索引。/ The device index.</param>
        /// <returns>包含设备属性的结构体。/ The structure containing device properties.</returns>
        public static CudaDeviceProp GetDeviceProperties(int device)
        {
            CudaDeviceProp prop = new CudaDeviceProp();
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaGetDeviceProperties(ref prop, device));
            return prop;
        }

        /// <summary>
        /// 返回有关设备的请求信息。
        /// Returns information about the device.
        /// </summary>
        /// <param name="attr">要请求的属性。/ The attribute to request.</param>
        /// <param name="device">设备索引。/ The device index.</param>
        /// <returns>属性的值。/ The value of the attribute.</returns>
        public static int GetAttribute(CudaDeviceAttr attr, int device)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaDeviceGetAttribute(out int value, attr, device));
            return value;
        }

        /// <summary>
        /// 获取指定设备的默认内存池。
        /// Gets the default memory pool of the specified device.
        /// </summary>
        /// <param name="device">设备索引。/ The device index.</param>
        /// <returns>内存池句柄。/ The memory pool handle.</returns>
        public static CudaMemPoolStr GetDefaultMemPool(int device)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaDeviceGetDefaultMemPool(out CudaMemPoolStr memPool, device));
            return memPool;
        }

        /// <summary>
        /// 为指定设备设置内存池。
        /// Sets the memory pool for a specific device.
        /// </summary>
        /// <param name="device">设备索引。/ The device index.</param>
        /// <param name="memPool">内存池句柄。/ The memory pool handle.</param>
        public static void SetMemPool(int device, CudaMemPoolStr memPool)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaDeviceSetMemPool(device, memPool));
        }

        /// <summary>
        /// 获取指定设备的内存池。
        /// Gets the memory pool of the specified device.
        /// </summary>
        /// <param name="device">设备索引。/ The device index.</param>
        /// <returns>内存池句柄。/ The memory pool handle.</returns>
        public static CudaMemPoolStr GetMemPool(int device)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaDeviceGetMemPool(out CudaMemPoolStr memPool, device));
            return memPool;
        }

        /// <summary>
        /// 查询点对点 (P2P) 访问属性。
        /// Queries peer-to-peer (P2P) access attributes.
        /// </summary>
        /// <param name="attr">要查询的 P2P 属性。/ The P2P attribute to query.</param>
        /// <param name="srcDevice">源设备索引。/ The source device index.</param>
        /// <param name="dstDevice">目标设备索引。/ The destination device index.</param>
        /// <returns>属性的值。/ The value of the attribute.</returns>
        public static int GetP2PAttribute(CudaDeviceP2PAttr attr, int srcDevice, int dstDevice)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaDeviceGetP2PAttribute(out int value, attr, srcDevice, dstDevice));
            return value;
        }

        /// <summary>
        /// 根据提供的属性选择能够匹配的设备。
        /// Selects the device that best matches the provided properties.
        /// </summary>
        /// <param name="prop">设备属性参考。/ The device properties reference.</param>
        /// <returns>匹配的设备索引。/ The device index that matches.</returns>
        public static int ChooseDevice(ref CudaDeviceProp prop)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaChooseDevice(out int device, ref prop));
            return device;
        }

        /// <summary>
        /// 设置执行设备。
        /// Sets device as the current device for the calling host thread.
        /// </summary>
        /// <param name="device">设备索引。/ The device index.</param>
        public static void SetDevice(int device)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaSetDevice(device));
        }

        /// <summary>
        /// 获取调用主机线程的当前设备。
        /// Gets the current device for the calling host thread.
        /// </summary>
        /// <returns>当前设备索引。/ The current device index.</returns>
        public static int GetDevice()
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaGetDevice(out int device));
            return device;
        }

        /// <summary>
        /// 设置设备列表，仅允许 CUDA 使用这些设备。
        /// Sets the list of devices for CUDA to use, restricting CUDA to only these devices.
        /// </summary>
        /// <param name="device_arr">包含设备索引的数组。/ An array containing device indices.</param>
        public static void SetValidDevices(int[] device_arr)
        {
            if (device_arr == null) throw new ArgumentNullException(nameof(device_arr));
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaSetValidDevices(device_arr, device_arr.Length));
        }

        /// <summary>
        /// 设置获取设备调用的标志。
        /// Sets flags for the device get calls.
        /// </summary>
        /// <param name="flags">设备标志。/ The device flags.</param>
        public static void SetDeviceFlags(uint flags)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaSetDeviceFlags(flags));
        }

        /// <summary>
        /// 获取设备的当前标志。
        /// Gets the flags for the current device.
        /// </summary>
        /// <returns>当前设备标志。/ The current device flags.</returns>
        public static uint GetDeviceFlags()
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaGetDeviceFlags(out uint flags));
            return flags;
        }

        /// <summary>
        /// 刷新 GPU Direct RDMA 写操作。
        /// Flushes GPU Direct RDMA writes.
        /// </summary>
        /// <param name="target">刷新目标。/ The flush target.</param>
        /// <param name="scope">刷新范围。/ The flush scope.</param>
        public static void FlushGPUDirectRDMAWrites(CudaFlushGPUDirectRDMAWritesTarget target, CudaFlushGPUDirectRDMAWritesScope scope)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaDeviceFlushGPUDirectRDMAWrites(target, scope));
        }

        /// <summary>
        /// 查询设备是否可以访问对等设备的内存。
        /// Queries if a device can access the memory of a peer device.
        /// </summary>
        /// <param name="device">执行查询的设备索引。/ The device index performing the query.</param>
        /// <param name="peerDevice">要检查的对等设备索引。/ The peer device index to check.</param>
        /// <returns>如果可以访问则为 true，否则为 false。/ True if access is possible, false otherwise.</returns>
        public static bool CanAccessPeer(int device, int peerDevice)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaDeviceCanAccessPeer(out int canAccessPeer, device, peerDevice));
            return canAccessPeer != 0;
        }

        /// <summary>
        /// 启用对等地址访问。
        /// Enables peer address access.
        /// </summary>
        /// <param name="peerDevice">要对等访问的设备索引。/ The device index to be accessed.</param>
        /// <param name="flags">保留供将来使用，必须为 0。/ Reserved for future use, must be 0.</param>
        public static void EnablePeerAccess(int peerDevice, uint flags)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaDeviceEnablePeerAccess(peerDevice, flags));
        }

        /// <summary>
        /// 禁用对等地址访问。
        /// Disables peer address access.
        /// </summary>
        /// <param name="peerDevice">要对等访问的设备索引。/ The device index to be accessed.</param>
        public static void DisablePeerAccess(int peerDevice)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaDeviceDisablePeerAccess(peerDevice));
        }

        /// <summary>
        /// 重置当前上下文的持久化 L2 缓存。
        /// Resets the persisting L2 cache of the current context.
        /// </summary>
        public static void ResetPersistingL2Cache()
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaCtxResetPersistingL2Cache());
        }
    }
}
