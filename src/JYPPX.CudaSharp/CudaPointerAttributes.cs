namespace JYPPX.CudaSharp;

/// <summary>
/// Describes how CUDA classifies a memory pointer returned by the runtime.
/// 描述 CUDA Runtime 对内存指针的分类方式。
/// </summary>
public enum CudaMemoryPointerType
{
    /// <summary>
    /// The pointer is not registered with CUDA.
    /// 该指针未注册到 CUDA。
    /// </summary>
    Unregistered = 0,

    /// <summary>
    /// The pointer refers to host memory.
    /// 该指针指向主机内存。
    /// </summary>
    Host = 1,

    /// <summary>
    /// The pointer refers to device memory.
    /// 该指针指向设备内存。
    /// </summary>
    Device = 2,

    /// <summary>
    /// The pointer refers to managed memory.
    /// 该指针指向统一托管内存。
    /// </summary>
    Managed = 3
}

/// <summary>
/// Provides diagnostic CUDA pointer metadata without exposing a raw pointer as the main user API.
/// 提供 CUDA 指针诊断元数据，同时避免把裸指针作为主要用户 API 暴露。
/// </summary>
public readonly struct CudaPointerAttributes
{
    internal CudaPointerAttributes(CudaMemoryPointerType memoryType, int deviceOrdinal, ulong devicePointerAddress, ulong hostPointerAddress)
    {
        MemoryType = memoryType;
        DeviceOrdinal = deviceOrdinal;
        DevicePointerAddress = devicePointerAddress;
        HostPointerAddress = hostPointerAddress;
    }

    /// <summary>
    /// Gets the CUDA memory classification for the pointer.
    /// 获取 CUDA 对该指针的内存类型分类。
    /// </summary>
    public CudaMemoryPointerType MemoryType { get; }

    /// <summary>
    /// Gets the CUDA device associated with the allocation or registration.
    /// 获取与该分配或注册关联的 CUDA 设备序号。
    /// </summary>
    public int DeviceOrdinal { get; }

    /// <summary>
    /// Gets the numeric device pointer address for diagnostics.
    /// 获取用于诊断的设备指针地址数值。
    /// </summary>
    public ulong DevicePointerAddress { get; }

    /// <summary>
    /// Gets the numeric host pointer address for diagnostics.
    /// 获取用于诊断的主机指针地址数值。
    /// </summary>
    public ulong HostPointerAddress { get; }

    /// <summary>
    /// Returns a readable diagnostic summary.
    /// 返回可读的诊断摘要。
    /// </summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{MemoryType} device={DeviceOrdinal} devicePtr=0x{DevicePointerAddress:X} hostPtr=0x{HostPointerAddress:X}";
    }
}
