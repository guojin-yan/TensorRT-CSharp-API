using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Captures copied scalar metadata from a CUDA graph memory-allocation node.
/// 捕获 CUDA graph memory-allocation 节点的复制型标量元数据。
/// </summary>
public sealed class CudaGraphMemoryAllocationNodeSnapshot
{
    internal CudaGraphMemoryAllocationNodeSnapshot(NativeCudaGraphMemAllocNodeParamsSnapshot native)
    {
        ByteCount = native.ByteCount;
        AccessDescriptorCount = native.AccessDescriptorCount;
        AllocationType = native.AllocationType;
        HandleTypes = native.HandleTypes;
        LocationType = (CudaMemoryLocationType)native.LocationType;
        LocationId = native.LocationId;
        HasAccessDescriptors = native.HasAccessDescriptors != 0;
        HasDevicePointer = native.HasDevicePointer != 0;
        HasSecurityAttributes = native.HasSecurityAttributes != 0;
    }

    /// <summary>Gets the requested allocation size in bytes. 获取请求的分配字节数。</summary>
    public ulong ByteCount { get; }
    /// <summary>Gets the copied access-descriptor count. 获取复制的 access descriptor 数量。</summary>
    public ulong AccessDescriptorCount { get; }
    /// <summary>Gets the native allocation-type value. 获取 native allocation type 值。</summary>
    public int AllocationType { get; }
    /// <summary>Gets the native shareable-handle type mask. 获取 native 可共享 handle 类型掩码。</summary>
    public ulong HandleTypes { get; }
    /// <summary>Gets the allocation location type. 获取分配位置类型。</summary>
    public CudaMemoryLocationType LocationType { get; }
    /// <summary>Gets the allocation location identifier. 获取分配位置标识符。</summary>
    public int LocationId { get; }
    /// <summary>Gets whether access descriptors were present. 获取是否存在 access descriptor。</summary>
    public bool HasAccessDescriptors { get; }
    /// <summary>Gets whether CUDA produced a device pointer without exposing its value. 获取 CUDA 是否生成了 device pointer，但不暴露其值。</summary>
    public bool HasDevicePointer { get; }
    /// <summary>Gets whether security attributes were present. 获取是否存在安全属性。</summary>
    public bool HasSecurityAttributes { get; }

    /// <inheritdoc />
    public override string ToString() =>
        $"Bytes={ByteCount}, AccessDescriptors={AccessDescriptorCount}, AllocationType={AllocationType}, HandleTypes={HandleTypes}, Location={LocationType}:{LocationId}, HasDevicePointer={HasDevicePointer}";
}
