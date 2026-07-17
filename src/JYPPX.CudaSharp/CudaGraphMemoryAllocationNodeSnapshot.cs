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

    public ulong ByteCount { get; }
    public ulong AccessDescriptorCount { get; }
    public int AllocationType { get; }
    public ulong HandleTypes { get; }
    public CudaMemoryLocationType LocationType { get; }
    public int LocationId { get; }
    public bool HasAccessDescriptors { get; }
    public bool HasDevicePointer { get; }
    public bool HasSecurityAttributes { get; }

    /// <inheritdoc />
    public override string ToString() =>
        $"Bytes={ByteCount}, AccessDescriptors={AccessDescriptorCount}, AllocationType={AllocationType}, HandleTypes={HandleTypes}, Location={LocationType}:{LocationId}, HasDevicePointer={HasDevicePointer}";
}
