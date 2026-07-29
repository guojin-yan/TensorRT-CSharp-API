using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public sealed partial class CudaArray
{
    /// <summary>
    /// Queries the device-memory requirements for this array on the selected device. 查询该 array 在指定设备上的显存需求。
    /// </summary>
    public CudaArrayMemoryRequirements GetMemoryRequirements(int device)
    {
        return CudaArrayMemoryRequirements.FromNative(NativeCudaApi.GetArrayMemoryRequirements(_handle, device));
    }

    /// <summary>
    /// Attempts to query device-memory requirements without surfacing a CUDA exception. 尝试查询显存需求而不直接抛出 CUDA 异常。
    /// </summary>
    public bool TryGetMemoryRequirements(int device, out CudaArrayMemoryRequirements requirements, out string diagnostic)
    {
        try
        {
            requirements = GetMemoryRequirements(device);
            diagnostic = string.Empty;
            return true;
        }
        catch (CudaException exception)
        {
            requirements = default;
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Queries sparse-array metadata for this array. 查询该 array 的 sparse 元数据。
    /// </summary>
    public CudaArraySparseProperties GetSparseProperties()
    {
        return CudaArraySparseProperties.FromNative(NativeCudaApi.GetArraySparseProperties(_handle));
    }

    /// <summary>
    /// Attempts to query sparse-array metadata without surfacing a CUDA exception. 尝试查询 sparse 元数据而不直接抛出 CUDA 异常。
    /// </summary>
    public bool TryGetSparseProperties(out CudaArraySparseProperties properties, out string diagnostic)
    {
        try
        {
            properties = GetSparseProperties();
            diagnostic = string.Empty;
            return true;
        }
        catch (CudaException exception)
        {
            properties = default;
            diagnostic = exception.Message;
            return false;
        }
    }

}
