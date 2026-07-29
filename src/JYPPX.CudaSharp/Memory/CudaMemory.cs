using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Managed wrapper around a device memory allocation.
/// CUDA 设备内存分配的托管封装。
/// </summary>
public partial class CudaMemory : IDisposable
{
    private readonly SafeCudaMemoryHandle _handle;

    /// <summary>
    /// Allocates CUDA device memory.
    /// 分配 CUDA 设备内存。
    /// </summary>
    /// <param name="sizeInBytes">The allocation size in bytes. 分配大小，单位为字节。</param>
    public CudaMemory(int sizeInBytes)
        : this(AllocateDeviceMemory(sizeInBytes))
    {
    }

    internal CudaMemory(SafeCudaMemoryHandle handle)
        : this(handle, isIpcImported: false)
    {
    }

    private CudaMemory(SafeCudaMemoryHandle handle, bool isIpcImported)
    {
        _handle = handle ?? throw new ArgumentNullException(nameof(handle));
        IsIpcImported = isIpcImported;
        SizeInBytes = checked((int)NativeCudaApi.GetMemorySize(_handle));
    }

    /// <summary>
    /// Gets the allocation size in bytes.
    /// 获取分配大小，单位为字节。
    /// </summary>
    public int SizeInBytes { get; }

    /// <summary>Gets whether this wrapper owns a process-local CUDA IPC mapping. 获取此 wrapper 是否拥有进程内 CUDA IPC mapping。</summary>
    public bool IsIpcImported { get; }

    internal SafeCudaMemoryHandle Handle => _handle;

    /// <summary>
    /// Releases the device allocation.
    /// 释放设备内存分配。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ValidateCount(int count, string parameterName)
    {
        if (count < 0 || count > SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(parameterName);
        }
    }

    /// <summary>Validates a non-empty byte range within this allocation. 验证当前分配内的非空字节范围。</summary>
    /// <param name="offset">The byte offset to validate. 要验证的字节偏移。</param>
    /// <param name="count">The byte count to validate. 要验证的字节数。</param>
    /// <param name="offsetParameterName">The offset parameter name used by exceptions. 异常使用的 offset 参数名。</param>
    /// <param name="countParameterName">The count parameter name used by exceptions. 异常使用的 count 参数名。</param>
    protected void ValidateRange(int offset, int count, string offsetParameterName, string countParameterName)
    {
        if (offset < 0 || offset > SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(offsetParameterName);
        }

        if (count <= 0 || count > SizeInBytes - offset)
        {
            throw new ArgumentOutOfRangeException(countParameterName);
        }
    }

    private static void ValidateScalarRangeAttribute(CudaMemoryRangeAttribute attribute, string parameterName)
    {
        if (!Enum.IsDefined(typeof(CudaMemoryRangeAttribute), attribute))
        {
            throw new ArgumentOutOfRangeException(parameterName, attribute, "Unsupported CUDA memory range attribute.");
        }

        if (attribute == CudaMemoryRangeAttribute.AccessedBy)
        {
            throw new ArgumentException("cudaMemRangeAttributeAccessedBy returns a device-id array and is not exposed by the scalar range attribute helpers.", parameterName);
        }
    }

    /// <summary>Validates a CUDA memory-advice value. 验证 CUDA 内存建议值。</summary>
    /// <param name="advice">The memory advice to validate. 要验证的内存建议。</param>
    /// <param name="parameterName">The parameter name used by exceptions. 异常使用的参数名。</param>
    protected static void ValidateMemoryAdvice(CudaMemoryAdvice advice, string parameterName)
    {
        if (!Enum.IsDefined(typeof(CudaMemoryAdvice), advice))
        {
            throw new ArgumentOutOfRangeException(parameterName, advice, "Unsupported CUDA memory advice.");
        }
    }

    private static SafeCudaMemoryHandle AllocateDeviceMemory(int sizeInBytes)
    {
        if (sizeInBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sizeInBytes));
        }

        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.AllocateMemory(sizeInBytes);
    }
}
