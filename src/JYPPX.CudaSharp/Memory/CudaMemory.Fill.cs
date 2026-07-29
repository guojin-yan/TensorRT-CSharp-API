using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public partial class CudaMemory
{
    /// <summary>
    /// Fills part of the allocation with a byte value.
    /// 使用一个字节值填充部分分配区域。
    /// </summary>
    /// <param name="value">The fill byte value. 填充值。</param>
    /// <param name="count">The number of bytes to fill. 要填充的字节数。</param>
    public void Fill(byte value, int count)
    {
        if (count < 0 || count > SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.FillMemory(_handle, value, count);
    }

    /// <summary>
    /// Fills the entire allocation with a byte value.
    /// 使用一个字节值填充整个设备内存分配。
    /// </summary>
    /// <param name="value">The fill byte value. 填充值。</param>
    public void Fill(byte value)
    {
        Fill(value, SizeInBytes);
    }

    /// <summary>
    /// Asynchronously fills part of the allocation with a byte value.
    /// 使用一个字节值异步填充分配的一部分区域。
    /// </summary>
    /// <param name="value">The fill byte value. 填充值。</param>
    /// <param name="count">The number of bytes to fill. 要填充的字节数。</param>
    /// <param name="stream">The CUDA stream that orders the fill. 用于排序填充操作的 CUDA stream。</param>
    public void FillAsync(byte value, int count, CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        ValidateCount(count, nameof(count));
        NativeCudaApi.FillMemoryAsync(_handle, value, count, stream.Handle);
    }

    /// <summary>
    /// Asynchronously fills the entire allocation with a byte value.
    /// 异步使用一个字节值填充整个设备内存分配。
    /// </summary>
    /// <param name="value">The fill byte value. 填充值。</param>
    /// <param name="stream">The CUDA stream that orders the fill. 用于排序填充操作的 CUDA stream。</param>
    public void FillAsync(byte value, CudaStream stream)
    {
        FillAsync(value, SizeInBytes, stream);
    }

}
