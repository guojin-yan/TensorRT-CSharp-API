using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public sealed partial class CudaPitchedMemory
{
    /// <summary>
    /// Fills the full logical 2D allocation with a byte value.
    /// 使用指定字节值填充完整的逻辑二维分配区域。
    /// </summary>
    /// <param name="value">The byte value used by CUDA memset. CUDA memset 使用的字节值。</param>
    public void Fill2D(byte value)
    {
        Fill2D(value, WidthInBytes, Height);
    }

    /// <summary>
    /// Fills a logical 2D subregion with a byte value.
    /// 使用指定字节值填充逻辑二维子区域。
    /// </summary>
    /// <param name="value">The byte value used by CUDA memset. CUDA memset 使用的字节值。</param>
    /// <param name="widthInBytes">The row width to fill in bytes. 要填充的每行字节数。</param>
    /// <param name="height">The number of rows to fill. 要填充的行数。</param>
    public void Fill2D(byte value, int widthInBytes, int height)
    {
        Validate2DExtent(widthInBytes, height);
        NativeCudaApi.FillPitched2D(_handle, value, widthInBytes, height);
    }

    /// <summary>
    /// Asynchronously fills the full logical 2D allocation with a byte value.
    /// 使用指定字节值异步填充完整的逻辑二维分配区域。
    /// </summary>
    /// <param name="value">The byte value used by CUDA memset. CUDA memset 使用的字节值。</param>
    /// <param name="stream">The CUDA stream that orders the fill. 用于排序填充操作的 CUDA stream。</param>
    public void Fill2DAsync(byte value, CudaStream stream)
    {
        Fill2DAsync(value, WidthInBytes, Height, stream);
    }

    /// <summary>
    /// Asynchronously fills a logical 2D subregion with a byte value.
    /// 使用指定字节值异步填充逻辑二维子区域。
    /// </summary>
    /// <param name="value">The byte value used by CUDA memset. CUDA memset 使用的字节值。</param>
    /// <param name="widthInBytes">The row width to fill in bytes. 要填充的每行字节数。</param>
    /// <param name="height">The number of rows to fill. 要填充的行数。</param>
    /// <param name="stream">The CUDA stream that orders the fill. 用于排序填充操作的 CUDA stream。</param>
    public void Fill2DAsync(byte value, int widthInBytes, int height, CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        Validate2DExtent(widthInBytes, height);
        NativeCudaApi.FillPitched2DAsync(_handle, value, widthInBytes, height, stream.Handle);
    }

    /// <summary>
    /// Fills a logical 3D region with a byte value.
    /// 使用指定字节值填充逻辑 3D 区域。
    /// </summary>
    /// <param name="value">The byte value used by CUDA memset. CUDA memset 使用的字节值。</param>
    /// <param name="height">The height of one logical slice. 单个逻辑 slice 的高度。</param>
    /// <param name="depth">The number of logical slices. 逻辑 slice 数量。</param>
    public void Fill3D(byte value, int height, int depth)
    {
        Fill3D(value, WidthInBytes, height, depth);
    }

    /// <summary>
    /// Fills a logical 3D subregion with a byte value.
    /// 使用指定字节值填充逻辑 3D 子区域。
    /// </summary>
    /// <param name="value">The byte value used by CUDA memset. CUDA memset 使用的字节值。</param>
    /// <param name="widthInBytes">The row width to fill in bytes. 要填充的每行字节数。</param>
    /// <param name="height">The height of one logical slice. 单个逻辑 slice 的高度。</param>
    /// <param name="depth">The number of logical slices. 逻辑 slice 数量。</param>
    public void Fill3D(byte value, int widthInBytes, int height, int depth)
    {
        Validate3DExtent(widthInBytes, height, depth);
        NativeCudaApi.FillPitched3D(_handle, value, widthInBytes, height, depth);
    }

    /// <summary>
    /// Asynchronously fills a logical 3D region with a byte value.
    /// 异步使用指定字节值填充逻辑 3D 区域。
    /// </summary>
    /// <param name="value">The byte value used by CUDA memset. CUDA memset 使用的字节值。</param>
    /// <param name="height">The height of one logical slice. 单个逻辑 slice 的高度。</param>
    /// <param name="depth">The number of logical slices. 逻辑 slice 数量。</param>
    /// <param name="stream">The CUDA stream that orders the fill. 用于排序填充操作的 CUDA stream。</param>
    public void Fill3DAsync(byte value, int height, int depth, CudaStream stream)
    {
        Fill3DAsync(value, WidthInBytes, height, depth, stream);
    }

    /// <summary>
    /// Asynchronously fills a logical 3D subregion with a byte value.
    /// 异步使用指定字节值填充逻辑 3D 子区域。
    /// </summary>
    /// <param name="value">The byte value used by CUDA memset. CUDA memset 使用的字节值。</param>
    /// <param name="widthInBytes">The row width to fill in bytes. 要填充的每行字节数。</param>
    /// <param name="height">The height of one logical slice. 单个逻辑 slice 的高度。</param>
    /// <param name="depth">The number of logical slices. 逻辑 slice 数量。</param>
    /// <param name="stream">The CUDA stream that orders the fill. 用于排序填充操作的 CUDA stream。</param>
    public void Fill3DAsync(byte value, int widthInBytes, int height, int depth, CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        Validate3DExtent(widthInBytes, height, depth);
        NativeCudaApi.FillPitched3DAsync(_handle, value, widthInBytes, height, depth, stream.Handle);
    }

}
