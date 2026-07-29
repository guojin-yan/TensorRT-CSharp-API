using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a managed TensorRT Tensor Rt Layer wrapper.
/// 表示托管 TensorRT Tensor Rt Layer 包装器。
/// </summary>
public sealed partial class TensorRtLayer
{
    /// <summary>
    /// Sets the Slice Start value.
    /// 设置 Slice Start 值。
    /// </summary>
    public void SetSliceStart(TensorRtDims start)
    {
        if (start == null)
        {
            throw new ArgumentNullException(nameof(start));
        }

        NativeBridgeApi.SetSliceStart(Line, _handle, start);
    }

    /// <summary>
    /// Gets the Slice Start value.
    /// 获取 Slice Start 值。
    /// </summary>
    public TensorRtDims GetSliceStart()
    {
        return NativeBridgeApi.GetSliceStart(Line, _handle);
    }

    /// <summary>
    /// Sets the Slice Size value.
    /// 设置 Slice Size 值。
    /// </summary>
    public void SetSliceSize(TensorRtDims size)
    {
        if (size == null)
        {
            throw new ArgumentNullException(nameof(size));
        }

        NativeBridgeApi.SetSliceSize(Line, _handle, size);
    }

    /// <summary>
    /// Gets the Slice Size value.
    /// 获取 Slice Size 值。
    /// </summary>
    public TensorRtDims GetSliceSize()
    {
        return NativeBridgeApi.GetSliceSize(Line, _handle);
    }

    /// <summary>
    /// Sets the Slice Stride value.
    /// 设置 Slice Stride 值。
    /// </summary>
    public void SetSliceStride(TensorRtDims stride)
    {
        if (stride == null)
        {
            throw new ArgumentNullException(nameof(stride));
        }

        NativeBridgeApi.SetSliceStride(Line, _handle, stride);
    }

    /// <summary>
    /// Gets the Slice Stride value.
    /// 获取 Slice Stride 值。
    /// </summary>
    public TensorRtDims GetSliceStride()
    {
        return NativeBridgeApi.GetSliceStride(Line, _handle);
    }

    /// <summary>
    /// Sets the axes vector used by a TensorRT 10 or TensorRT 11 slice layer.
    /// 设置 TensorRT 10 或 TensorRT 11 slice 层使用的 axes 向量。
    /// </summary>
    /// <param name="axes">The axes dimensions to apply. 要应用的 axes 维度。</param>
    public void SetSliceAxes(TensorRtDims axes)
    {
        if (axes == null)
        {
            throw new ArgumentNullException(nameof(axes));
        }

        NativeBridgeApi.SetSliceAxes(Line, _handle, axes);
    }

    /// <summary>
    /// Gets the axes vector used by a TensorRT 10 or TensorRT 11 slice layer.
    /// 获取 TensorRT 10 或 TensorRT 11 slice 层使用的 axes 向量。
    /// </summary>
    /// <returns>The current slice axes dimensions. 当前 slice axes 维度。</returns>
    public TensorRtDims GetSliceAxes()
    {
        return NativeBridgeApi.GetSliceAxes(Line, _handle);
    }

    /// <summary>
    /// Sets the Slice Mode value.
    /// 设置 Slice Mode 值。
    /// </summary>
    public void SetSliceMode(TensorRtSliceMode mode)
    {
        NativeBridgeApi.SetSliceMode(Line, _handle, mode);
    }

    /// <summary>
    /// Gets the Slice Mode value.
    /// 获取 Slice Mode 值。
    /// </summary>
    public TensorRtSliceMode GetSliceMode()
    {
        return NativeBridgeApi.GetSliceMode(Line, _handle);
    }

}
