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
    /// Gets the Top K Operation value.
    /// 获取 Top K Operation 值。
    /// </summary>
    public TensorRtTopKOperation GetTopKOperation()
    {
        return NativeBridgeApi.GetTopKOperation(Line, _handle);
    }

    /// <summary>
    /// Sets the Top K Operation value.
    /// 设置 Top K Operation 值。
    /// </summary>
    public void SetTopKOperation(TensorRtTopKOperation operation)
    {
        NativeBridgeApi.SetTopKOperation(Line, _handle, operation);
    }

    /// <summary>
    /// Gets the Top K Value value.
    /// 获取 Top K Value 值。
    /// </summary>
    public int GetTopKValue()
    {
        return NativeBridgeApi.GetTopKValue(Line, _handle);
    }

    /// <summary>
    /// Sets the Top K Value value.
    /// 设置 Top K Value 值。
    /// </summary>
    public void SetTopKValue(int k)
    {
        if (k <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(k));
        }

        NativeBridgeApi.SetTopKValue(Line, _handle, k);
    }

    /// <summary>
    /// Gets the Top K Axes value.
    /// 获取 Top K Axes 值。
    /// </summary>
    public uint GetTopKAxes()
    {
        return NativeBridgeApi.GetTopKAxes(Line, _handle);
    }

    /// <summary>
    /// Sets the Top K Axes value.
    /// 设置 Top K Axes 值。
    /// </summary>
    public void SetTopKAxes(uint axes)
    {
        if (axes == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(axes), "TopK axes bitmask must not be zero.");
        }

        NativeBridgeApi.SetTopKAxes(Line, _handle, axes);
    }

    /// <summary>
    /// Gets the index output data type configured on a TensorRT 11 TopK layer.
    /// 获取 TensorRT 11 TopK 层索引输出的数据类型。
    /// </summary>
    /// <returns>The configured TopK indices data type. 已配置的 TopK indices 数据类型。</returns>
    public TensorRtDataType GetTopKIndicesType()
    {
        return NativeBridgeApi.GetTopKIndicesType(Line, _handle);
    }

    /// <summary>
    /// Sets the index output data type on a TensorRT 11 TopK layer.
    /// 设置 TensorRT 11 TopK 层索引输出的数据类型。
    /// </summary>
    /// <param name="dataType">The requested indices data type. 请求设置的 indices 数据类型。</param>
    /// <returns><c>true</c> when TensorRT accepts the value; otherwise <c>false</c>. TensorRT 接受该值时返回 <c>true</c>，否则返回 <c>false</c>。</returns>
    public bool SetTopKIndicesType(TensorRtDataType dataType)
    {
        return NativeBridgeApi.SetTopKIndicesType(Line, _handle, dataType);
    }

}
