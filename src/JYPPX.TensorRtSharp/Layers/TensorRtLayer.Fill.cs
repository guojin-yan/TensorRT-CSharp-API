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
    /// Sets the Fill Dimensions value.
    /// 设置 Fill Dimensions 值。
    /// </summary>
    public void SetFillDimensions(TensorRtDims dimensions)
    {
        if (dimensions == null)
        {
            throw new ArgumentNullException(nameof(dimensions));
        }

        NativeBridgeApi.SetFillDimensions(Line, _handle, dimensions);
    }

    /// <summary>
    /// Gets the Fill Dimensions value.
    /// 获取 Fill Dimensions 值。
    /// </summary>
    public TensorRtDims GetFillDimensions()
    {
        return NativeBridgeApi.GetFillDimensions(Line, _handle);
    }

    /// <summary>
    /// Sets the Fill Operation value.
    /// 设置 Fill Operation 值。
    /// </summary>
    public void SetFillOperation(TensorRtFillOperation operation)
    {
        NativeBridgeApi.SetFillOperation(Line, _handle, operation);
    }

    /// <summary>
    /// Gets the Fill Operation value.
    /// 获取 Fill Operation 值。
    /// </summary>
    public TensorRtFillOperation GetFillOperation()
    {
        return NativeBridgeApi.GetFillOperation(Line, _handle);
    }

    /// <summary>
    /// Sets the Fill Alpha value.
    /// 设置 Fill Alpha 值。
    /// </summary>
    public void SetFillAlpha(double alpha)
    {
        NativeBridgeApi.SetFillAlpha(Line, _handle, alpha);
    }

    /// <summary>
    /// Gets the Fill Alpha value.
    /// 获取 Fill Alpha 值。
    /// </summary>
    public double GetFillAlpha()
    {
        return NativeBridgeApi.GetFillAlpha(Line, _handle);
    }

    /// <summary>
    /// Sets the Fill Beta value.
    /// 设置 Fill Beta 值。
    /// </summary>
    public void SetFillBeta(double beta)
    {
        NativeBridgeApi.SetFillBeta(Line, _handle, beta);
    }

    /// <summary>
    /// Gets the Fill Beta value.
    /// 获取 Fill Beta 值。
    /// </summary>
    public double GetFillBeta()
    {
        return NativeBridgeApi.GetFillBeta(Line, _handle);
    }

}
