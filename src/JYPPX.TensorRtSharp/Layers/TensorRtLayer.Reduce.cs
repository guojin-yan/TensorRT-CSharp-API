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
    /// Gets the Reduce Operation value.
    /// 获取 Reduce Operation 值。
    /// </summary>
    public TensorRtReduceOperation GetReduceOperation()
    {
        return NativeBridgeApi.GetReduceOperation(Line, _handle);
    }

    /// <summary>
    /// Sets the Reduce Operation value.
    /// 设置 Reduce Operation 值。
    /// </summary>
    public void SetReduceOperation(TensorRtReduceOperation operation)
    {
        NativeBridgeApi.SetReduceOperation(Line, _handle, operation);
    }

    /// <summary>
    /// Gets the Reduce Axes value.
    /// 获取 Reduce Axes 值。
    /// </summary>
    public uint GetReduceAxes()
    {
        return NativeBridgeApi.GetReduceAxes(Line, _handle);
    }

    /// <summary>
    /// Sets the Reduce Axes value.
    /// 设置 Reduce Axes 值。
    /// </summary>
    public void SetReduceAxes(uint axes)
    {
        if (axes == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(axes), "Reduce axes bitmask must not be zero.");
        }

        NativeBridgeApi.SetReduceAxes(Line, _handle, axes);
    }

    /// <summary>
    /// Gets the Reduce Keep Dimensions value.
    /// 获取 Reduce Keep Dimensions 值。
    /// </summary>
    public bool GetReduceKeepDimensions()
    {
        return NativeBridgeApi.GetReduceKeepDimensions(Line, _handle);
    }

    /// <summary>
    /// Sets the Reduce Keep Dimensions value.
    /// 设置 Reduce Keep Dimensions 值。
    /// </summary>
    public void SetReduceKeepDimensions(bool keepDimensions)
    {
        NativeBridgeApi.SetReduceKeepDimensions(Line, _handle, keepDimensions);
    }

}
