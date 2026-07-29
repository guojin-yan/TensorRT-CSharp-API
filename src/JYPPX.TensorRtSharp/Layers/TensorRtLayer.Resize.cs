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
    /// Sets the Resize Output Dimensions value.
    /// 设置 Resize Output Dimensions 值。
    /// </summary>
    public void SetResizeOutputDimensions(TensorRtDims dimensions)
    {
        if (dimensions == null)
        {
            throw new ArgumentNullException(nameof(dimensions));
        }

        NativeBridgeApi.SetResizeOutputDimensions(Line, _handle, dimensions);
    }

    /// <summary>
    /// Gets the Resize Output Dimensions value.
    /// 获取 Resize Output Dimensions 值。
    /// </summary>
    public TensorRtDims GetResizeOutputDimensions()
    {
        return NativeBridgeApi.GetResizeOutputDimensions(Line, _handle);
    }

    /// <summary>
    /// Sets the Resize Mode value.
    /// 设置 Resize Mode 值。
    /// </summary>
    public void SetResizeMode(TensorRtResizeMode resizeMode)
    {
        NativeBridgeApi.SetResizeMode(Line, _handle, resizeMode);
    }

    /// <summary>
    /// Gets the Resize Mode value.
    /// 获取 Resize Mode 值。
    /// </summary>
    public TensorRtResizeMode GetResizeMode()
    {
        return NativeBridgeApi.GetResizeMode(Line, _handle);
    }

    /// <summary>
    /// Gets the TensorRT 8 resize align-corners flag.
    /// 获取 TensorRT 8 resize 层的 align-corners 标志。
    /// </summary>
    /// <returns><c>true</c> when corner alignment is enabled. 启用 corner alignment 时返回 <c>true</c>。</returns>
    public bool GetResizeAlignCorners()
    {
        return NativeBridgeApi.GetResizeAlignCorners(Line, _handle);
    }

    /// <summary>
    /// Sets the TensorRT 8 resize align-corners flag.
    /// 设置 TensorRT 8 resize 层的 align-corners 标志。
    /// </summary>
    /// <param name="alignCorners">Whether resize should align the corner pixels. resize 是否应对齐角点像素。</param>
    public void SetResizeAlignCorners(bool alignCorners)
    {
        NativeBridgeApi.SetResizeAlignCorners(Line, _handle, alignCorners);
    }

    /// <summary>
    /// Gets the Resize Coordinate Transformation value.
    /// 获取 Resize Coordinate Transformation 值。
    /// </summary>
    public TensorRtResizeCoordinateTransformation GetResizeCoordinateTransformation()
    {
        return NativeBridgeApi.GetResizeCoordinateTransformation(Line, _handle);
    }

    /// <summary>
    /// Sets the Resize Coordinate Transformation value.
    /// 设置 Resize Coordinate Transformation 值。
    /// </summary>
    public void SetResizeCoordinateTransformation(TensorRtResizeCoordinateTransformation transformation)
    {
        NativeBridgeApi.SetResizeCoordinateTransformation(Line, _handle, transformation);
    }

    /// <summary>
    /// Gets the Resize Selector For Single Pixel value.
    /// 获取 Resize Selector For Single Pixel 值。
    /// </summary>
    public TensorRtResizeSelector GetResizeSelectorForSinglePixel()
    {
        return NativeBridgeApi.GetResizeSelectorForSinglePixel(Line, _handle);
    }

    /// <summary>
    /// Sets the Resize Selector For Single Pixel value.
    /// 设置 Resize Selector For Single Pixel 值。
    /// </summary>
    public void SetResizeSelectorForSinglePixel(TensorRtResizeSelector selector)
    {
        NativeBridgeApi.SetResizeSelectorForSinglePixel(Line, _handle, selector);
    }

    /// <summary>
    /// Gets the Resize Nearest Rounding value.
    /// 获取 Resize Nearest Rounding 值。
    /// </summary>
    public TensorRtResizeRoundMode GetResizeNearestRounding()
    {
        return NativeBridgeApi.GetResizeNearestRounding(Line, _handle);
    }

    /// <summary>
    /// Sets the Resize Nearest Rounding value.
    /// 设置 Resize Nearest Rounding 值。
    /// </summary>
    public void SetResizeNearestRounding(TensorRtResizeRoundMode rounding)
    {
        NativeBridgeApi.SetResizeNearestRounding(Line, _handle, rounding);
    }

    /// <summary>
    /// Gets the Resize Cubic Coefficient value.
    /// 获取 Resize Cubic Coefficient 值。
    /// </summary>
    public double GetResizeCubicCoefficient()
    {
        return NativeBridgeApi.GetResizeCubicCoefficient(Line, _handle);
    }

    /// <summary>
    /// Sets the Resize Cubic Coefficient value.
    /// 设置 Resize Cubic Coefficient 值。
    /// </summary>
    public void SetResizeCubicCoefficient(double value)
    {
        NativeBridgeApi.SetResizeCubicCoefficient(Line, _handle, value);
    }

    /// <summary>
    /// Gets the Resize Exclude Outside value.
    /// 获取 Resize Exclude Outside 值。
    /// </summary>
    public bool GetResizeExcludeOutside()
    {
        return NativeBridgeApi.GetResizeExcludeOutside(Line, _handle);
    }

    /// <summary>
    /// Sets the Resize Exclude Outside value.
    /// 设置 Resize Exclude Outside 值。
    /// </summary>
    public void SetResizeExcludeOutside(bool value)
    {
        NativeBridgeApi.SetResizeExcludeOutside(Line, _handle, value);
    }

    /// <summary>
    /// Sets the Resize Scales value.
    /// 设置 Resize Scales 值。
    /// </summary>
    public void SetResizeScales(float[] scales)
    {
        NativeBridgeApi.SetResizeScales(Line, _handle, scales);
    }

    /// <summary>
    /// Gets the Resize Scales value.
    /// 获取 Resize Scales 值。
    /// </summary>
    public float[] GetResizeScales()
    {
        return NativeBridgeApi.GetResizeScales(Line, _handle);
    }

}
