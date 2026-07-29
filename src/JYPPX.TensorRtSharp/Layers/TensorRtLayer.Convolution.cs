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
    /// Gets the Convolution Output Maps value.
    /// 获取 Convolution Output Maps 值。
    /// </summary>
    public int GetConvolutionOutputMaps()
    {
        return NativeBridgeApi.GetConvolutionOutputMaps(Line, _handle);
    }

    /// <summary>
    /// Sets the Convolution Output Maps value.
    /// 设置 Convolution Output Maps 值。
    /// </summary>
    public void SetConvolutionOutputMaps(int outputMaps)
    {
        NativeBridgeApi.SetConvolutionOutputMaps(Line, _handle, outputMaps);
    }

    /// <summary>
    /// Gets the Convolution Groups value.
    /// 获取 Convolution Groups 值。
    /// </summary>
    public int GetConvolutionGroups()
    {
        return NativeBridgeApi.GetConvolutionGroups(Line, _handle);
    }

    /// <summary>
    /// Sets the Convolution Groups value.
    /// 设置 Convolution Groups 值。
    /// </summary>
    public void SetConvolutionGroups(int groups)
    {
        NativeBridgeApi.SetConvolutionGroups(Line, _handle, groups);
    }

    /// <summary>
    /// Gets the Convolution Stride value.
    /// 获取 Convolution Stride 值。
    /// </summary>
    public TensorRtDims GetConvolutionStride()
    {
        return NativeBridgeApi.GetConvolutionStride(Line, _handle);
    }

    /// <summary>
    /// Sets the Convolution Stride value.
    /// 设置 Convolution Stride 值。
    /// </summary>
    public void SetConvolutionStride(TensorRtDims stride)
    {
        ValidateDims(stride, nameof(stride));
        NativeBridgeApi.SetConvolutionStride(Line, _handle, stride);
    }

    /// <summary>
    /// Gets the explicit N-D padding of a convolution layer.
    /// 获取 convolution 层的显式 N-D padding。
    /// </summary>
    /// <returns>The current convolution padding dimensions. 当前 convolution padding 维度。</returns>
    public TensorRtDims GetConvolutionPadding()
    {
        return NativeBridgeApi.GetConvolutionPadding(Line, _handle);
    }

    /// <summary>
    /// Sets the explicit N-D padding of a convolution layer.
    /// 设置 convolution 层的显式 N-D padding。
    /// </summary>
    /// <param name="padding">The padding dimensions to apply. 要应用的 padding 维度。</param>
    public void SetConvolutionPadding(TensorRtDims padding)
    {
        ValidateDims(padding, nameof(padding));
        NativeBridgeApi.SetConvolutionPadding(Line, _handle, padding);
    }

    /// <summary>
    /// Gets the Convolution Pre Padding value.
    /// 获取 Convolution Pre Padding 值。
    /// </summary>
    public TensorRtDims GetConvolutionPrePadding()
    {
        return NativeBridgeApi.GetConvolutionPrePadding(Line, _handle);
    }

    /// <summary>
    /// Sets the Convolution Pre Padding value.
    /// 设置 Convolution Pre Padding 值。
    /// </summary>
    public void SetConvolutionPrePadding(TensorRtDims padding)
    {
        ValidateDims(padding, nameof(padding));
        NativeBridgeApi.SetConvolutionPrePadding(Line, _handle, padding);
    }

    /// <summary>
    /// Gets the Convolution Post Padding value.
    /// 获取 Convolution Post Padding 值。
    /// </summary>
    public TensorRtDims GetConvolutionPostPadding()
    {
        return NativeBridgeApi.GetConvolutionPostPadding(Line, _handle);
    }

    /// <summary>
    /// Sets the Convolution Post Padding value.
    /// 设置 Convolution Post Padding 值。
    /// </summary>
    public void SetConvolutionPostPadding(TensorRtDims padding)
    {
        ValidateDims(padding, nameof(padding));
        NativeBridgeApi.SetConvolutionPostPadding(Line, _handle, padding);
    }

    /// <summary>
    /// Gets the Convolution Dilation value.
    /// 获取 Convolution Dilation 值。
    /// </summary>
    public TensorRtDims GetConvolutionDilation()
    {
        return NativeBridgeApi.GetConvolutionDilation(Line, _handle);
    }

    /// <summary>
    /// Sets the Convolution Dilation value.
    /// 设置 Convolution Dilation 值。
    /// </summary>
    public void SetConvolutionDilation(TensorRtDims dilation)
    {
        ValidateDims(dilation, nameof(dilation));
        NativeBridgeApi.SetConvolutionDilation(Line, _handle, dilation);
    }

    /// <summary>
    /// Gets the Convolution Padding Mode value.
    /// 获取 Convolution Padding Mode 值。
    /// </summary>
    public TensorRtPaddingMode GetConvolutionPaddingMode()
    {
        return NativeBridgeApi.GetConvolutionPaddingMode(Line, _handle);
    }

    /// <summary>
    /// Sets the Convolution Padding Mode value.
    /// 设置 Convolution Padding Mode 值。
    /// </summary>
    public void SetConvolutionPaddingMode(TensorRtPaddingMode paddingMode)
    {
        NativeBridgeApi.SetConvolutionPaddingMode(Line, _handle, paddingMode);
    }

}
