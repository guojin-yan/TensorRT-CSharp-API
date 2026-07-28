using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a managed TensorRT Tensor Rt Layer wrapper.
/// 表示托管 TensorRT Tensor Rt Layer 包装器。
/// </summary>
public sealed partial class TensorRtLayer
{
    /// <summary>
    /// Gets the Deconvolution Output Maps value.
    /// 获取 Deconvolution Output Maps 值。
    /// </summary>
    public int GetDeconvolutionOutputMaps()
    {
        return NativeBridgeApi.GetDeconvolutionOutputMaps(Line, _handle);
    }

    /// <summary>
    /// Sets the Deconvolution Output Maps value.
    /// 设置 Deconvolution Output Maps 值。
    /// </summary>
    public void SetDeconvolutionOutputMaps(int outputMaps)
    {
        NativeBridgeApi.SetDeconvolutionOutputMaps(Line, _handle, outputMaps);
    }

    /// <summary>
    /// Gets the Deconvolution Groups value.
    /// 获取 Deconvolution Groups 值。
    /// </summary>
    public int GetDeconvolutionGroups()
    {
        return NativeBridgeApi.GetDeconvolutionGroups(Line, _handle);
    }

    /// <summary>
    /// Sets the Deconvolution Groups value.
    /// 设置 Deconvolution Groups 值。
    /// </summary>
    public void SetDeconvolutionGroups(int groups)
    {
        NativeBridgeApi.SetDeconvolutionGroups(Line, _handle, groups);
    }

    /// <summary>
    /// Gets the Deconvolution Kernel Size value.
    /// 获取 Deconvolution Kernel Size 值。
    /// </summary>
    public TensorRtDims GetDeconvolutionKernelSize()
    {
        return NativeBridgeApi.GetDeconvolutionKernelSize(Line, _handle);
    }

    /// <summary>
    /// Sets the Deconvolution Kernel Size value.
    /// 设置 Deconvolution Kernel Size 值。
    /// </summary>
    public void SetDeconvolutionKernelSize(TensorRtDims kernelSize)
    {
        ValidateDims(kernelSize, nameof(kernelSize));
        NativeBridgeApi.SetDeconvolutionKernelSize(Line, _handle, kernelSize);
    }

    /// <summary>
    /// Gets the Deconvolution Stride value.
    /// 获取 Deconvolution Stride 值。
    /// </summary>
    public TensorRtDims GetDeconvolutionStride()
    {
        return NativeBridgeApi.GetDeconvolutionStride(Line, _handle);
    }

    /// <summary>
    /// Sets the Deconvolution Stride value.
    /// 设置 Deconvolution Stride 值。
    /// </summary>
    public void SetDeconvolutionStride(TensorRtDims stride)
    {
        ValidateDims(stride, nameof(stride));
        NativeBridgeApi.SetDeconvolutionStride(Line, _handle, stride);
    }

    /// <summary>
    /// Gets the explicit N-D padding of a deconvolution layer.
    /// 获取 deconvolution 层的显式 N-D padding。
    /// </summary>
    /// <returns>The current deconvolution padding dimensions. 当前 deconvolution padding 维度。</returns>
    public TensorRtDims GetDeconvolutionPadding()
    {
        return NativeBridgeApi.GetDeconvolutionPadding(Line, _handle);
    }

    /// <summary>
    /// Sets the explicit N-D padding of a deconvolution layer.
    /// 设置 deconvolution 层的显式 N-D padding。
    /// </summary>
    /// <param name="padding">The padding dimensions to apply. 要应用的 padding 维度。</param>
    public void SetDeconvolutionPadding(TensorRtDims padding)
    {
        ValidateDims(padding, nameof(padding));
        NativeBridgeApi.SetDeconvolutionPadding(Line, _handle, padding);
    }

    /// <summary>
    /// Gets the Deconvolution Pre Padding value.
    /// 获取 Deconvolution Pre Padding 值。
    /// </summary>
    public TensorRtDims GetDeconvolutionPrePadding()
    {
        return NativeBridgeApi.GetDeconvolutionPrePadding(Line, _handle);
    }

    /// <summary>
    /// Sets the Deconvolution Pre Padding value.
    /// 设置 Deconvolution Pre Padding 值。
    /// </summary>
    public void SetDeconvolutionPrePadding(TensorRtDims padding)
    {
        ValidateDims(padding, nameof(padding));
        NativeBridgeApi.SetDeconvolutionPrePadding(Line, _handle, padding);
    }

    /// <summary>
    /// Gets the Deconvolution Post Padding value.
    /// 获取 Deconvolution Post Padding 值。
    /// </summary>
    public TensorRtDims GetDeconvolutionPostPadding()
    {
        return NativeBridgeApi.GetDeconvolutionPostPadding(Line, _handle);
    }

    /// <summary>
    /// Sets the Deconvolution Post Padding value.
    /// 设置 Deconvolution Post Padding 值。
    /// </summary>
    public void SetDeconvolutionPostPadding(TensorRtDims padding)
    {
        ValidateDims(padding, nameof(padding));
        NativeBridgeApi.SetDeconvolutionPostPadding(Line, _handle, padding);
    }

    /// <summary>
    /// Gets the Deconvolution Dilation value.
    /// 获取 Deconvolution Dilation 值。
    /// </summary>
    public TensorRtDims GetDeconvolutionDilation()
    {
        return NativeBridgeApi.GetDeconvolutionDilation(Line, _handle);
    }

    /// <summary>
    /// Sets the Deconvolution Dilation value.
    /// 设置 Deconvolution Dilation 值。
    /// </summary>
    public void SetDeconvolutionDilation(TensorRtDims dilation)
    {
        ValidateDims(dilation, nameof(dilation));
        NativeBridgeApi.SetDeconvolutionDilation(Line, _handle, dilation);
    }

    /// <summary>
    /// Gets the Deconvolution Padding Mode value.
    /// 获取 Deconvolution Padding Mode 值。
    /// </summary>
    public TensorRtPaddingMode GetDeconvolutionPaddingMode()
    {
        return NativeBridgeApi.GetDeconvolutionPaddingMode(Line, _handle);
    }

    /// <summary>
    /// Sets the Deconvolution Padding Mode value.
    /// 设置 Deconvolution Padding Mode 值。
    /// </summary>
    public void SetDeconvolutionPaddingMode(TensorRtPaddingMode paddingMode)
    {
        NativeBridgeApi.SetDeconvolutionPaddingMode(Line, _handle, paddingMode);
    }
}
