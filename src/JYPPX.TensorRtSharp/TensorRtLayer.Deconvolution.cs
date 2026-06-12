using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtLayer
{
    public int GetDeconvolutionOutputMaps()
    {
        return NativeBridgeApi.GetDeconvolutionOutputMaps(Line, _handle);
    }

    public void SetDeconvolutionOutputMaps(int outputMaps)
    {
        NativeBridgeApi.SetDeconvolutionOutputMaps(Line, _handle, outputMaps);
    }

    public int GetDeconvolutionGroups()
    {
        return NativeBridgeApi.GetDeconvolutionGroups(Line, _handle);
    }

    public void SetDeconvolutionGroups(int groups)
    {
        NativeBridgeApi.SetDeconvolutionGroups(Line, _handle, groups);
    }

    public TensorRtDims GetDeconvolutionKernelSize()
    {
        return NativeBridgeApi.GetDeconvolutionKernelSize(Line, _handle);
    }

    public void SetDeconvolutionKernelSize(TensorRtDims kernelSize)
    {
        ValidateDims(kernelSize, nameof(kernelSize));
        NativeBridgeApi.SetDeconvolutionKernelSize(Line, _handle, kernelSize);
    }

    public TensorRtDims GetDeconvolutionStride()
    {
        return NativeBridgeApi.GetDeconvolutionStride(Line, _handle);
    }

    public void SetDeconvolutionStride(TensorRtDims stride)
    {
        ValidateDims(stride, nameof(stride));
        NativeBridgeApi.SetDeconvolutionStride(Line, _handle, stride);
    }

    /// <summary>
    /// Gets the explicit N-D padding of a deconvolution layer.
    /// 获取 deconvolution 层的显式 N-D padding。
    /// </summary>
    /// <returns>The current deconvolution padding dimensions.</returns>
    public TensorRtDims GetDeconvolutionPadding()
    {
        return NativeBridgeApi.GetDeconvolutionPadding(Line, _handle);
    }

    /// <summary>
    /// Sets the explicit N-D padding of a deconvolution layer.
    /// 设置 deconvolution 层的显式 N-D padding。
    /// </summary>
    /// <param name="padding">The padding dimensions to apply.</param>
    public void SetDeconvolutionPadding(TensorRtDims padding)
    {
        ValidateDims(padding, nameof(padding));
        NativeBridgeApi.SetDeconvolutionPadding(Line, _handle, padding);
    }

    public TensorRtDims GetDeconvolutionPrePadding()
    {
        return NativeBridgeApi.GetDeconvolutionPrePadding(Line, _handle);
    }

    public void SetDeconvolutionPrePadding(TensorRtDims padding)
    {
        ValidateDims(padding, nameof(padding));
        NativeBridgeApi.SetDeconvolutionPrePadding(Line, _handle, padding);
    }

    public TensorRtDims GetDeconvolutionPostPadding()
    {
        return NativeBridgeApi.GetDeconvolutionPostPadding(Line, _handle);
    }

    public void SetDeconvolutionPostPadding(TensorRtDims padding)
    {
        ValidateDims(padding, nameof(padding));
        NativeBridgeApi.SetDeconvolutionPostPadding(Line, _handle, padding);
    }

    public TensorRtDims GetDeconvolutionDilation()
    {
        return NativeBridgeApi.GetDeconvolutionDilation(Line, _handle);
    }

    public void SetDeconvolutionDilation(TensorRtDims dilation)
    {
        ValidateDims(dilation, nameof(dilation));
        NativeBridgeApi.SetDeconvolutionDilation(Line, _handle, dilation);
    }

    public TensorRtPaddingMode GetDeconvolutionPaddingMode()
    {
        return NativeBridgeApi.GetDeconvolutionPaddingMode(Line, _handle);
    }

    public void SetDeconvolutionPaddingMode(TensorRtPaddingMode paddingMode)
    {
        NativeBridgeApi.SetDeconvolutionPaddingMode(Line, _handle, paddingMode);
    }
}
