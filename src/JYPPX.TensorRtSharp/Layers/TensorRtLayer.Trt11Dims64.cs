using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtLayer
{
    /// <summary>
    /// Gets one input tensor slot shape using TensorRT 11 64-bit dimension extents.
    /// 使用 TensorRT 11 的 64 位维度 extent 获取输入 tensor 槽位形状。
    /// </summary>
    public TensorRtDims64 GetInputTensorShape64(int index)
    {
        ValidateInputIndex(index);
        return NativeBridgeApi.GetLayerInputTensorShape64(Line, _handle, index);
    }

    /// <summary>
    /// Gets one output tensor slot shape using TensorRT 11 64-bit dimension extents.
    /// 使用 TensorRT 11 的 64 位维度 extent 获取输出 tensor 槽位形状。
    /// </summary>
    public TensorRtDims64 GetOutputTensorShape64(int index)
    {
        ValidateOutputIndex(index);
        return NativeBridgeApi.GetLayerOutputTensorShape64(Line, _handle, index);
    }

    /// <summary>
    /// Gets one input tensor slot dimension extent as a 64-bit value.
    /// 以 64 位整数获取输入 tensor 槽位的单个维度 extent。
    /// </summary>
    public long GetInputTensorDimensionExtent64(int index, int dimensionIndex)
    {
        ValidateInputIndex(index);
        return NativeBridgeApi.GetLayerInputTensorDimensionExtent64(Line, _handle, index, dimensionIndex);
    }

    /// <summary>
    /// Gets one output tensor slot dimension extent as a 64-bit value.
    /// 以 64 位整数获取输出 tensor 槽位的单个维度 extent。
    /// </summary>
    public long GetOutputTensorDimensionExtent64(int index, int dimensionIndex)
    {
        ValidateOutputIndex(index);
        return NativeBridgeApi.GetLayerOutputTensorDimensionExtent64(Line, _handle, index, dimensionIndex);
    }

    /// <summary>
    /// Gets shuffle reshape dimensions with TensorRT 11 64-bit extents.
    /// 获取 shuffle reshape 维度，并保留 TensorRT 11 的 64 位 extent。
    /// </summary>
    public TensorRtDims64 GetShuffleReshapeDimensions64() => NativeBridgeApi.GetShuffleReshapeDimensions64(Line, _handle);

    /// <summary>
    /// Gets slice start dimensions with TensorRT 11 64-bit extents.
    /// 获取 slice start 维度，并保留 TensorRT 11 的 64 位 extent。
    /// </summary>
    public TensorRtDims64 GetSliceStart64() => NativeBridgeApi.GetSliceStart64(Line, _handle);

    /// <summary>
    /// Gets slice size dimensions with TensorRT 11 64-bit extents.
    /// 获取 slice size 维度，并保留 TensorRT 11 的 64 位 extent。
    /// </summary>
    public TensorRtDims64 GetSliceSize64() => NativeBridgeApi.GetSliceSize64(Line, _handle);

    /// <summary>
    /// Gets slice stride dimensions with TensorRT 11 64-bit extents.
    /// 获取 slice stride 维度，并保留 TensorRT 11 的 64 位 extent。
    /// </summary>
    public TensorRtDims64 GetSliceStride64() => NativeBridgeApi.GetSliceStride64(Line, _handle);

    /// <summary>
    /// Gets dynamic-quantize block shape with TensorRT 11 64-bit extents.
    /// 获取 dynamic-quantize block shape，并保留 TensorRT 11 的 64 位 extent。
    /// </summary>
    public TensorRtDims64 GetDynamicQuantizeBlockShape64() => NativeBridgeApi.GetDynamicQuantizeBlockShape64(Line, _handle);

    /// <summary>
    /// Gets pooling window dimensions with TensorRT 11 64-bit extents.
    /// 获取 pooling window 维度，并保留 TensorRT 11 的 64 位 extent。
    /// </summary>
    public TensorRtDims64 GetPoolingWindowSize64() => NativeBridgeApi.GetPoolingWindowSize64(Line, _handle);

    /// <summary>
    /// Gets pooling stride dimensions with TensorRT 11 64-bit extents.
    /// 获取 pooling stride 维度，并保留 TensorRT 11 的 64 位 extent。
    /// </summary>
    public TensorRtDims64 GetPoolingStride64() => NativeBridgeApi.GetPoolingStride64(Line, _handle);

    /// <summary>
    /// Gets pooling padding dimensions with TensorRT 11 64-bit extents.
    /// 获取 pooling padding 维度，并保留 TensorRT 11 的 64 位 extent。
    /// </summary>
    public TensorRtDims64 GetPoolingPadding64() => NativeBridgeApi.GetPoolingPadding64(Line, _handle);

    /// <summary>
    /// Gets pooling pre-padding dimensions with TensorRT 11 64-bit extents.
    /// 获取 pooling pre-padding 维度，并保留 TensorRT 11 的 64 位 extent。
    /// </summary>
    public TensorRtDims64 GetPoolingPrePadding64() => NativeBridgeApi.GetPoolingPrePadding64(Line, _handle);

    /// <summary>
    /// Gets pooling post-padding dimensions with TensorRT 11 64-bit extents.
    /// 获取 pooling post-padding 维度，并保留 TensorRT 11 的 64 位 extent。
    /// </summary>
    public TensorRtDims64 GetPoolingPostPadding64() => NativeBridgeApi.GetPoolingPostPadding64(Line, _handle);

    /// <summary>
    /// Gets resize output dimensions with TensorRT 11 64-bit extents.
    /// 获取 resize output 维度，并保留 TensorRT 11 的 64 位 extent。
    /// </summary>
    public TensorRtDims64 GetResizeOutputDimensions64() => NativeBridgeApi.GetResizeOutputDimensions64(Line, _handle);

    /// <summary>
    /// Gets fill dimensions with TensorRT 11 64-bit extents.
    /// 获取 fill 维度，并保留 TensorRT 11 的 64 位 extent。
    /// </summary>
    public TensorRtDims64 GetFillDimensions64() => NativeBridgeApi.GetFillDimensions64(Line, _handle);

    /// <summary>
    /// Gets convolution stride dimensions with TensorRT 11 64-bit extents.
    /// 获取 convolution stride 维度，并保留 TensorRT 11 的 64 位 extent。
    /// </summary>
    public TensorRtDims64 GetConvolutionStride64() => NativeBridgeApi.GetConvolutionStride64(Line, _handle);

    /// <summary>
    /// Gets convolution pre-padding dimensions with TensorRT 11 64-bit extents.
    /// 获取 convolution pre-padding 维度，并保留 TensorRT 11 的 64 位 extent。
    /// </summary>
    public TensorRtDims64 GetConvolutionPrePadding64() => NativeBridgeApi.GetConvolutionPrePadding64(Line, _handle);

    /// <summary>
    /// Gets convolution post-padding dimensions with TensorRT 11 64-bit extents.
    /// 获取 convolution post-padding 维度，并保留 TensorRT 11 的 64 位 extent。
    /// </summary>
    public TensorRtDims64 GetConvolutionPostPadding64() => NativeBridgeApi.GetConvolutionPostPadding64(Line, _handle);

    /// <summary>
    /// Gets convolution dilation dimensions with TensorRT 11 64-bit extents.
    /// 获取 convolution dilation 维度，并保留 TensorRT 11 的 64 位 extent。
    /// </summary>
    public TensorRtDims64 GetConvolutionDilation64() => NativeBridgeApi.GetConvolutionDilation64(Line, _handle);

    /// <summary>
    /// Gets deconvolution kernel dimensions with TensorRT 11 64-bit extents.
    /// 获取 deconvolution kernel 维度，并保留 TensorRT 11 的 64 位 extent。
    /// </summary>
    public TensorRtDims64 GetDeconvolutionKernelSize64() => NativeBridgeApi.GetDeconvolutionKernelSize64(Line, _handle);

    /// <summary>
    /// Gets deconvolution stride dimensions with TensorRT 11 64-bit extents.
    /// 获取 deconvolution stride 维度，并保留 TensorRT 11 的 64 位 extent。
    /// </summary>
    public TensorRtDims64 GetDeconvolutionStride64() => NativeBridgeApi.GetDeconvolutionStride64(Line, _handle);

    /// <summary>
    /// Gets deconvolution pre-padding dimensions with TensorRT 11 64-bit extents.
    /// 获取 deconvolution pre-padding 维度，并保留 TensorRT 11 的 64 位 extent。
    /// </summary>
    public TensorRtDims64 GetDeconvolutionPrePadding64() => NativeBridgeApi.GetDeconvolutionPrePadding64(Line, _handle);

    /// <summary>
    /// Gets deconvolution post-padding dimensions with TensorRT 11 64-bit extents.
    /// 获取 deconvolution post-padding 维度，并保留 TensorRT 11 的 64 位 extent。
    /// </summary>
    public TensorRtDims64 GetDeconvolutionPostPadding64() => NativeBridgeApi.GetDeconvolutionPostPadding64(Line, _handle);

    /// <summary>
    /// Gets deconvolution dilation dimensions with TensorRT 11 64-bit extents.
    /// 获取 deconvolution dilation 维度，并保留 TensorRT 11 的 64 位 extent。
    /// </summary>
    public TensorRtDims64 GetDeconvolutionDilation64() => NativeBridgeApi.GetDeconvolutionDilation64(Line, _handle);

    /// <summary>
    /// Gets padding layer pre-padding dimensions with TensorRT 11 64-bit extents.
    /// 获取 padding layer pre-padding 维度，并保留 TensorRT 11 的 64 位 extent。
    /// </summary>
    public TensorRtDims64 GetPaddingPrePadding64() => NativeBridgeApi.GetPaddingPrePadding64(Line, _handle);

    /// <summary>
    /// Gets padding layer post-padding dimensions with TensorRT 11 64-bit extents.
    /// 获取 padding layer post-padding 维度，并保留 TensorRT 11 的 64 位 extent。
    /// </summary>
    public TensorRtDims64 GetPaddingPostPadding64() => NativeBridgeApi.GetPaddingPostPadding64(Line, _handle);
}
