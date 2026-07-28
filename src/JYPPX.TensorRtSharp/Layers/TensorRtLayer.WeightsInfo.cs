using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtLayer
{
    /// <summary>
    /// Gets metadata for weights stored by a TensorRT constant layer.
    /// 获取 TensorRT constant 层保存的权重元数据。
    /// </summary>
    /// <remarks>
    /// The returned object reports data type, element count, and whether TensorRT has a native value pointer; it does not expose the native pointer.
    /// 返回对象只报告数据类型、元素数量以及 TensorRT 是否持有原生 value 指针；不会暴露原生指针。
    /// </remarks>
    public TensorRtWeightsInfo GetConstantWeightsInfo()
    {
        return new TensorRtWeightsInfo(NativeBridgeApi.GetConstantLayerWeightsInfo(Line, _handle));
    }

    /// <summary>
    /// Gets metadata for the kernel weights of a TensorRT convolution layer.
    /// 获取 TensorRT convolution 层 kernel 权重的元数据。
    /// </summary>
    public TensorRtWeightsInfo GetConvolutionKernelWeightsInfo()
    {
        return new TensorRtWeightsInfo(NativeBridgeApi.GetConvolutionLayerKernelWeightsInfo(Line, _handle));
    }

    /// <summary>
    /// Gets metadata for the bias weights of a TensorRT convolution layer.
    /// 获取 TensorRT convolution 层 bias 权重的元数据。
    /// </summary>
    public TensorRtWeightsInfo GetConvolutionBiasWeightsInfo()
    {
        return new TensorRtWeightsInfo(NativeBridgeApi.GetConvolutionLayerBiasWeightsInfo(Line, _handle));
    }

    /// <summary>
    /// Gets metadata for the kernel weights of a TensorRT deconvolution layer.
    /// 获取 TensorRT deconvolution 层 kernel 权重的元数据。
    /// </summary>
    public TensorRtWeightsInfo GetDeconvolutionKernelWeightsInfo()
    {
        return new TensorRtWeightsInfo(NativeBridgeApi.GetDeconvolutionLayerKernelWeightsInfo(Line, _handle));
    }

    /// <summary>
    /// Gets metadata for the bias weights of a TensorRT deconvolution layer.
    /// 获取 TensorRT deconvolution 层 bias 权重的元数据。
    /// </summary>
    public TensorRtWeightsInfo GetDeconvolutionBiasWeightsInfo()
    {
        return new TensorRtWeightsInfo(NativeBridgeApi.GetDeconvolutionLayerBiasWeightsInfo(Line, _handle));
    }

    /// <summary>
    /// Gets metadata for the shift weights of a TensorRT scale layer.
    /// 获取 TensorRT scale 层 shift 权重的元数据。
    /// </summary>
    public TensorRtWeightsInfo GetScaleShiftWeightsInfo()
    {
        return new TensorRtWeightsInfo(NativeBridgeApi.GetScaleLayerShiftWeightsInfo(Line, _handle));
    }

    /// <summary>
    /// Gets metadata for the scale weights of a TensorRT scale layer.
    /// 获取 TensorRT scale 层 scale 权重的元数据。
    /// </summary>
    public TensorRtWeightsInfo GetScaleScaleWeightsInfo()
    {
        return new TensorRtWeightsInfo(NativeBridgeApi.GetScaleLayerScaleWeightsInfo(Line, _handle));
    }

    /// <summary>
    /// Gets metadata for the power weights of a TensorRT scale layer.
    /// 获取 TensorRT scale 层 power 权重的元数据。
    /// </summary>
    public TensorRtWeightsInfo GetScalePowerWeightsInfo()
    {
        return new TensorRtWeightsInfo(NativeBridgeApi.GetScaleLayerPowerWeightsInfo(Line, _handle));
    }
}
