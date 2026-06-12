using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtNetworkDefinition
{
    public TensorRtLayer AddQuantize(TensorRtTensor input, TensorRtTensor scale)
    {
        ValidateInputTensor(input, nameof(input));
        ValidateInputTensor(scale, nameof(scale));
        return new TensorRtLayer(Line, NativeBridgeApi.AddQuantizeLayer(Line, _handle, input.Handle, scale.Handle));
    }

    public TensorRtLayer AddDequantize(TensorRtTensor input, TensorRtTensor scale)
    {
        ValidateInputTensor(input, nameof(input));
        ValidateInputTensor(scale, nameof(scale));
        return new TensorRtLayer(Line, NativeBridgeApi.AddDequantizeLayer(Line, _handle, input.Handle, scale.Handle));
    }

    /// <summary>
    /// Adds a TensorRT 10 quantize layer and selects the requested output type at creation time.
    /// 添加 TensorRT 10 Quantize 层，并在创建时指定输出数据类型。
    /// </summary>
    /// <param name="input">Floating-point input tensor. / 浮点输入张量。</param>
    /// <param name="scale">Scale tensor used by quantization. / 量化使用的 scale 张量。</param>
    /// <param name="outputType">Requested quantized output type. / 量化输出数据类型。</param>
    /// <returns>The created network-owned layer. / 返回由网络持有生命周期的层对象。</returns>
    public TensorRtLayer AddQuantizeV2(TensorRtTensor input, TensorRtTensor scale, TensorRtDataType outputType)
    {
        ValidateInputTensor(input, nameof(input));
        ValidateInputTensor(scale, nameof(scale));
        return new TensorRtLayer(Line, NativeBridgeApi.AddQuantizeV2Layer(Line, _handle, input.Handle, scale.Handle, outputType));
    }

    /// <summary>
    /// Adds a TensorRT 10 or TensorRT 11 dequantize layer and selects the requested output type at creation time.
    /// 添加 TensorRT 10 或 TensorRT 11 Dequantize 层，并在创建时指定输出数据类型。
    /// </summary>
    /// <param name="input">Quantized input tensor. / 量化输入张量。</param>
    /// <param name="scale">Scale tensor used by dequantization. / 反量化使用的 scale 张量。</param>
    /// <param name="outputType">Requested dequantized output type. / 反量化输出数据类型。</param>
    /// <returns>The created network-owned layer. / 返回由网络持有生命周期的层对象。</returns>
    public TensorRtLayer AddDequantizeV2(TensorRtTensor input, TensorRtTensor scale, TensorRtDataType outputType)
    {
        ValidateInputTensor(input, nameof(input));
        ValidateInputTensor(scale, nameof(scale));
        return new TensorRtLayer(Line, NativeBridgeApi.AddDequantizeV2Layer(Line, _handle, input.Handle, scale.Handle, outputType));
    }
}
