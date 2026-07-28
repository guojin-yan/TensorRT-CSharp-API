using System;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtLayer
{
    /// <summary>
    /// Gets whether a TensorRT 11 rotary-embedding layer uses interleaved RoPE layout.
    /// 获取 TensorRT 11 rotary-embedding 层是否使用 interleaved RoPE 布局。
    /// </summary>
    public bool GetRotaryEmbeddingInterleaved()
    {
        return NativeBridgeApi.GetRotaryEmbeddingInterleaved(Line, _handle);
    }

    /// <summary>
    /// Sets whether a TensorRT 11 rotary-embedding layer uses interleaved RoPE layout.
    /// 设置 TensorRT 11 rotary-embedding 层是否使用 interleaved RoPE 布局。
    /// </summary>
    public void SetRotaryEmbeddingInterleaved(bool interleaved)
    {
        NativeBridgeApi.SetRotaryEmbeddingInterleaved(Line, _handle, interleaved);
    }

    /// <summary>
    /// Gets the rotary embedding dimension configured on a TensorRT 11 rotary-embedding layer.
    /// 获取 TensorRT 11 rotary-embedding 层配置的 rotary embedding 维度。
    /// </summary>
    public int GetRotaryEmbeddingDimension()
    {
        return NativeBridgeApi.GetRotaryEmbeddingDimension(Line, _handle);
    }

    /// <summary>
    /// Sets the rotary embedding dimension configured on a TensorRT 11 rotary-embedding layer.
    /// 设置 TensorRT 11 rotary-embedding 层配置的 rotary embedding 维度。
    /// </summary>
    /// <returns><see langword="true"/> when TensorRT accepts the value. TensorRT 接受该值时返回 <see langword="true"/>。</returns>
    public bool SetRotaryEmbeddingDimension(int rotaryEmbeddingDimension)
    {
        if (rotaryEmbeddingDimension < 0 || (rotaryEmbeddingDimension % 2) != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(rotaryEmbeddingDimension), "Rotary embedding dimension must be a non-negative even value.");
        }

        return NativeBridgeApi.SetRotaryEmbeddingDimension(Line, _handle, rotaryEmbeddingDimension);
    }

    /// <summary>
    /// Gets the cache mode of a TensorRT 11 KV-cache update layer.
    /// 获取 TensorRT 11 KV-cache update 层的 cache mode。
    /// </summary>
    public TensorRtKvCacheMode GetKvCacheUpdateMode()
    {
        return NativeBridgeApi.GetKvCacheUpdateMode(Line, _handle);
    }

    /// <summary>
    /// Sets the cache mode of a TensorRT 11 KV-cache update layer.
    /// 设置 TensorRT 11 KV-cache update 层的 cache mode。
    /// </summary>
    /// <returns><see langword="true"/> when TensorRT accepts the mode. TensorRT 接受该模式时返回 <see langword="true"/>。</returns>
    public bool SetKvCacheUpdateMode(TensorRtKvCacheMode mode)
    {
        return NativeBridgeApi.SetKvCacheUpdateMode(Line, _handle, mode);
    }

    /// <summary>
    /// Gets the update form of a TensorRT 11 KV-cache update layer.
    /// 获取 TensorRT 11 KV-cache update 层的 update form。
    /// </summary>
    public TensorRtAttentionIoForm GetKvCacheUpdateForm()
    {
        return NativeBridgeApi.GetKvCacheUpdateForm(Line, _handle);
    }

    /// <summary>
    /// Sets the update form of a TensorRT 11 KV-cache update layer.
    /// 设置 TensorRT 11 KV-cache update 层的 update form。
    /// </summary>
    /// <returns><see langword="true"/> when TensorRT accepts the form. TensorRT 接受该排布时返回 <see langword="true"/>。</returns>
    public bool SetKvCacheUpdateForm(TensorRtAttentionIoForm form)
    {
        return NativeBridgeApi.SetKvCacheUpdateForm(Line, _handle, form);
    }

    /// <summary>
    /// Sets the update-lengths tensor of a TensorRT 11 KV-cache update layer.
    /// 设置 TensorRT 11 KV-cache update 层的 update-lengths 张量。
    /// </summary>
    /// <returns><see langword="true"/> when TensorRT accepts the tensor. TensorRT 接受该张量时返回 <see langword="true"/>。</returns>
    public bool SetKvCacheUpdateLengths(TensorRtTensor lengths)
    {
        ValidateLayerTensor(lengths, nameof(lengths));
        return NativeBridgeApi.SetKvCacheUpdateLengths(Line, _handle, lengths.Handle);
    }

    /// <summary>
    /// Tries to get the optional update-lengths tensor from a TensorRT 11 KV-cache update layer.
    /// 尝试获取 TensorRT 11 KV-cache update 层上的可选 update-lengths 张量。
    /// </summary>
    public bool TryGetKvCacheUpdateLengths(out TensorRtTensor? lengths)
    {
        SafeTensorRtObjectHandle? handle = NativeBridgeApi.GetKvCacheUpdateLengths(Line, _handle);
        if (handle == null)
        {
            lengths = null;
            return false;
        }

        lengths = new TensorRtTensor(Line, handle);
        return true;
    }

    /// <summary>
    /// Gets the target output type of a TensorRT 11 dequantize layer.
    /// 获取 TensorRT 11 dequantize 层的目标输出类型。
    /// </summary>
    public TensorRtDataType GetDequantizeToType()
    {
        return NativeBridgeApi.GetDequantizeToType(Line, _handle);
    }

    /// <summary>
    /// Sets the target output type of a TensorRT 11 dequantize layer.
    /// 设置 TensorRT 11 dequantize 层的目标输出类型。
    /// </summary>
    public void SetDequantizeToType(TensorRtDataType dataType)
    {
        NativeBridgeApi.SetDequantizeToType(Line, _handle, dataType);
    }

    /// <summary>
    /// Gets the output type of a TensorRT 11 fill layer.
    /// 获取 TensorRT 11 fill 层的输出类型。
    /// </summary>
    public TensorRtDataType GetFillToType()
    {
        return NativeBridgeApi.GetFillToType(Line, _handle);
    }

    /// <summary>
    /// Sets the output type of a TensorRT 11 fill layer.
    /// 设置 TensorRT 11 fill 层的输出类型。
    /// </summary>
    public void SetFillToType(TensorRtDataType dataType)
    {
        NativeBridgeApi.SetFillToType(Line, _handle, dataType);
    }

    /// <summary>
    /// Gets the activation type configured on a TensorRT 11 MoE layer.
    /// 获取 TensorRT 11 MoE 层配置的激活类型。
    /// </summary>
    public TensorRtMoEActivationType GetMoEActivationType()
    {
        return NativeBridgeApi.GetMoEActivationType(Line, _handle);
    }

    /// <summary>
    /// Sets the activation type configured on a TensorRT 11 MoE layer.
    /// 设置 TensorRT 11 MoE 层配置的激活类型。
    /// </summary>
    public void SetMoEActivationType(TensorRtMoEActivationType activationType)
    {
        NativeBridgeApi.SetMoEActivationType(Line, _handle, activationType);
    }

    /// <summary>
    /// Gets the quantized type used for the MoE mul output.
    /// 获取 MoE mul 输出使用的量化类型。
    /// </summary>
    public TensorRtDataType GetMoEQuantizationToType()
    {
        return NativeBridgeApi.GetMoEQuantizationToType(Line, _handle);
    }

    /// <summary>
    /// Sets the quantized type used for the MoE mul output.
    /// 设置 MoE mul 输出使用的量化类型。
    /// </summary>
    public void SetMoEQuantizationToType(TensorRtDataType dataType)
    {
        NativeBridgeApi.SetMoEQuantizationToType(Line, _handle, dataType);
    }

    /// <summary>
    /// Gets the quantization block shape used for the MoE mul output.
    /// 获取 MoE mul 输出使用的量化 block shape。
    /// </summary>
    public TensorRtDims GetMoEQuantizationBlockShape()
    {
        return NativeBridgeApi.GetMoEQuantizationBlockShape(Line, _handle);
    }

    /// <summary>
    /// Sets the quantization block shape used for the MoE mul output.
    /// 设置 MoE mul 输出使用的量化 block shape。
    /// </summary>
    public void SetMoEQuantizationBlockShape(TensorRtDims blockShape)
    {
        NativeBridgeApi.SetMoEQuantizationBlockShape(Line, _handle, blockShape);
    }

    /// <summary>
    /// Gets the data type of the dynamic-quantization output scale on a TensorRT 11 MoE layer.
    /// 获取 TensorRT 11 MoE 层动态量化输出 scale 的数据类型。
    /// </summary>
    public TensorRtDataType GetMoEDynamicQuantizationOutputScaleType()
    {
        return NativeBridgeApi.GetMoEDynamicQuantizationOutputScaleType(Line, _handle);
    }

    /// <summary>
    /// Sets the data type of the dynamic-quantization output scale on a TensorRT 11 MoE layer.
    /// 设置 TensorRT 11 MoE 层动态量化输出 scale 的数据类型。
    /// </summary>
    public void SetMoEDynamicQuantizationOutputScaleType(TensorRtDataType dataType)
    {
        NativeBridgeApi.SetMoEDynamicQuantizationOutputScaleType(Line, _handle, dataType);
    }

    /// <summary>
    /// Sets the gated expert weight tensors for a TensorRT 11 MoE layer.
    /// 设置 TensorRT 11 MoE 层的 gated expert 权重张量。
    /// </summary>
    public void SetMoEGatedWeights(
        TensorRtTensor fcGateWeights,
        TensorRtTensor fcUpWeights,
        TensorRtTensor fcDownWeights,
        TensorRtMoEActivationType activationType)
    {
        ValidateLayerTensor(fcGateWeights, nameof(fcGateWeights));
        ValidateLayerTensor(fcUpWeights, nameof(fcUpWeights));
        ValidateLayerTensor(fcDownWeights, nameof(fcDownWeights));
        NativeBridgeApi.SetMoEGatedWeights(
            Line,
            _handle,
            fcGateWeights.Handle,
            fcUpWeights.Handle,
            fcDownWeights.Handle,
            activationType);
    }

    /// <summary>
    /// Sets all optional gated expert bias tensors for a TensorRT 11 MoE layer.
    /// 设置 TensorRT 11 MoE 层的全部可选 gated expert bias 张量。
    /// </summary>
    public void SetMoEGatedBiases(
        TensorRtTensor fcGateBiases,
        TensorRtTensor fcUpBiases,
        TensorRtTensor fcDownBiases)
    {
        ValidateLayerTensor(fcGateBiases, nameof(fcGateBiases));
        ValidateLayerTensor(fcUpBiases, nameof(fcUpBiases));
        ValidateLayerTensor(fcDownBiases, nameof(fcDownBiases));
        NativeBridgeApi.SetMoEGatedBiases(
            Line,
            _handle,
            fcGateBiases.Handle,
            fcUpBiases.Handle,
            fcDownBiases.Handle);
    }

    /// <summary>
    /// Configures static quantization after the MoE mul operation.
    /// 配置 MoE mul 操作后的静态量化。
    /// </summary>
    public void SetMoEQuantizationStatic(TensorRtTensor fcDownActivationScale, TensorRtDataType dataType)
    {
        ValidateLayerTensor(fcDownActivationScale, nameof(fcDownActivationScale));
        NativeBridgeApi.SetMoEQuantizationStatic(Line, _handle, fcDownActivationScale.Handle, dataType);
    }

    /// <summary>
    /// Configures dynamic double quantization after the MoE mul operation.
    /// 配置 MoE mul 操作后的动态 double quantization。
    /// </summary>
    public void SetMoEQuantizationDynamicDblQ(
        TensorRtTensor fcDownActivationDblQScale,
        TensorRtDataType dataType,
        TensorRtDims blockShape,
        TensorRtDataType dynamicQuantizationOutputScaleType)
    {
        ValidateLayerTensor(fcDownActivationDblQScale, nameof(fcDownActivationDblQScale));
        if (blockShape == null)
        {
            throw new ArgumentNullException(nameof(blockShape));
        }

        NativeBridgeApi.SetMoEQuantizationDynamicDblQ(
            Line,
            _handle,
            fcDownActivationDblQScale.Handle,
            dataType,
            blockShape,
            dynamicQuantizationOutputScaleType);
    }

    /// <summary>
    /// Sets all SwiGLU parameters on a TensorRT 11 MoE layer.
    /// 一次性设置 TensorRT 11 MoE 层的全部 SwiGLU 参数。
    /// </summary>
    public void SetMoESwigluParameters(float limit, float alpha, float beta)
    {
        NativeBridgeApi.SetMoESwigluParameters(Line, _handle, limit, alpha, beta);
    }

    /// <summary>
    /// Gets the SwiGLU limit parameter from a TensorRT 11 MoE layer.
    /// 获取 TensorRT 11 MoE 层的 SwiGLU limit 参数。
    /// </summary>
    public float GetMoESwigluLimit()
    {
        return NativeBridgeApi.GetMoESwigluLimit(Line, _handle);
    }

    /// <summary>
    /// Sets the SwiGLU limit parameter on a TensorRT 11 MoE layer.
    /// 设置 TensorRT 11 MoE 层的 SwiGLU limit 参数。
    /// </summary>
    public void SetMoESwigluLimit(float limit)
    {
        NativeBridgeApi.SetMoESwigluLimit(Line, _handle, limit);
    }

    /// <summary>
    /// Gets the SwiGLU alpha parameter from a TensorRT 11 MoE layer.
    /// 获取 TensorRT 11 MoE 层的 SwiGLU alpha 参数。
    /// </summary>
    public float GetMoESwigluAlpha()
    {
        return NativeBridgeApi.GetMoESwigluAlpha(Line, _handle);
    }

    /// <summary>
    /// Sets the SwiGLU alpha parameter on a TensorRT 11 MoE layer.
    /// 设置 TensorRT 11 MoE 层的 SwiGLU alpha 参数。
    /// </summary>
    public void SetMoESwigluAlpha(float alpha)
    {
        NativeBridgeApi.SetMoESwigluAlpha(Line, _handle, alpha);
    }

    /// <summary>
    /// Gets the SwiGLU beta parameter from a TensorRT 11 MoE layer.
    /// 获取 TensorRT 11 MoE 层的 SwiGLU beta 参数。
    /// </summary>
    public float GetMoESwigluBeta()
    {
        return NativeBridgeApi.GetMoESwigluBeta(Line, _handle);
    }

    /// <summary>
    /// Sets the SwiGLU beta parameter on a TensorRT 11 MoE layer.
    /// 设置 TensorRT 11 MoE 层的 SwiGLU beta 参数。
    /// </summary>
    public void SetMoESwigluBeta(float beta)
    {
        NativeBridgeApi.SetMoESwigluBeta(Line, _handle, beta);
    }

}
