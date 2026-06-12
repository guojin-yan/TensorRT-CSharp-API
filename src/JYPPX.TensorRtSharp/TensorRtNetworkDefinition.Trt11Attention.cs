namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtNetworkDefinition
{
    /// <summary>
    /// Adds a TensorRT 11 attention object to the network.
    /// 向 network 添加一个 TensorRT 11 attention 对象。
    /// </summary>
    /// <param name="query">Query tensor. / Query 张量。</param>
    /// <param name="key">Key tensor. / Key 张量。</param>
    /// <param name="value">Value tensor. / Value 张量。</param>
    /// <param name="normalizationOperation">Normalization operation inside attention. / attention 内部归一化操作。</param>
    /// <param name="causalKind">Causal mask alignment kind. / 因果 mask 对齐方向。</param>
    /// <returns>The network-owned TensorRT attention wrapper. / network 持有生命周期的 TensorRT attention 包装对象。</returns>
    public TensorRtAttention AddAttentionV2(
        TensorRtTensor query,
        TensorRtTensor key,
        TensorRtTensor value,
        TensorRtAttentionNormalizationOperation normalizationOperation = TensorRtAttentionNormalizationOperation.Softmax,
        TensorRtCausalMaskKind causalKind = TensorRtCausalMaskKind.None)
    {
        ValidateInputTensor(query, nameof(query));
        ValidateInputTensor(key, nameof(key));
        ValidateInputTensor(value, nameof(value));
        return new TensorRtAttention(
            Line,
            Internal.Interop.NativeBridgeApi.AddAttentionV2(
                Line,
                _handle,
                query.Handle,
                key.Handle,
                value.Handle,
                normalizationOperation,
                causalKind));
    }
}
