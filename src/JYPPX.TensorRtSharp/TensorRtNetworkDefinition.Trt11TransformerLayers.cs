using System;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtNetworkDefinition
{
    /// <summary>
    /// Adds a TensorRT 11 rotary-embedding layer.
    /// 添加 TensorRT 11 rotary-embedding 层。
    /// </summary>
    /// <remarks>
    /// TensorRT validates the RoPE cache tensor shapes and whether <paramref name="rotaryEmbeddingDimension"/> is legal for the input.
    /// RoPE cache 张量形状以及 <paramref name="rotaryEmbeddingDimension"/> 是否合法由 TensorRT 原生侧校验。
    /// </remarks>
    public TensorRtLayer AddRotaryEmbedding(
        TensorRtTensor input,
        TensorRtTensor cosCache,
        TensorRtTensor sinCache,
        bool interleaved,
        int rotaryEmbeddingDimension)
    {
        ValidateInputTensor(input, nameof(input));
        ValidateInputTensor(cosCache, nameof(cosCache));
        ValidateInputTensor(sinCache, nameof(sinCache));
        if (rotaryEmbeddingDimension < 0 || (rotaryEmbeddingDimension % 2) != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(rotaryEmbeddingDimension), "Rotary embedding dimension must be a non-negative even value.");
        }

        return new TensorRtLayer(
            Line,
            Internal.Interop.NativeBridgeApi.AddRotaryEmbeddingLayer(
                Line,
                _handle,
                input.Handle,
                cosCache.Handle,
                sinCache.Handle,
                interleaved,
                rotaryEmbeddingDimension));
    }

    /// <summary>
    /// Adds a TensorRT 11 KV-cache update layer.
    /// 添加 TensorRT 11 KV-cache update 层。
    /// </summary>
    /// <remarks>
    /// The returned layer can be configured with update-form and optional update-length metadata.
    /// 返回层可继续配置 update form 以及可选 update lengths 元数据。
    /// </remarks>
    public TensorRtLayer AddKvCacheUpdate(
        TensorRtTensor cache,
        TensorRtTensor update,
        TensorRtTensor writeIndices,
        TensorRtKvCacheMode cacheMode = TensorRtKvCacheMode.Linear)
    {
        ValidateInputTensor(cache, nameof(cache));
        ValidateInputTensor(update, nameof(update));
        ValidateInputTensor(writeIndices, nameof(writeIndices));
        return new TensorRtLayer(
            Line,
            Internal.Interop.NativeBridgeApi.AddKvCacheUpdateLayer(
                Line,
                _handle,
                cache.Handle,
                update.Handle,
                writeIndices.Handle,
                cacheMode));
    }

    /// <summary>
    /// Adds a TensorRT 11 MoE layer.
    /// 添加 TensorRT 11 MoE 层。
    /// </summary>
    /// <remarks>
    /// This wraps TensorRT's lightweight MoE layer creation only; expert weights and quantization metadata are configured on the returned layer.
    /// 此方法仅封装 TensorRT 轻量级 MoE 层创建；专家权重和量化元数据在返回层上继续配置。
    /// </remarks>
    public TensorRtLayer AddMoE(
        TensorRtTensor hiddenStates,
        TensorRtTensor selectedExpertsForTokens,
        TensorRtTensor scoresForSelectedExperts)
    {
        ValidateInputTensor(hiddenStates, nameof(hiddenStates));
        ValidateInputTensor(selectedExpertsForTokens, nameof(selectedExpertsForTokens));
        ValidateInputTensor(scoresForSelectedExperts, nameof(scoresForSelectedExperts));
        return new TensorRtLayer(
            Line,
            Internal.Interop.NativeBridgeApi.AddMoELayer(
                Line,
                _handle,
                hiddenStates.Handle,
                selectedExpertsForTokens.Handle,
                scoresForSelectedExperts.Handle));
    }
}
