using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle AddRotaryEmbeddingLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle input,
        SafeTensorRtObjectHandle cosCache,
        SafeTensorRtObjectHandle sinCache,
        bool interleaved,
        int rotaryEmbeddingDimension)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(AddRotaryEmbeddingLayer));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_network_add_rotary_embedding(
            network,
            input,
            cosCache,
            sinCache,
            interleaved ? 1 : 0,
            rotaryEmbeddingDimension,
            out SafeTensorRtObjectHandle layer);
        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddKvCacheUpdateLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle cache,
        SafeTensorRtObjectHandle update,
        SafeTensorRtObjectHandle writeIndices,
        TensorRtKvCacheMode cacheMode)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(AddKvCacheUpdateLayer));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_network_add_kv_cache_update(
            network,
            cache,
            update,
            writeIndices,
            (int)cacheMode,
            out SafeTensorRtObjectHandle layer);
        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddMoELayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle hiddenStates,
        SafeTensorRtObjectHandle selectedExperts,
        SafeTensorRtObjectHandle scores)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(AddMoELayer));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_network_add_moe(
            network,
            hiddenStates,
            selectedExperts,
            scores,
            out SafeTensorRtObjectHandle layer);
        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static void SetRotaryEmbeddingInterleaved(TensorRtApiLine line, SafeTensorRtObjectHandle layer, bool interleaved)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetRotaryEmbeddingInterleaved));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_rotary_embedding_layer_set_interleaved(layer, interleaved ? 1 : 0);
        NativeStatus.ThrowIfFailed(status);
    }

    public static bool GetRotaryEmbeddingInterleaved(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetRotaryEmbeddingInterleaved));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_rotary_embedding_layer_get_interleaved(layer, out int interleaved);
        NativeStatus.ThrowIfFailed(status);
        return interleaved != 0;
    }

    public static bool SetRotaryEmbeddingDimension(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int rotaryEmbeddingDimension)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetRotaryEmbeddingDimension));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_rotary_embedding_layer_set_rotary_embedding_dim(layer, rotaryEmbeddingDimension, out int success);
        NativeStatus.ThrowIfFailed(status);
        return success != 0;
    }

    public static int GetRotaryEmbeddingDimension(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetRotaryEmbeddingDimension));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_rotary_embedding_layer_get_rotary_embedding_dim(layer, out int dimension);
        NativeStatus.ThrowIfFailed(status);
        return dimension;
    }

    public static bool SetKvCacheUpdateMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtKvCacheMode mode)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetKvCacheUpdateMode));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_kv_cache_update_layer_set_cache_mode(layer, (int)mode, out int success);
        NativeStatus.ThrowIfFailed(status);
        return success != 0;
    }

    public static TensorRtKvCacheMode GetKvCacheUpdateMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetKvCacheUpdateMode));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_kv_cache_update_layer_get_cache_mode(layer, out int mode);
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtKvCacheMode)mode;
    }

    public static bool SetKvCacheUpdateForm(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtAttentionIoForm form)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetKvCacheUpdateForm));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_kv_cache_update_layer_set_update_form(layer, (int)form, out int success);
        NativeStatus.ThrowIfFailed(status);
        return success != 0;
    }

    public static TensorRtAttentionIoForm GetKvCacheUpdateForm(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetKvCacheUpdateForm));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_kv_cache_update_layer_get_update_form(layer, out int form);
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtAttentionIoForm)form;
    }

    public static bool SetKvCacheUpdateLengths(TensorRtApiLine line, SafeTensorRtObjectHandle layer, SafeTensorRtObjectHandle lengths)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetKvCacheUpdateLengths));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_kv_cache_update_layer_set_update_lengths(layer, lengths, out int success);
        NativeStatus.ThrowIfFailed(status);
        return success != 0;
    }

    public static SafeTensorRtObjectHandle? GetKvCacheUpdateLengths(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetKvCacheUpdateLengths));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_kv_cache_update_layer_get_update_lengths(layer, out SafeTensorRtObjectHandle tensor, out int hasTensor);
        NativeStatus.ThrowIfFailed(status);
        if (hasTensor == 0)
        {
            tensor?.Dispose();
            return null;
        }

        return tensor;
    }

    public static void SetDequantizeToType(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDataType dataType)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetDequantizeToType));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_dequantize_layer_set_to_type(layer, (int)dataType);
        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDataType GetDequantizeToType(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetDequantizeToType));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_dequantize_layer_get_to_type(layer, out int dataType);
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtDataType)dataType;
    }

    public static void SetFillToType(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDataType dataType)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetFillToType));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_fill_layer_set_to_type(layer, (int)dataType);
        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDataType GetFillToType(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetFillToType));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_fill_layer_get_to_type(layer, out int dataType);
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtDataType)dataType;
    }

    public static void SetMoEActivationType(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtMoEActivationType activationType)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetMoEActivationType));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_moe_layer_set_activation_type(layer, (int)activationType);
        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtMoEActivationType GetMoEActivationType(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetMoEActivationType));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_moe_layer_get_activation_type(layer, out int activationType);
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtMoEActivationType)activationType;
    }

    public static void SetMoEQuantizationToType(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDataType dataType)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetMoEQuantizationToType));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_moe_layer_set_quantization_to_type(layer, (int)dataType);
        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDataType GetMoEQuantizationToType(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetMoEQuantizationToType));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_moe_layer_get_quantization_to_type(layer, out int dataType);
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtDataType)dataType;
    }

    public static void SetMoEQuantizationBlockShape(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims blockShape)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetMoEQuantizationBlockShape));
        if (blockShape == null)
        {
            throw new ArgumentNullException(nameof(blockShape));
        }

        NativeTensorRtDims nativeBlockShape = blockShape.ToNative();
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_moe_layer_set_quantization_block_shape(layer, ref nativeBlockShape);
        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDims GetMoEQuantizationBlockShape(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetMoEQuantizationBlockShape));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_moe_layer_get_quantization_block_shape(layer, out NativeTensorRtDims blockShape);
        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(blockShape);
    }

    public static void SetMoEDynamicQuantizationOutputScaleType(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDataType dataType)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetMoEDynamicQuantizationOutputScaleType));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_moe_layer_set_dyn_q_output_scale_type(layer, (int)dataType);
        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDataType GetMoEDynamicQuantizationOutputScaleType(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetMoEDynamicQuantizationOutputScaleType));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_moe_layer_get_dyn_q_output_scale_type(layer, out int dataType);
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtDataType)dataType;
    }

    public static void SetMoEGatedWeights(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer,
        SafeTensorRtObjectHandle fcGateWeights,
        SafeTensorRtObjectHandle fcUpWeights,
        SafeTensorRtObjectHandle fcDownWeights,
        TensorRtMoEActivationType activationType)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetMoEGatedWeights));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_moe_layer_set_gated_weights(
            layer,
            fcGateWeights,
            fcUpWeights,
            fcDownWeights,
            (int)activationType);
        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetMoEGatedBiases(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer,
        SafeTensorRtObjectHandle fcGateBiases,
        SafeTensorRtObjectHandle fcUpBiases,
        SafeTensorRtObjectHandle fcDownBiases)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetMoEGatedBiases));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_moe_layer_set_gated_biases(
            layer,
            fcGateBiases,
            fcUpBiases,
            fcDownBiases);
        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetMoEQuantizationStatic(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer,
        SafeTensorRtObjectHandle fcDownActivationScale,
        TensorRtDataType dataType)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetMoEQuantizationStatic));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_moe_layer_set_quantization_static(
            layer,
            fcDownActivationScale,
            (int)dataType);
        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetMoEQuantizationDynamicDblQ(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer,
        SafeTensorRtObjectHandle fcDownActivationDblQScale,
        TensorRtDataType dataType,
        TensorRtDims blockShape,
        TensorRtDataType dynamicQuantizationOutputScaleType)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetMoEQuantizationDynamicDblQ));
        if (blockShape == null)
        {
            throw new ArgumentNullException(nameof(blockShape));
        }

        NativeTensorRtDims nativeBlockShape = blockShape.ToNative();
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_moe_layer_set_quantization_dynamic_dbl_q(
            layer,
            fcDownActivationDblQScale,
            (int)dataType,
            ref nativeBlockShape,
            (int)dynamicQuantizationOutputScaleType);
        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetMoESwigluParameters(TensorRtApiLine line, SafeTensorRtObjectHandle layer, float limit, float alpha, float beta)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetMoESwigluParameters));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_moe_layer_set_swiglu_params(layer, limit, alpha, beta);
        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetMoESwigluLimit(TensorRtApiLine line, SafeTensorRtObjectHandle layer, float limit)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetMoESwigluLimit));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_moe_layer_set_swiglu_param_limit(layer, limit);
        NativeStatus.ThrowIfFailed(status);
    }

    public static float GetMoESwigluLimit(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetMoESwigluLimit));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_moe_layer_get_swiglu_param_limit(layer, out float limit);
        NativeStatus.ThrowIfFailed(status);
        return limit;
    }

    public static void SetMoESwigluAlpha(TensorRtApiLine line, SafeTensorRtObjectHandle layer, float alpha)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetMoESwigluAlpha));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_moe_layer_set_swiglu_param_alpha(layer, alpha);
        NativeStatus.ThrowIfFailed(status);
    }

    public static float GetMoESwigluAlpha(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetMoESwigluAlpha));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_moe_layer_get_swiglu_param_alpha(layer, out float alpha);
        NativeStatus.ThrowIfFailed(status);
        return alpha;
    }

    public static void SetMoESwigluBeta(TensorRtApiLine line, SafeTensorRtObjectHandle layer, float beta)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetMoESwigluBeta));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_moe_layer_set_swiglu_param_beta(layer, beta);
        NativeStatus.ThrowIfFailed(status);
    }

    public static float GetMoESwigluBeta(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetMoESwigluBeta));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_moe_layer_get_swiglu_param_beta(layer, out float beta);
        NativeStatus.ThrowIfFailed(status);
        return beta;
    }
}
