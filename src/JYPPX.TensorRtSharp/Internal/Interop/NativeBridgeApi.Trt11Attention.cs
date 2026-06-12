using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle AddAttentionV2(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle query,
        SafeTensorRtObjectHandle key,
        SafeTensorRtObjectHandle value,
        TensorRtAttentionNormalizationOperation normalizationOperation,
        TensorRtCausalMaskKind causalKind)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(AddAttentionV2));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_network_add_attention_v2(
            network,
            query,
            key,
            value,
            (int)normalizationOperation,
            (int)causalKind,
            out SafeTensorRtObjectHandle attention);
        NativeStatus.ThrowIfFailed(status);
        return attention;
    }

    public static bool SetAttentionNormalizationOperation(TensorRtApiLine line, SafeTensorRtObjectHandle attention, TensorRtAttentionNormalizationOperation operation)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetAttentionNormalizationOperation));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_set_normalization_operation(attention, (int)operation, out int success);
        NativeStatus.ThrowIfFailed(status);
        return success != 0;
    }

    public static TensorRtAttentionNormalizationOperation GetAttentionNormalizationOperation(TensorRtApiLine line, SafeTensorRtObjectHandle attention)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetAttentionNormalizationOperation));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_get_normalization_operation(attention, out int operation);
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtAttentionNormalizationOperation)operation;
    }

    public static bool SetAttentionMask(TensorRtApiLine line, SafeTensorRtObjectHandle attention, SafeTensorRtObjectHandle mask)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetAttentionMask));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_set_mask(attention, mask, out int success);
        NativeStatus.ThrowIfFailed(status);
        return success != 0;
    }

    public static SafeTensorRtObjectHandle? GetAttentionMask(TensorRtApiLine line, SafeTensorRtObjectHandle attention)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetAttentionMask));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_get_mask(attention, out SafeTensorRtObjectHandle tensor, out int hasTensor);
        NativeStatus.ThrowIfFailed(status);
        if (hasTensor == 0)
        {
            tensor?.Dispose();
            return null;
        }

        return tensor;
    }

    public static bool SetAttentionCausalKind(TensorRtApiLine line, SafeTensorRtObjectHandle attention, TensorRtCausalMaskKind causalKind)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetAttentionCausalKind));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_set_causal_kind(attention, (int)causalKind, out int success);
        NativeStatus.ThrowIfFailed(status);
        return success != 0;
    }

    public static TensorRtCausalMaskKind GetAttentionCausalKind(TensorRtApiLine line, SafeTensorRtObjectHandle attention)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetAttentionCausalKind));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_get_causal_kind(attention, out int causalKind);
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtCausalMaskKind)causalKind;
    }

    public static bool SetAttentionDecomposable(TensorRtApiLine line, SafeTensorRtObjectHandle attention, bool decomposable)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetAttentionDecomposable));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_set_decomposable(attention, decomposable ? 1 : 0, out int success);
        NativeStatus.ThrowIfFailed(status);
        return success != 0;
    }

    public static bool GetAttentionDecomposable(TensorRtApiLine line, SafeTensorRtObjectHandle attention)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetAttentionDecomposable));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_get_decomposable(attention, out int decomposable);
        NativeStatus.ThrowIfFailed(status);
        return decomposable != 0;
    }

    public static bool SetAttentionInput(TensorRtApiLine line, SafeTensorRtObjectHandle attention, int index, SafeTensorRtObjectHandle input)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetAttentionInput));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_set_input(attention, index, input, out int success);
        NativeStatus.ThrowIfFailed(status);
        return success != 0;
    }

    public static int GetAttentionInputCount(TensorRtApiLine line, SafeTensorRtObjectHandle attention)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetAttentionInputCount));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_get_input_count(attention, out int count);
        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static SafeTensorRtObjectHandle GetAttentionInput(TensorRtApiLine line, SafeTensorRtObjectHandle attention, int index)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetAttentionInput));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_get_input(attention, index, out SafeTensorRtObjectHandle tensor);
        NativeStatus.ThrowIfFailed(status);
        return tensor;
    }

    public static int GetAttentionOutputCount(TensorRtApiLine line, SafeTensorRtObjectHandle attention)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetAttentionOutputCount));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_get_output_count(attention, out int count);
        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static SafeTensorRtObjectHandle GetAttentionOutput(TensorRtApiLine line, SafeTensorRtObjectHandle attention, int index)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetAttentionOutput));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_get_output(attention, index, out SafeTensorRtObjectHandle tensor);
        NativeStatus.ThrowIfFailed(status);
        return tensor;
    }

    public static bool SetAttentionName(TensorRtApiLine line, SafeTensorRtObjectHandle attention, string name)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetAttentionName));
        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(name ?? string.Empty);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_set_name(attention, nameUtf8.Pointer, out int success);
        NativeStatus.ThrowIfFailed(status);
        return success != 0;
    }

    public static string GetAttentionName(TensorRtApiLine line, SafeTensorRtObjectHandle attention)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetAttentionName));
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt11_attention_get_name(attention, buffer, size, out required),
            "Attention name is too large for the managed buffer.");
    }

    public static bool SetAttentionNormalizationQuantizeScale(TensorRtApiLine line, SafeTensorRtObjectHandle attention, SafeTensorRtObjectHandle scale)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetAttentionNormalizationQuantizeScale));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_set_normalization_quantize_scale(attention, scale, out int success);
        NativeStatus.ThrowIfFailed(status);
        return success != 0;
    }

    public static SafeTensorRtObjectHandle? GetAttentionNormalizationQuantizeScale(TensorRtApiLine line, SafeTensorRtObjectHandle attention)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetAttentionNormalizationQuantizeScale));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_get_normalization_quantize_scale(attention, out SafeTensorRtObjectHandle tensor, out int hasTensor);
        NativeStatus.ThrowIfFailed(status);
        if (hasTensor == 0)
        {
            tensor?.Dispose();
            return null;
        }

        return tensor;
    }

    public static bool SetAttentionNormalizationQuantizeToType(TensorRtApiLine line, SafeTensorRtObjectHandle attention, TensorRtDataType dataType)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetAttentionNormalizationQuantizeToType));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_set_normalization_quantize_to_type(attention, (int)dataType, out int success);
        NativeStatus.ThrowIfFailed(status);
        return success != 0;
    }

    public static TensorRtDataType GetAttentionNormalizationQuantizeToType(TensorRtApiLine line, SafeTensorRtObjectHandle attention)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetAttentionNormalizationQuantizeToType));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_get_normalization_quantize_to_type(attention, out int dataType);
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtDataType)dataType;
    }

    public static bool SetAttentionMetadata(TensorRtApiLine line, SafeTensorRtObjectHandle attention, string metadata)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetAttentionMetadata));
        using Utf8Interop.Utf8StringScope metadataUtf8 = Utf8Interop.ToNativeString(metadata ?? string.Empty);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_set_metadata(attention, metadataUtf8.Pointer, out int success);
        NativeStatus.ThrowIfFailed(status);
        return success != 0;
    }

    public static string GetAttentionMetadata(TensorRtApiLine line, SafeTensorRtObjectHandle attention)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetAttentionMetadata));
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt11_attention_get_metadata(attention, buffer, size, out required),
            "Attention metadata is too large for the managed buffer.");
    }

    public static bool SetAttentionRankCount(TensorRtApiLine line, SafeTensorRtObjectHandle attention, int rankCount)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetAttentionRankCount));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_set_nb_ranks(attention, rankCount, out int success);
        NativeStatus.ThrowIfFailed(status);
        return success != 0;
    }

    public static int GetAttentionRankCount(TensorRtApiLine line, SafeTensorRtObjectHandle attention)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetAttentionRankCount));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_get_nb_ranks(attention, out int rankCount);
        NativeStatus.ThrowIfFailed(status);
        return rankCount;
    }

    public static bool SetAttentionQueryForm(TensorRtApiLine line, SafeTensorRtObjectHandle attention, TensorRtAttentionIoForm form)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetAttentionQueryForm));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_set_query_form(attention, (int)form, out int success);
        NativeStatus.ThrowIfFailed(status);
        return success != 0;
    }

    public static TensorRtAttentionIoForm GetAttentionQueryForm(TensorRtApiLine line, SafeTensorRtObjectHandle attention)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetAttentionQueryForm));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_get_query_form(attention, out int form);
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtAttentionIoForm)form;
    }

    public static bool SetAttentionKeyValueForm(TensorRtApiLine line, SafeTensorRtObjectHandle attention, TensorRtAttentionIoForm form)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetAttentionKeyValueForm));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_set_key_value_form(attention, (int)form, out int success);
        NativeStatus.ThrowIfFailed(status);
        return success != 0;
    }

    public static TensorRtAttentionIoForm GetAttentionKeyValueForm(TensorRtApiLine line, SafeTensorRtObjectHandle attention)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetAttentionKeyValueForm));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_get_key_value_form(attention, out int form);
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtAttentionIoForm)form;
    }

    public static bool SetAttentionQueryLengths(TensorRtApiLine line, SafeTensorRtObjectHandle attention, SafeTensorRtObjectHandle lengths)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetAttentionQueryLengths));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_set_query_lengths(attention, lengths, out int success);
        NativeStatus.ThrowIfFailed(status);
        return success != 0;
    }

    public static SafeTensorRtObjectHandle? GetAttentionQueryLengths(TensorRtApiLine line, SafeTensorRtObjectHandle attention)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetAttentionQueryLengths));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_get_query_lengths(attention, out SafeTensorRtObjectHandle tensor, out int hasTensor);
        NativeStatus.ThrowIfFailed(status);
        if (hasTensor == 0)
        {
            tensor?.Dispose();
            return null;
        }

        return tensor;
    }

    public static bool SetAttentionKeyValueLengths(TensorRtApiLine line, SafeTensorRtObjectHandle attention, SafeTensorRtObjectHandle lengths)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetAttentionKeyValueLengths));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_set_key_value_lengths(attention, lengths, out int success);
        NativeStatus.ThrowIfFailed(status);
        return success != 0;
    }

    public static SafeTensorRtObjectHandle? GetAttentionKeyValueLengths(TensorRtApiLine line, SafeTensorRtObjectHandle attention)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetAttentionKeyValueLengths));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_get_key_value_lengths(attention, out SafeTensorRtObjectHandle tensor, out int hasTensor);
        NativeStatus.ThrowIfFailed(status);
        if (hasTensor == 0)
        {
            tensor?.Dispose();
            return null;
        }

        return tensor;
    }
}
