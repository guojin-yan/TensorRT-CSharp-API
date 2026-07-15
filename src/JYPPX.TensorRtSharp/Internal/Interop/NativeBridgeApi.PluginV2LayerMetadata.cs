using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static TensorRtPluginV2LayerMetadata GetPluginV2LayerMetadata(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer)
    {
        string pluginType = GetPluginV2LayerString(
            line,
            layer,
            NativeMethodsTensorRt.jyppx_trt8_plugin_v2_layer_get_plugin_type,
            NativeMethodsTensorRt.jyppx_trt10_plugin_v2_layer_get_plugin_type,
            NativeMethodsTensorRt.jyppx_trt11_plugin_v2_layer_get_plugin_type,
            "PluginV2 type is too large for the managed buffer.");
        string pluginVersion = GetPluginV2LayerString(
            line,
            layer,
            NativeMethodsTensorRt.jyppx_trt8_plugin_v2_layer_get_plugin_version,
            NativeMethodsTensorRt.jyppx_trt10_plugin_v2_layer_get_plugin_version,
            NativeMethodsTensorRt.jyppx_trt11_plugin_v2_layer_get_plugin_version,
            "PluginV2 version is too large for the managed buffer.");
        string pluginNamespace = GetPluginV2LayerString(
            line,
            layer,
            NativeMethodsTensorRt.jyppx_trt8_plugin_v2_layer_get_plugin_namespace,
            NativeMethodsTensorRt.jyppx_trt10_plugin_v2_layer_get_plugin_namespace,
            NativeMethodsTensorRt.jyppx_trt11_plugin_v2_layer_get_plugin_namespace,
            "PluginV2 namespace is too large for the managed buffer.");

        UIntPtr serializationSize;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_plugin_v2_layer_get_serialization_size(layer, out serializationSize),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_plugin_v2_layer_get_serialization_size(layer, out serializationSize),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_plugin_v2_layer_get_serialization_size(layer, out serializationSize),
            _ => throw UnsupportedPluginV2LayerMetadataLine()
        };
        NativeStatus.ThrowIfFailed(status);

        int packedTensorRtVersion;
        status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_plugin_v2_layer_get_tensor_rt_version(layer, out packedTensorRtVersion),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_plugin_v2_layer_get_tensor_rt_version(layer, out packedTensorRtVersion),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_plugin_v2_layer_get_tensor_rt_version(layer, out packedTensorRtVersion),
            _ => throw UnsupportedPluginV2LayerMetadataLine()
        };
        NativeStatus.ThrowIfFailed(status);

        int outputCount;
        status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_plugin_v2_layer_get_plugin_output_count(layer, out outputCount),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_plugin_v2_layer_get_plugin_output_count(layer, out outputCount),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_plugin_v2_layer_get_plugin_output_count(layer, out outputCount),
            _ => throw UnsupportedPluginV2LayerMetadataLine()
        };
        NativeStatus.ThrowIfFailed(status);

        int hasExt;
        int hasIoExt;
        int hasDynamicExt;
        status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_plugin_v2_layer_get_capability_presence(
                layer, out hasExt, out hasIoExt, out hasDynamicExt),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_plugin_v2_layer_get_capability_presence(
                layer, out hasExt, out hasIoExt, out hasDynamicExt),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_plugin_v2_layer_get_capability_presence(
                layer, out hasExt, out hasIoExt, out hasDynamicExt),
            _ => throw UnsupportedPluginV2LayerMetadataLine()
        };
        NativeStatus.ThrowIfFailed(status);

        return new TensorRtPluginV2LayerMetadata(
            line,
            pluginType,
            pluginVersion,
            pluginNamespace,
            serializationSize.ToUInt64(),
            packedTensorRtVersion,
            outputCount,
            hasExt != 0,
            hasIoExt != 0,
            hasDynamicExt != 0);
    }

    public static TensorRtDims GetPluginV2LegacyOutputDimensions(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer,
        int outputIndex)
    {
        NativeTensorRtDims dimensions;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_plugin_v2_layer_get_legacy_output_dimensions(layer, outputIndex, out dimensions),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_plugin_v2_layer_get_legacy_output_dimensions(layer, outputIndex, out dimensions),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_plugin_v2_layer_get_legacy_output_dimensions(layer, outputIndex, out dimensions),
            _ => throw UnsupportedPluginV2LayerMetadataLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(dimensions);
    }

    public static ulong GetPluginV2LegacyWorkspaceSize(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer,
        int maxBatchSize)
    {
        UIntPtr workspaceSize;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_plugin_v2_layer_get_legacy_workspace_size(layer, maxBatchSize, out workspaceSize),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_plugin_v2_layer_get_legacy_workspace_size(layer, maxBatchSize, out workspaceSize),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_plugin_v2_layer_get_legacy_workspace_size(layer, maxBatchSize, out workspaceSize),
            _ => throw UnsupportedPluginV2LayerMetadataLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return workspaceSize.ToUInt64();
    }

    public static bool SupportsPluginV2LegacyFormat(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer,
        TensorRtDataType dataType,
        TensorRtTensorFormat tensorFormat)
    {
        int supported;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_plugin_v2_layer_supports_legacy_format(
                layer, (int)dataType, (int)tensorFormat, out supported),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_plugin_v2_layer_supports_legacy_format(
                layer, (int)dataType, (int)tensorFormat, out supported),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_plugin_v2_layer_supports_legacy_format(
                layer, (int)dataType, (int)tensorFormat, out supported),
            _ => throw UnsupportedPluginV2LayerMetadataLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return supported != 0;
    }

    public static TensorRtDataType GetPluginV2OutputDataType(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer,
        int outputIndex)
    {
        int dataType;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_plugin_v2_layer_get_output_data_type(layer, outputIndex, out dataType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_plugin_v2_layer_get_output_data_type(layer, outputIndex, out dataType),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_plugin_v2_layer_get_output_data_type(layer, outputIndex, out dataType),
            _ => throw UnsupportedPluginV2LayerMetadataLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtDataType)dataType;
    }

    public static bool CanPluginV2BroadcastInputAcrossBatch(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer,
        int inputIndex)
    {
        EnsurePluginV2BroadcastLine(line);
        int canBroadcast;
        BridgeStatusCode status = line == TensorRtApiLine.TensorRt8
            ? NativeMethodsTensorRt.jyppx_trt8_plugin_v2_layer_can_broadcast_input_across_batch(layer, inputIndex, out canBroadcast)
            : NativeMethodsTensorRt.jyppx_trt10_plugin_v2_layer_can_broadcast_input_across_batch(layer, inputIndex, out canBroadcast);
        NativeStatus.ThrowIfFailed(status);
        return canBroadcast != 0;
    }

    public static bool IsPluginV2OutputBroadcastAcrossBatch(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer,
        int outputIndex,
        byte[] inputIsBroadcasted)
    {
        EnsurePluginV2BroadcastLine(line);
        int isBroadcast;
        UIntPtr inputCount = new UIntPtr((uint)inputIsBroadcasted.Length);
        BridgeStatusCode status = line == TensorRtApiLine.TensorRt8
            ? NativeMethodsTensorRt.jyppx_trt8_plugin_v2_layer_is_output_broadcast_across_batch(
                layer, outputIndex, inputIsBroadcasted, inputCount, out isBroadcast)
            : NativeMethodsTensorRt.jyppx_trt10_plugin_v2_layer_is_output_broadcast_across_batch(
                layer, outputIndex, inputIsBroadcasted, inputCount, out isBroadcast);
        NativeStatus.ThrowIfFailed(status);
        return isBroadcast != 0;
    }

    private static string GetPluginV2LayerString(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer,
        PluginV2LayerStringDelegate trt8,
        PluginV2LayerStringDelegate trt10,
        PluginV2LayerStringDelegate trt11,
        string bufferError)
    {
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt8 => trt8(layer, buffer, size, out required),
                TensorRtApiLine.TensorRt10 => trt10(layer, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => trt11(layer, buffer, size, out required),
                _ => throw UnsupportedPluginV2LayerMetadataLine()
            },
            bufferError);
    }

    private static BridgeProbeException UnsupportedPluginV2LayerMetadataLine()
    {
        return new BridgeProbeException(
            BridgeStatusCode.NotSupported,
            BridgeErrorCategory.TensorRt,
            "PluginV2 layer metadata is supported only for TensorRT 8, 10, and 11 adapters.");
    }

    private static void EnsurePluginV2BroadcastLine(TensorRtApiLine line)
    {
        if (line != TensorRtApiLine.TensorRt8 && line != TensorRtApiLine.TensorRt10)
        {
            throw new BridgeProbeException(
                BridgeStatusCode.NotSupported,
                BridgeErrorCategory.TensorRt,
                "PluginV2 implicit-batch broadcast queries are supported only for TensorRT 8 and 10 adapters.");
        }
    }

    private delegate BridgeStatusCode PluginV2LayerStringDelegate(
        SafeTensorRtObjectHandle layer,
        byte[] outputBuffer,
        UIntPtr outputBufferSize,
        out UIntPtr requiredSize);
}
