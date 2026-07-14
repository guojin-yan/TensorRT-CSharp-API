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

        return new TensorRtPluginV2LayerMetadata(
            line,
            pluginType,
            pluginVersion,
            pluginNamespace,
            serializationSize.ToUInt64(),
            packedTensorRtVersion);
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

    private delegate BridgeStatusCode PluginV2LayerStringDelegate(
        SafeTensorRtObjectHandle layer,
        byte[] outputBuffer,
        UIntPtr outputBufferSize,
        out UIntPtr requiredSize);
}
