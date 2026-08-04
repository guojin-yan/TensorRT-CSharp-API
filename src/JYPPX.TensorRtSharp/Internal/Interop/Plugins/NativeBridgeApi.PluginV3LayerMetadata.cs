using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static TensorRtPluginV3LayerMetadata GetPluginV3LayerMetadata(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer)
    {
        EnsurePluginV3LayerMetadataLine(line);

        TensorRtPluginV3InterfaceMetadata pluginInterface = GetPluginV3LayerInterfaceInfo(
            line,
            layer,
            NativeMethodsTensorRt.jyppx_trt10_plugin_v3_layer_get_plugin_interface_info,
            NativeMethodsTensorRt.jyppx_trt11_plugin_v3_layer_get_plugin_interface_info,
            "PluginV3 interface kind is too large for the managed buffer.");

        BridgeStatusCode status;
        int hasCore;
        int hasBuild;
        int hasRuntime;
        if (line == TensorRtApiLine.TensorRt10)
        {
            status = NativeMethodsTensorRt.jyppx_trt10_plugin_v3_layer_get_capability_presence(
                layer, out hasCore, out hasBuild, out hasRuntime);
        }
        else
        {
            status = NativeMethodsTensorRt.jyppx_trt11_plugin_v3_layer_get_capability_presence(
                layer, out hasCore, out hasBuild, out hasRuntime);
        }
        NativeStatus.ThrowIfFailed(status);

        string pluginName = GetPluginV3LayerString(
            line,
            layer,
            NativeMethodsTensorRt.jyppx_trt10_plugin_v3_layer_get_core_plugin_name,
            NativeMethodsTensorRt.jyppx_trt11_plugin_v3_layer_get_core_plugin_name,
            "PluginV3 core plugin name is too large for the managed buffer.");
        string pluginVersion = GetPluginV3LayerString(
            line,
            layer,
            NativeMethodsTensorRt.jyppx_trt10_plugin_v3_layer_get_core_plugin_version,
            NativeMethodsTensorRt.jyppx_trt11_plugin_v3_layer_get_core_plugin_version,
            "PluginV3 core plugin version is too large for the managed buffer.");
        string pluginNamespace = GetPluginV3LayerString(
            line,
            layer,
            NativeMethodsTensorRt.jyppx_trt10_plugin_v3_layer_get_core_plugin_namespace,
            NativeMethodsTensorRt.jyppx_trt11_plugin_v3_layer_get_core_plugin_namespace,
            "PluginV3 core plugin namespace is too large for the managed buffer.");
        TensorRtPluginV3InterfaceMetadata coreInterface = GetPluginV3LayerInterfaceInfo(
            line,
            layer,
            NativeMethodsTensorRt.jyppx_trt10_plugin_v3_layer_get_core_interface_info,
            NativeMethodsTensorRt.jyppx_trt11_plugin_v3_layer_get_core_interface_info,
            "PluginV3 core interface kind is too large for the managed buffer.");

        TensorRtPluginV3CoreMetadata core = new TensorRtPluginV3CoreMetadata(
            pluginName,
            pluginVersion,
            pluginNamespace,
            coreInterface);

        TensorRtPluginV3BuildMetadata? build = null;
        if (hasBuild != 0)
        {
            TensorRtPluginV3InterfaceMetadata buildInterface = GetPluginV3LayerInterfaceInfo(
                line,
                layer,
                NativeMethodsTensorRt.jyppx_trt10_plugin_v3_layer_get_build_interface_info,
                NativeMethodsTensorRt.jyppx_trt11_plugin_v3_layer_get_build_interface_info,
                "PluginV3 build interface kind is too large for the managed buffer.");
            int outputCount = GetPluginV3LayerScalar(
                line,
                layer,
                NativeMethodsTensorRt.jyppx_trt10_plugin_v3_layer_get_build_nb_outputs,
                NativeMethodsTensorRt.jyppx_trt11_plugin_v3_layer_get_build_nb_outputs);
            int tacticCount = GetPluginV3LayerScalar(
                line,
                layer,
                NativeMethodsTensorRt.jyppx_trt10_plugin_v3_layer_get_build_nb_tactics,
                NativeMethodsTensorRt.jyppx_trt11_plugin_v3_layer_get_build_nb_tactics);
            int formatCombinationLimit = GetPluginV3LayerScalar(
                line,
                layer,
                NativeMethodsTensorRt.jyppx_trt10_plugin_v3_layer_get_build_format_combination_limit,
                NativeMethodsTensorRt.jyppx_trt11_plugin_v3_layer_get_build_format_combination_limit);
            string timingCacheId = GetPluginV3LayerString(
                line,
                layer,
                NativeMethodsTensorRt.jyppx_trt10_plugin_v3_layer_get_build_timing_cache_id,
                NativeMethodsTensorRt.jyppx_trt11_plugin_v3_layer_get_build_timing_cache_id,
                "PluginV3 timing-cache ID is too large for the managed buffer.");
            string metadataString = GetPluginV3LayerString(
                line,
                layer,
                NativeMethodsTensorRt.jyppx_trt10_plugin_v3_layer_get_build_metadata_string,
                NativeMethodsTensorRt.jyppx_trt11_plugin_v3_layer_get_build_metadata_string,
                "PluginV3 build metadata string is too large for the managed buffer.");

            build = new TensorRtPluginV3BuildMetadata(
                buildInterface,
                outputCount,
                tacticCount,
                formatCombinationLimit,
                timingCacheId,
                metadataString);
        }

        TensorRtPluginV3RuntimeMetadata? runtime = null;
        if (hasRuntime != 0)
        {
            TensorRtPluginV3InterfaceMetadata runtimeInterface = GetPluginV3LayerInterfaceInfo(
                line,
                layer,
                NativeMethodsTensorRt.jyppx_trt10_plugin_v3_layer_get_runtime_interface_info,
                NativeMethodsTensorRt.jyppx_trt11_plugin_v3_layer_get_runtime_interface_info,
                "PluginV3 runtime interface kind is too large for the managed buffer.");
            runtime = new TensorRtPluginV3RuntimeMetadata(runtimeInterface);
        }

        return new TensorRtPluginV3LayerMetadata(
            line,
            pluginInterface,
            hasCore != 0,
            core,
            build,
            runtime);
    }

    private static TensorRtPluginV3InterfaceMetadata GetPluginV3LayerInterfaceInfo(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer,
        PluginV3LayerInterfaceInfoDelegate trt10,
        PluginV3LayerInterfaceInfoDelegate trt11,
        string bufferError)
    {
        int major = 0;
        int minor = 0;
        int apiLanguage = (int)TensorRtApiLanguage.Unknown;
        string kind = ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt10 => trt10(layer, buffer, size, out required, out major, out minor, out apiLanguage),
                TensorRtApiLine.TensorRt11 => trt11(layer, buffer, size, out required, out major, out minor, out apiLanguage),
                _ => throw UnsupportedPluginV3LayerMetadataLine()
            },
            bufferError);

        return new TensorRtPluginV3InterfaceMetadata(
            new TensorRtInterfaceInfo(kind, major, minor),
            ToTensorRtApiLanguage(apiLanguage));
    }

    private static string GetPluginV3LayerString(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer,
        PluginV3LayerStringDelegate trt10,
        PluginV3LayerStringDelegate trt11,
        string bufferError)
    {
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt10 => trt10(layer, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => trt11(layer, buffer, size, out required),
                _ => throw UnsupportedPluginV3LayerMetadataLine()
            },
            bufferError);
    }

    private static int GetPluginV3LayerScalar(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer,
        PluginV3LayerScalarDelegate trt10,
        PluginV3LayerScalarDelegate trt11)
    {
        BridgeStatusCode status;
        int value;
        if (line == TensorRtApiLine.TensorRt10)
        {
            status = trt10(layer, out value);
        }
        else
        {
            status = trt11(layer, out value);
        }
        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    private static void EnsurePluginV3LayerMetadataLine(TensorRtApiLine line)
    {
        if (line != TensorRtApiLine.TensorRt10 && line != TensorRtApiLine.TensorRt11)
        {
            throw UnsupportedPluginV3LayerMetadataLine();
        }
    }

    private static BridgeProbeException UnsupportedPluginV3LayerMetadataLine()
    {
        return new BridgeProbeException(
            BridgeStatusCode.NotSupported,
            BridgeErrorCategory.TensorRt,
            "PluginV3 layer metadata is supported only for TensorRT 10 and 11 adapters.");
    }

    private delegate BridgeStatusCode PluginV3LayerInterfaceInfoDelegate(
        SafeTensorRtObjectHandle layer,
        byte[] outputBuffer,
        UIntPtr outputBufferSize,
        out UIntPtr requiredSize,
        out int major,
        out int minor,
        out int apiLanguage);

    private delegate BridgeStatusCode PluginV3LayerStringDelegate(
        SafeTensorRtObjectHandle layer,
        byte[] outputBuffer,
        UIntPtr outputBufferSize,
        out UIntPtr requiredSize);

    private delegate BridgeStatusCode PluginV3LayerScalarDelegate(
        SafeTensorRtObjectHandle layer,
        out int value);
}
