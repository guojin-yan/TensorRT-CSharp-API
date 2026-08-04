using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static void ResetBuilderConfig(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(ResetBuilderConfig));
        NativeStatus.ThrowIfFailed(NativeMethodsTensorRt.jyppx_trt11_builder_config_reset(config));
    }

    public static bool HasBuilderConfigTimingCache(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(HasBuilderConfigTimingCache));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_builder_config_has_timing_cache(config, out int hasCache);
        NativeStatus.ThrowIfFailed(status);
        return hasCache != 0;
    }

    public static bool CanBuilderConfigRunLayerOnDla(TensorRtApiLine line, SafeTensorRtObjectHandle config, SafeTensorRtObjectHandle layer)
    {
        int canRun;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_can_run_on_dla(config, layer, out canRun),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_can_run_on_dla(config, layer, out canRun),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_can_run_on_dla(config, layer, out canRun),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return canRun != 0;
    }

    public static void ClearBuilderConfigPluginsToSerialize(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_plugins_to_serialize(config, IntPtr.Zero, 0, out int _),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_clear_plugins_to_serialize(config),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_clear_plugins_to_serialize(config),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static int GetBuilderConfigPluginToSerializeCount(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_nb_plugins_to_serialize(config, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_nb_plugins_to_serialize(config, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_nb_plugins_to_serialize(config, out count),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static string GetBuilderConfigPluginToSerialize(TensorRtApiLine line, SafeTensorRtObjectHandle config, int index)
    {
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_plugin_to_serialize(config, index, buffer, size, out required),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_plugin_to_serialize(config, index, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_plugin_to_serialize(config, index, buffer, size, out required),
                _ => throw UnsupportedLine()
            },
            "Serialized plugin path is too large for the managed buffer.");
    }

    public static bool HasBuilderConfigProgressMonitor(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int hasMonitor;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_has_progress_monitor(config, out hasMonitor),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_has_progress_monitor(config, out hasMonitor),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{nameof(HasBuilderConfigProgressMonitor)} is available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return hasMonitor != 0;
    }

    public static void ClearBuilderConfigProgressMonitor(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_clear_progress_monitor(config),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_clear_progress_monitor(config),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{nameof(ClearBuilderConfigProgressMonitor)} is available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetBuilderConfigProgressMonitor(TensorRtApiLine line, SafeTensorRtObjectHandle config, SafeTensorRtObjectHandle monitor)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_progress_monitor(config, monitor),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_progress_monitor(config, monitor),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{nameof(SetBuilderConfigProgressMonitor)} is available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

}
