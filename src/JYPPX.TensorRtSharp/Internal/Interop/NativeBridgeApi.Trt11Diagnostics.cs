using System;
using JYPPX.Shared.Interop;
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

    public static bool MarkNetworkDebugTensor(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle tensor)
    {
        int marked;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_mark_debug(network, tensor, out marked),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_mark_debug(network, tensor, out marked),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{nameof(MarkNetworkDebugTensor)} is available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return marked != 0;
    }

    public static bool UnmarkNetworkDebugTensor(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle tensor)
    {
        int unmarked;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_unmark_debug(network, tensor, out unmarked),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_unmark_debug(network, tensor, out unmarked),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{nameof(UnmarkNetworkDebugTensor)} is available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return unmarked != 0;
    }

    public static bool IsNetworkDebugTensor(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle tensor)
    {
        int isDebug;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_is_debug_tensor(network, tensor, out isDebug),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_is_debug_tensor(network, tensor, out isDebug),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{nameof(IsNetworkDebugTensor)} is available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return isDebug != 0;
    }

    public static bool MarkNetworkUnfusedTensorsAsDebugTensors(TensorRtApiLine line, SafeTensorRtObjectHandle network)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(MarkNetworkUnfusedTensorsAsDebugTensors));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_network_mark_unfused_tensors_as_debug_tensors(network, out int marked);
        NativeStatus.ThrowIfFailed(status);
        return marked != 0;
    }

    public static bool UnmarkNetworkUnfusedTensorsAsDebugTensors(TensorRtApiLine line, SafeTensorRtObjectHandle network)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(UnmarkNetworkUnfusedTensorsAsDebugTensors));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_network_unmark_unfused_tensors_as_debug_tensors(network, out int unmarked);
        NativeStatus.ThrowIfFailed(status);
        return unmarked != 0;
    }

    public static bool MarkNetworkOutputForShapes(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle tensor)
    {
        int marked;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_mark_output_for_shapes(network, tensor, out marked),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_mark_output_for_shapes(network, tensor, out marked),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_mark_output_for_shapes(network, tensor, out marked),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return marked != 0;
    }

    public static bool UnmarkNetworkOutputForShapes(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle tensor)
    {
        int unmarked;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_unmark_output_for_shapes(network, tensor, out unmarked),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_unmark_output_for_shapes(network, tensor, out unmarked),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_unmark_output_for_shapes(network, tensor, out unmarked),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return unmarked != 0;
    }

    public static string GetEngineInspectorLayerInformation(TensorRtApiLine line, SafeTensorRtObjectHandle inspector, int layerIndex, TensorRtLayerInformationFormat format)
    {
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_inspector_get_layer_information(inspector, layerIndex, (int)format, buffer, size, out required),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_inspector_get_layer_information(inspector, layerIndex, (int)format, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_inspector_get_layer_information(inspector, layerIndex, (int)format, buffer, size, out required),
                _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
            },
            "Engine inspector layer information is too large for the managed buffer.");
    }

    public static bool HasEngineInspectorExecutionContext(TensorRtApiLine line, SafeTensorRtObjectHandle inspector)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(HasEngineInspectorExecutionContext));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_engine_inspector_has_execution_context(inspector, out int hasContext);
        NativeStatus.ThrowIfFailed(status);
        return hasContext != 0;
    }

    public static void ClearEngineInspectorExecutionContext(TensorRtApiLine line, SafeTensorRtObjectHandle inspector)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(ClearEngineInspectorExecutionContext));
        NativeStatus.ThrowIfFailed(NativeMethodsTensorRt.jyppx_trt11_engine_inspector_clear_execution_context(inspector));
    }

    public static bool HasEngineInspectorErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle inspector)
    {
        int hasRecorder = 0;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_inspector_has_error_recorder(inspector, out hasRecorder),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_inspector_has_error_recorder(inspector, out hasRecorder),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_inspector_has_error_recorder(inspector, out hasRecorder),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return hasRecorder != 0;
    }

    public static ulong GetExecutionContextTensorAddressValue(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetExecutionContextTensorAddressValue));
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_get_tensor_address_value(context, tensorNameUtf8.Pointer, out UIntPtr address);
        NativeStatus.ThrowIfFailed(status);
        return address.ToUInt64();
    }

    public static ulong GetExecutionContextOutputTensorAddressValue(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetExecutionContextOutputTensorAddressValue));
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_get_output_tensor_address_value(context, tensorNameUtf8.Pointer, out UIntPtr address);
        NativeStatus.ThrowIfFailed(status);
        return address.ToUInt64();
    }

    public static bool ClearExecutionContextOutputAllocator(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        int cleared = 0;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_clear_output_allocator(context, tensorNameUtf8.Pointer, out cleared),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_clear_output_allocator(context, tensorNameUtf8.Pointer, out cleared),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_clear_output_allocator(context, tensorNameUtf8.Pointer, out cleared),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return cleared != 0;
    }

    public static bool ClearExecutionContextTemporaryStorageAllocator(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        int cleared = 0;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_clear_temporary_storage_allocator(context, out cleared),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_clear_temporary_storage_allocator(context, out cleared),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_clear_temporary_storage_allocator(context, out cleared),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return cleared != 0;
    }

    public static bool ClearExecutionContextDebugListener(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        int cleared;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_clear_debug_listener(context, out cleared),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_clear_debug_listener(context, out cleared),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{nameof(ClearExecutionContextDebugListener)} is available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return cleared != 0;
    }

    public static bool HasExecutionContextDebugListener(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        int hasListener;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_has_debug_listener(context, out hasListener),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_has_debug_listener(context, out hasListener),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{nameof(HasExecutionContextDebugListener)} is available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return hasListener != 0;
    }

    public static void ClearExecutionContextProfiler(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_clear_profiler(context),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_clear_profiler(context),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_clear_profiler(context),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static bool HasExecutionContextProfiler(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        int hasProfiler;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_has_profiler(context, out hasProfiler),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_has_profiler(context, out hasProfiler),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_has_profiler(context, out hasProfiler),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return hasProfiler != 0;
    }

    public static void SetExecutionContextProfiler(TensorRtApiLine line, SafeTensorRtObjectHandle context, SafeTensorRtObjectHandle profiler)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_set_profiler(context, profiler),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_set_profiler(context, profiler),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_set_profiler(context, profiler),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static bool HasExecutionContextRuntimeConfig(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(HasExecutionContextRuntimeConfig));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_has_runtime_config(context, out int hasRuntimeConfig);
        NativeStatus.ThrowIfFailed(status);
        return hasRuntimeConfig != 0;
    }

    public static bool SetExecutionContextNvtxVerbosity(TensorRtApiLine line, SafeTensorRtObjectHandle context, TensorRtProfilingVerbosity verbosity)
    {
        int set = 0;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_set_nvtx_verbosity(context, (int)verbosity, out set),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_set_nvtx_verbosity(context, (int)verbosity, out set),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_set_nvtx_verbosity(context, (int)verbosity, out set),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return set != 0;
    }

    public static TensorRtProfilingVerbosity GetExecutionContextNvtxVerbosity(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        int verbosity = 0;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_get_nvtx_verbosity(context, out verbosity),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_get_nvtx_verbosity(context, out verbosity),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_get_nvtx_verbosity(context, out verbosity),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtProfilingVerbosity)verbosity;
    }

    public static void ClearExecutionContextAuxStreams(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(ClearExecutionContextAuxStreams));
        NativeStatus.ThrowIfFailed(NativeMethodsTensorRt.jyppx_trt11_execution_context_clear_aux_streams(context));
    }

    public static bool SetExecutionContextUnfusedTensorsDebugState(TensorRtApiLine line, SafeTensorRtObjectHandle context, bool enabled)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetExecutionContextUnfusedTensorsDebugState));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_set_unfused_tensors_debug_state(context, enabled ? 1 : 0, out int set);
        NativeStatus.ThrowIfFailed(status);
        return set != 0;
    }

    public static bool GetExecutionContextUnfusedTensorsDebugState(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetExecutionContextUnfusedTensorsDebugState));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_get_unfused_tensors_debug_state(context, out int debugState);
        NativeStatus.ThrowIfFailed(status);
        return debugState != 0;
    }
}
