using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    private delegate BridgeStatusCode VersionedMetadataNativeCall(
        byte[] buffer,
        UIntPtr bufferSize,
        out UIntPtr requiredSize,
        out int major,
        out int minor,
        out int apiLanguage);

    public static TensorRtVersionedInterfaceMetadata GetRuntimeErrorRecorderVersionedMetadata(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle runtime)
    {
        return ReadVersionedMetadata(
            line,
            (byte[] buffer, UIntPtr size, out UIntPtr required, out int major, out int minor, out int apiLanguage) => line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_runtime_get_error_recorder_versioned_metadata(runtime, buffer, size, out required, out major, out minor, out apiLanguage),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_runtime_get_error_recorder_versioned_metadata(runtime, buffer, size, out required, out major, out minor, out apiLanguage),
                _ => throw UnsupportedOwnerScopedVersionedMetadataLine("Runtime error-recorder versioned metadata is exposed for TensorRT 10 and 11.")
            });
    }

    public static TensorRtVersionedInterfaceMetadata GetRefitterErrorRecorderVersionedMetadata(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle refitter)
    {
        return ReadVersionedMetadata(
            line,
            (byte[] buffer, UIntPtr size, out UIntPtr required, out int major, out int minor, out int apiLanguage) => line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_get_error_recorder_versioned_metadata(refitter, buffer, size, out required, out major, out minor, out apiLanguage),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_refitter_get_error_recorder_versioned_metadata(refitter, buffer, size, out required, out major, out minor, out apiLanguage),
                _ => throw UnsupportedOwnerScopedVersionedMetadataLine("Refitter error-recorder versioned metadata is exposed for TensorRT 10 and 11.")
            });
    }

    public static TensorRtVersionedInterfaceMetadata GetEngineErrorRecorderVersionedMetadata(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle engine)
    {
        return ReadVersionedMetadata(
            line,
            (byte[] buffer, UIntPtr size, out UIntPtr required, out int major, out int minor, out int apiLanguage) => line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_error_recorder_versioned_metadata(engine, buffer, size, out required, out major, out minor, out apiLanguage),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_error_recorder_versioned_metadata(engine, buffer, size, out required, out major, out minor, out apiLanguage),
                _ => throw UnsupportedOwnerScopedVersionedMetadataLine("Engine error-recorder versioned metadata is exposed for TensorRT 10 and 11.")
            });
    }

    public static TensorRtVersionedInterfaceMetadata GetExecutionContextErrorRecorderVersionedMetadata(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle context)
    {
        return ReadVersionedMetadata(
            line,
            (byte[] buffer, UIntPtr size, out UIntPtr required, out int major, out int minor, out int apiLanguage) => line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_get_error_recorder_versioned_metadata(context, buffer, size, out required, out major, out minor, out apiLanguage),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_get_error_recorder_versioned_metadata(context, buffer, size, out required, out major, out minor, out apiLanguage),
                _ => throw UnsupportedOwnerScopedVersionedMetadataLine("Execution-context error-recorder versioned metadata is exposed for TensorRT 10 and 11.")
            });
    }

    public static TensorRtVersionedInterfaceMetadata GetBuilderErrorRecorderVersionedMetadata(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle builder)
    {
        return ReadVersionedMetadata(
            line,
            (byte[] buffer, UIntPtr size, out UIntPtr required, out int major, out int minor, out int apiLanguage) => line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_get_error_recorder_versioned_metadata(builder, buffer, size, out required, out major, out minor, out apiLanguage),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_get_error_recorder_versioned_metadata(builder, buffer, size, out required, out major, out minor, out apiLanguage),
                _ => throw UnsupportedOwnerScopedVersionedMetadataLine("Builder error-recorder versioned metadata is exposed for TensorRT 10 and 11.")
            });
    }

    public static TensorRtVersionedInterfaceMetadata GetNetworkErrorRecorderVersionedMetadata(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network)
    {
        return ReadVersionedMetadata(
            line,
            (byte[] buffer, UIntPtr size, out UIntPtr required, out int major, out int minor, out int apiLanguage) => line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_get_error_recorder_versioned_metadata(network, buffer, size, out required, out major, out minor, out apiLanguage),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_get_error_recorder_versioned_metadata(network, buffer, size, out required, out major, out minor, out apiLanguage),
                _ => throw UnsupportedOwnerScopedVersionedMetadataLine("Network error-recorder versioned metadata is exposed for TensorRT 10 and 11.")
            });
    }

    public static TensorRtVersionedInterfaceMetadata GetEngineInspectorErrorRecorderVersionedMetadata(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle inspector)
    {
        return ReadVersionedMetadata(
            line,
            (byte[] buffer, UIntPtr size, out UIntPtr required, out int major, out int minor, out int apiLanguage) => line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_inspector_get_error_recorder_versioned_metadata(inspector, buffer, size, out required, out major, out minor, out apiLanguage),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_inspector_get_error_recorder_versioned_metadata(inspector, buffer, size, out required, out major, out minor, out apiLanguage),
                _ => throw UnsupportedOwnerScopedVersionedMetadataLine("Engine-inspector error-recorder versioned metadata is exposed for TensorRT 10 and 11.")
            });
    }

    public static TensorRtVersionedInterfaceMetadata GetBuilderConfigProgressMonitorVersionedMetadata(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle config)
    {
        return ReadVersionedMetadata(
            line,
            (byte[] buffer, UIntPtr size, out UIntPtr required, out int major, out int minor, out int apiLanguage) => line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_progress_monitor_versioned_metadata(config, buffer, size, out required, out major, out minor, out apiLanguage),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_progress_monitor_versioned_metadata(config, buffer, size, out required, out major, out minor, out apiLanguage),
                _ => throw UnsupportedOwnerScopedVersionedMetadataLine("Builder-config progress-monitor versioned metadata is exposed for TensorRT 10 and 11.")
            });
    }

    public static TensorRtVersionedInterfaceMetadata GetExecutionContextOutputAllocatorVersionedMetadata(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle context,
        string tensorName)
    {
        TensorRtInterfaceInfo interfaceInfo = GetExecutionContextOutputAllocatorInterfaceInfo(line, context, tensorName);
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        int apiLanguage = -1;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_get_output_allocator_api_language(context, tensorNameUtf8.Pointer, out apiLanguage),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_get_output_allocator_api_language(context, tensorNameUtf8.Pointer, out apiLanguage),
            _ => throw UnsupportedOwnerScopedVersionedMetadataLine("Output-allocator versioned metadata is exposed for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
        return new TensorRtVersionedInterfaceMetadata(line, interfaceInfo, ToTensorRtApiLanguage(apiLanguage));
    }

    public static TensorRtVersionedInterfaceMetadata GetExecutionContextTemporaryStorageAllocatorVersionedMetadata(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle context)
    {
        TensorRtInterfaceInfo interfaceInfo = GetExecutionContextTemporaryStorageAllocatorInterfaceInfo(line, context);
        int apiLanguage = -1;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_get_temporary_storage_allocator_api_language(context, out apiLanguage),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_get_temporary_storage_allocator_api_language(context, out apiLanguage),
            _ => throw UnsupportedOwnerScopedVersionedMetadataLine("Temporary-storage allocator versioned metadata is exposed for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
        return new TensorRtVersionedInterfaceMetadata(line, interfaceInfo, ToTensorRtApiLanguage(apiLanguage));
    }

    public static TensorRtVersionedInterfaceMetadata GetExecutionContextDebugListenerVersionedMetadata(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle context)
    {
        TensorRtInterfaceInfo interfaceInfo = GetExecutionContextDebugListenerInterfaceInfo(line, context);
        int apiLanguage = -1;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_get_debug_listener_api_language(context, out apiLanguage),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_get_debug_listener_api_language(context, out apiLanguage),
            _ => throw UnsupportedOwnerScopedVersionedMetadataLine("Debug-listener versioned metadata is exposed for TensorRT 10 and 11.")
        };
        NativeStatus.ThrowIfFailed(status);
        return new TensorRtVersionedInterfaceMetadata(line, interfaceInfo, ToTensorRtApiLanguage(apiLanguage));
    }

    private static TensorRtVersionedInterfaceMetadata ReadVersionedMetadata(
        TensorRtApiLine line,
        VersionedMetadataNativeCall nativeCall)
    {
        int major = 0;
        int minor = 0;
        int apiLanguage = -1;
        string kind = ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) =>
                nativeCall(buffer, size, out required, out major, out minor, out apiLanguage),
            "TensorRT versioned-interface kind is too large for the managed buffer.");
        return new TensorRtVersionedInterfaceMetadata(
            line,
            new TensorRtInterfaceInfo(kind, major, minor),
            ToTensorRtApiLanguage(apiLanguage));
    }

    private static BridgeProbeException UnsupportedOwnerScopedVersionedMetadataLine(string message)
    {
        return new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, message);
    }
}
