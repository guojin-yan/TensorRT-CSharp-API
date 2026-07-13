using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static TensorRtInterfaceInfo GetLoggerInterfaceInfo(TensorRtApiLine line, SafeTensorRtObjectHandle logger)
    {
        int major = 0;
        int minor = 0;
        string kind = ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) =>
            {
                BridgeStatusCode status = line switch
                {
                    TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_logger_get_interface_info(logger, buffer, size, out required, out major, out minor),
                    _ => throw UnsupportedCallbackInterfaceInfoLine("Logger interface info is exposed by TensorRT 11 only.")
                };
                return status;
            },
            "TensorRT logger interface kind is too large for the managed buffer.");

        return new TensorRtInterfaceInfo(kind, major, minor);
    }

    public static TensorRtApiLanguage GetLoggerApiLanguage(TensorRtApiLine line, SafeTensorRtObjectHandle logger)
    {
        int apiLanguage = (int)TensorRtApiLanguage.Unknown;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_logger_get_api_language(logger, out apiLanguage),
            _ => throw UnsupportedCallbackInterfaceInfoLine("Logger API language metadata is exposed by TensorRT 11 only.")
        };

        NativeStatus.ThrowIfFailed(status);
        return ToTensorRtApiLanguage(apiLanguage);
    }

    public static TensorRtInterfaceInfo GetProfilerInterfaceInfo(TensorRtApiLine line, SafeTensorRtObjectHandle profiler)
    {
        int major = 0;
        int minor = 0;
        string kind = ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) =>
            {
                BridgeStatusCode status = line switch
                {
                    TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_profiler_get_interface_info(profiler, buffer, size, out required, out major, out minor),
                    _ => throw UnsupportedCallbackInterfaceInfoLine("Profiler interface info is exposed by TensorRT 11 only.")
                };
                return status;
            },
            "TensorRT profiler interface kind is too large for the managed buffer.");

        return new TensorRtInterfaceInfo(kind, major, minor);
    }

    public static TensorRtApiLanguage GetProfilerApiLanguage(TensorRtApiLine line, SafeTensorRtObjectHandle profiler)
    {
        int apiLanguage = (int)TensorRtApiLanguage.Unknown;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_profiler_get_api_language(profiler, out apiLanguage),
            _ => throw UnsupportedCallbackInterfaceInfoLine("Profiler API language metadata is exposed by TensorRT 11 only.")
        };

        NativeStatus.ThrowIfFailed(status);
        return ToTensorRtApiLanguage(apiLanguage);
    }

    public static TensorRtInterfaceInfo GetProgressMonitorInterfaceInfo(TensorRtApiLine line, SafeTensorRtObjectHandle monitor)
    {
        int major = 0;
        int minor = 0;
        string kind = ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) =>
            {
                BridgeStatusCode status = line switch
                {
                    TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_progress_monitor_get_interface_info(monitor, buffer, size, out required, out major, out minor),
                    TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_progress_monitor_get_interface_info(monitor, buffer, size, out required, out major, out minor),
                    _ => throw UnsupportedCallbackInterfaceInfoLine("Progress monitor interface info is exposed by TensorRT 10 and 11.")
                };
                return status;
            },
            "TensorRT progress monitor interface kind is too large for the managed buffer.");

        return new TensorRtInterfaceInfo(kind, major, minor);
    }

    public static TensorRtApiLanguage GetProgressMonitorApiLanguage(TensorRtApiLine line, SafeTensorRtObjectHandle monitor)
    {
        int apiLanguage = (int)TensorRtApiLanguage.Unknown;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_progress_monitor_get_api_language(monitor, out apiLanguage),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_progress_monitor_get_api_language(monitor, out apiLanguage),
            _ => throw UnsupportedCallbackInterfaceInfoLine("Progress monitor API language metadata is exposed by TensorRT 10 and 11.")
        };

        NativeStatus.ThrowIfFailed(status);
        return ToTensorRtApiLanguage(apiLanguage);
    }

    public static TensorRtInterfaceInfo GetExecutionContextOutputAllocatorInterfaceInfo(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);

        int major = 0;
        int minor = 0;
        string kind = ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) =>
            {
                BridgeStatusCode status = line switch
                {
                    TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_get_output_allocator_interface_info(context, tensorNameUtf8.Pointer, buffer, size, out required, out major, out minor),
                    TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_get_output_allocator_interface_info(context, tensorNameUtf8.Pointer, buffer, size, out required, out major, out minor),
                    _ => throw UnsupportedCallbackInterfaceInfoLine("Output allocator interface info is exposed by this bridge for TensorRT 10 and 11 execution contexts.")
                };
                return status;
            },
            "TensorRT output allocator interface kind is too large for the managed buffer.");

        return new TensorRtInterfaceInfo(kind, major, minor);
    }

    public static TensorRtInterfaceInfo GetExecutionContextTemporaryStorageAllocatorInterfaceInfo(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        int major = 0;
        int minor = 0;
        string kind = ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) =>
            {
                BridgeStatusCode status = line switch
                {
                    TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_get_temporary_storage_allocator_interface_info(context, buffer, size, out required, out major, out minor),
                    TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_get_temporary_storage_allocator_interface_info(context, buffer, size, out required, out major, out minor),
                    _ => throw UnsupportedCallbackInterfaceInfoLine("Temporary-storage allocator interface info is exposed by this bridge for TensorRT 10 and 11 execution contexts.")
                };
                return status;
            },
            "TensorRT temporary-storage allocator interface kind is too large for the managed buffer.");

        return new TensorRtInterfaceInfo(kind, major, minor);
    }

    public static TensorRtInterfaceInfo GetExecutionContextDebugListenerInterfaceInfo(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        int major = 0;
        int minor = 0;
        string kind = ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) =>
            {
                BridgeStatusCode status = line switch
                {
                    TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_get_debug_listener_interface_info(context, buffer, size, out required, out major, out minor),
                    TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_get_debug_listener_interface_info(context, buffer, size, out required, out major, out minor),
                    _ => throw UnsupportedCallbackInterfaceInfoLine("Debug listener interface info is exposed by this bridge for TensorRT 10 and 11 execution contexts.")
                };
                return status;
            },
            "TensorRT debug listener interface kind is too large for the managed buffer.");

        return new TensorRtInterfaceInfo(kind, major, minor);
    }

    private static BridgeProbeException UnsupportedCallbackInterfaceInfoLine(string message)
    {
        return new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, message);
    }

    private static TensorRtApiLanguage ToTensorRtApiLanguage(int apiLanguage)
    {
        return apiLanguage switch
        {
            0 => TensorRtApiLanguage.Cpp,
            1 => TensorRtApiLanguage.Python,
            _ => TensorRtApiLanguage.Unknown
        };
    }
}
