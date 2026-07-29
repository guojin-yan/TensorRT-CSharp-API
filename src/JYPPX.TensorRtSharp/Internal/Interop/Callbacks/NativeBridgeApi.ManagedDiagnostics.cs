using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle CreateLogger(TensorRtApiLine line)
    {
        return CreateLoggerCore(GetBindings(line));
    }

    public static SafeTensorRtObjectHandle CreateLogger(
        TensorRtApiLine line,
        TensorRtLoggerCallback callback,
        IntPtr userState,
        TensorRtLogSeverity minimumSeverity)
    {
        SafeTensorRtObjectHandle logger;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_logger_create_with_callback(callback, userState, (int)minimumSeverity, out logger),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_logger_create_with_callback(callback, userState, (int)minimumSeverity, out logger),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_logger_create_with_callback(callback, userState, (int)minimumSeverity, out logger),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return logger;
    }

    public static bool EmitLoggerDiagnostic(TensorRtApiLine line, SafeTensorRtObjectHandle logger, TensorRtLogSeverity severity, string message)
    {
        using Utf8Interop.Utf8StringScope messageUtf8 = Utf8Interop.ToNativeString(message);
        int callbackFailed;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_logger_emit_diagnostic(logger, (int)severity, messageUtf8.Pointer, out callbackFailed),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_logger_emit_diagnostic(logger, (int)severity, messageUtf8.Pointer, out callbackFailed),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_logger_emit_diagnostic(logger, (int)severity, messageUtf8.Pointer, out callbackFailed),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return callbackFailed == 0;
    }

    public static SafeTensorRtObjectHandle CreateProfiler(
        TensorRtApiLine line,
        TensorRtProfilerCallback callback,
        IntPtr userState)
    {
        SafeTensorRtObjectHandle profiler;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_profiler_create_with_callback(callback, userState, out profiler),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_profiler_create_with_callback(callback, userState, out profiler),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_profiler_create_with_callback(callback, userState, out profiler),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return profiler;
    }

    public static bool EmitProfilerDiagnostic(TensorRtApiLine line, SafeTensorRtObjectHandle profiler, string layerName, float milliseconds)
    {
        using Utf8Interop.Utf8StringScope layerNameUtf8 = Utf8Interop.ToNativeString(layerName);
        int callbackFailed;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_profiler_emit_diagnostic(profiler, layerNameUtf8.Pointer, milliseconds, out callbackFailed),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_profiler_emit_diagnostic(profiler, layerNameUtf8.Pointer, milliseconds, out callbackFailed),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_profiler_emit_diagnostic(profiler, layerNameUtf8.Pointer, milliseconds, out callbackFailed),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return callbackFailed == 0;
    }

    public static SafeTensorRtObjectHandle CreateProgressMonitor(
        TensorRtApiLine line,
        TensorRtProgressMonitorCallback callback,
        IntPtr userState)
    {
        SafeTensorRtObjectHandle monitor;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_progress_monitor_create_with_callback(callback, userState, out monitor),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_progress_monitor_create_with_callback(callback, userState, out monitor),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "TensorRT progress monitor callbacks are available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return monitor;
    }

    public static TensorRtProgressMonitorDiagnosticResult EmitProgressMonitorDiagnostic(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle monitor,
        TensorRtProgressMonitorEventKind kind,
        string phaseName,
        string? parentPhase,
        int step,
        int stepCount)
    {
        using Utf8Interop.Utf8StringScope phaseNameUtf8 = Utf8Interop.ToNativeString(phaseName);
        using Utf8Interop.Utf8StringScope parentPhaseUtf8 = Utf8Interop.ToNativeString(parentPhase);

        int shouldContinue;
        int callbackFailed;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_progress_monitor_emit_diagnostic(monitor, (int)kind, phaseNameUtf8.Pointer, parentPhaseUtf8.Pointer, step, stepCount, out shouldContinue, out callbackFailed),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_progress_monitor_emit_diagnostic(monitor, (int)kind, phaseNameUtf8.Pointer, parentPhaseUtf8.Pointer, step, stepCount, out shouldContinue, out callbackFailed),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "TensorRT progress monitor callbacks are available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return new TensorRtProgressMonitorDiagnosticResult(shouldContinue != 0, callbackFailed == 0);
    }

}
