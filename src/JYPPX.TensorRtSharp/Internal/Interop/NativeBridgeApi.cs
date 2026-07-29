using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    private delegate BridgeStatusCode QueryAdapterInfoDelegate(out NativeTensorRtAdapterInfo outInfo);
    private delegate BridgeStatusCode LoggerCreateDelegate(out SafeTensorRtObjectHandle outLogger);
    public static NativeTensorRtAdapterInfo GetAdapterInfo(TensorRtApiLine line)
    {
        return GetAdapterInfoCore(GetBindings(line));
    }

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

    public static SafeTensorRtObjectHandle CreateRuntime(TensorRtApiLine line, SafeTensorRtObjectHandle logger)
    {
        TensorRtLineBindings bindings = GetBindings(line);
        BridgeStatusCode status = bindings.RuntimeCreate(logger, out SafeTensorRtObjectHandle runtime);
        NativeStatus.ThrowIfFailed(status);
        return runtime;
    }

    public static TensorRtRuntimeCreateDiagnosticSnapshot GetRuntimeCreateDiagnostic(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle logger)
    {
        if (line != TensorRtApiLine.TensorRt11)
        {
            return new TensorRtRuntimeCreateDiagnosticSnapshot(
                line,
                diagnosticAvailable: false,
                attempted: false,
                loggerHandlePresent: logger != null && !logger.IsInvalid,
                loggerPayloadPresent: false,
                createInferRuntimeReturnedNonNull: false,
                createInferRuntimeReturnedNull: false,
                lastStatus: BridgeStatusCode.NotSupported,
                tensorRtAvailable: false,
                expectedMajor: 11,
                bridgeBuiltMajor: 0,
                detectedVersion: string.Empty,
                loggerCallbackAvailable: false,
                loggerMessageCount: 0,
                lastLoggerSeverity: 0,
                lastLoggerMessage: string.Empty,
                createRuntimePhase: "not-supported",
                nativeDetail: "TRT11 runtime create diagnostic is not implemented for this TensorRT API line.",
                diagnostic: "TRT11 runtime create diagnostic is only available for the TensorRT 11 adapter.");
        }

        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_runtime_create_diagnostic(logger, out NativeTensorRtRuntimeCreateDiagnosticInfo info);
        NativeStatus.ThrowIfFailed(status);
        return BridgeInfoMapper.ToManaged(info);
    }

    public static SafeTensorRtObjectHandle CreateBuilder(TensorRtApiLine line, SafeTensorRtObjectHandle logger)
    {
        TensorRtLineBindings bindings = GetBindings(line);
        BridgeStatusCode status = bindings.BuilderCreate(logger, out SafeTensorRtObjectHandle builder);
        NativeStatus.ThrowIfFailed(status);
        return builder;
    }

    public static bool BuilderPlatformHasFastFp16(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        int supported;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_platform_has_fast_fp16(builder, out supported),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_platform_has_fast_fp16(builder, out supported),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_platform_has_fast_fp16(builder, out supported),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return supported != 0;
    }

    public static bool BuilderPlatformHasFastInt8(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        int supported;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_platform_has_fast_int8(builder, out supported),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_platform_has_fast_int8(builder, out supported),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_platform_has_fast_int8(builder, out supported),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return supported != 0;
    }

    public static bool BuilderPlatformHasTf32(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        int supported;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_platform_has_tf32(builder, out supported),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_platform_has_tf32(builder, out supported),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_platform_has_tf32(builder, out supported),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return supported != 0;
    }

    public static int GetBuilderDlaCoreCount(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_get_dla_core_count(builder, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_get_dla_core_count(builder, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_get_dla_core_count(builder, out count),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static SafeTensorRtObjectHandle CreateBuilderConfig(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        TensorRtLineBindings bindings = GetBindings(line);
        BridgeStatusCode status = bindings.ConfigCreate(builder, out SafeTensorRtObjectHandle config);
        NativeStatus.ThrowIfFailed(status);
        return config;
    }

    public static SafeTensorRtObjectHandle CreateNetwork(TensorRtApiLine line, SafeTensorRtObjectHandle builder, uint creationFlags)
    {
        TensorRtLineBindings bindings = GetBindings(line);
        BridgeStatusCode status = bindings.NetworkCreate(builder, creationFlags, out SafeTensorRtObjectHandle network);
        NativeStatus.ThrowIfFailed(status);
        return network;
    }

    public static SafeTensorRtObjectHandle BuildSerializedNetwork(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle builder,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle config)
    {
        TensorRtLineBindings bindings = GetBindings(line);
        BridgeStatusCode status = bindings.SerializedBuild(builder, network, config, out SafeTensorRtObjectHandle hostMemory);
        NativeStatus.ThrowIfFailed(status);
        return hostMemory;
    }

    public static SafeTensorRtObjectHandle DeserializeHostMemory(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle runtime,
        SafeTensorRtObjectHandle hostMemory)
    {
        TensorRtLineBindings bindings = GetBindings(line);
        BridgeStatusCode status = bindings.DeserializeHostMemory(runtime, hostMemory, out SafeTensorRtObjectHandle engine);
        NativeStatus.ThrowIfFailed(status);
        return engine;
    }

    public static SafeTensorRtObjectHandle DeserializeEngineData(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle runtime,
        byte[] engineData)
    {
        if (engineData == null)
        {
            throw new ArgumentNullException(nameof(engineData));
        }

        if (engineData.Length == 0)
        {
            throw new ArgumentException("Serialized engine data must not be empty.", nameof(engineData));
        }

        GCHandle pinned = GCHandle.Alloc(engineData, GCHandleType.Pinned);
        try
        {
            SafeTensorRtObjectHandle engine;
            BridgeStatusCode status;
            switch (line)
            {
                case TensorRtApiLine.TensorRt8:
                    status = NativeMethodsTensorRt.jyppx_trt8_runtime_deserialize_engine(runtime, pinned.AddrOfPinnedObject(), (UIntPtr)engineData.Length, out engine);
                    break;
                case TensorRtApiLine.TensorRt10:
                    status = NativeMethodsTensorRt.jyppx_trt10_runtime_deserialize_engine(runtime, pinned.AddrOfPinnedObject(), (UIntPtr)engineData.Length, out engine);
                    break;
                case TensorRtApiLine.TensorRt11:
                    status = NativeMethodsTensorRt.jyppx_trt11_runtime_deserialize_engine(runtime, pinned.AddrOfPinnedObject(), (UIntPtr)engineData.Length, out engine);
                    break;
                default:
                    throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.");
            }

            NativeStatus.ThrowIfFailed(status);
            return engine;
        }
        finally
        {
            pinned.Free();
        }
    }

    public static SafeTensorRtObjectHandle CreateExecutionContext(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        TensorRtLineBindings bindings = GetBindings(line);
        BridgeStatusCode status = bindings.ExecutionContextCreate(engine, out SafeTensorRtObjectHandle context);
        NativeStatus.ThrowIfFailed(status);
        return context;
    }

    public static SafeTensorRtObjectHandle CreateExecutionContextWithoutDeviceMemory(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        SafeTensorRtObjectHandle context;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_create_execution_context_without_device_memory(engine, out context),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_create_execution_context_without_device_memory(engine, out context),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_create_execution_context_without_device_memory(engine, out context),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return context;
    }

    public static SafeTensorRtObjectHandle CreateOptimizationProfile(TensorRtApiLine line, SafeTensorRtObjectHandle builder)
    {
        SafeTensorRtObjectHandle profile;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_builder_create_optimization_profile(builder, out profile);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_builder_create_optimization_profile(builder, out profile);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_builder_create_optimization_profile(builder, out profile);
                break;
            default:
                throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.");
        }

        NativeStatus.ThrowIfFailed(status);
        return profile;
    }

    public static void SetOptimizationProfileShape(TensorRtApiLine line, SafeTensorRtObjectHandle profile, string inputName, TensorRtOptimizationProfileSelector selector, TensorRtDims dims)
    {
        if (string.IsNullOrWhiteSpace(inputName))
        {
            throw new ArgumentException("Input name must not be null or empty.", nameof(inputName));
        }

        if (dims == null)
        {
            throw new ArgumentNullException(nameof(dims));
        }

        using Utf8Interop.Utf8StringScope inputNameUtf8 = Utf8Interop.ToNativeString(inputName);
        NativeTensorRtDims nativeDims = dims.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_optimization_profile_set_shape(profile, inputNameUtf8.Pointer, (int)selector, ref nativeDims),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_optimization_profile_set_shape(profile, inputNameUtf8.Pointer, (int)selector, ref nativeDims),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_optimization_profile_set_shape(profile, inputNameUtf8.Pointer, (int)selector, ref nativeDims),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDims GetOptimizationProfileShape(TensorRtApiLine line, SafeTensorRtObjectHandle profile, string inputName, TensorRtOptimizationProfileSelector selector)
    {
        if (string.IsNullOrWhiteSpace(inputName))
        {
            throw new ArgumentException("Input name must not be null or empty.", nameof(inputName));
        }

        using Utf8Interop.Utf8StringScope inputNameUtf8 = Utf8Interop.ToNativeString(inputName);
        NativeTensorRtDims nativeDims;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_optimization_profile_get_shape(profile, inputNameUtf8.Pointer, (int)selector, out nativeDims),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_optimization_profile_get_shape(profile, inputNameUtf8.Pointer, (int)selector, out nativeDims),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_optimization_profile_get_shape(profile, inputNameUtf8.Pointer, (int)selector, out nativeDims),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(nativeDims);
    }

    public static void SetOptimizationProfileShapeValues(TensorRtApiLine line, SafeTensorRtObjectHandle profile, string inputName, TensorRtOptimizationProfileSelector selector, IReadOnlyList<int> values)
    {
        if (string.IsNullOrWhiteSpace(inputName))
        {
            throw new ArgumentException("Input name must not be null or empty.", nameof(inputName));
        }

        if (values == null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        if (values.Count == 0)
        {
            throw new ArgumentException("Shape values must not be empty.", nameof(values));
        }

        int[] valueArray = new int[values.Count];
        for (int i = 0; i < values.Count; i++)
        {
            valueArray[i] = values[i];
        }

        using Utf8Interop.Utf8StringScope inputNameUtf8 = Utf8Interop.ToNativeString(inputName);
        GCHandle pinned = GCHandle.Alloc(valueArray, GCHandleType.Pinned);
        try
        {
            IntPtr valuePointer = pinned.AddrOfPinnedObject();
            BridgeStatusCode status = line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_optimization_profile_set_shape_values(profile, inputNameUtf8.Pointer, (int)selector, valuePointer, valueArray.Length),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_optimization_profile_set_shape_values(profile, inputNameUtf8.Pointer, (int)selector, valuePointer, valueArray.Length),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_optimization_profile_set_shape_values(profile, inputNameUtf8.Pointer, (int)selector, valuePointer, valueArray.Length),
                _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
            };

            NativeStatus.ThrowIfFailed(status);
        }
        finally
        {
            pinned.Free();
        }
    }

    public static int GetOptimizationProfileShapeValueCount(TensorRtApiLine line, SafeTensorRtObjectHandle profile, string inputName)
    {
        if (string.IsNullOrWhiteSpace(inputName))
        {
            throw new ArgumentException("Input name must not be null or empty.", nameof(inputName));
        }

        using Utf8Interop.Utf8StringScope inputNameUtf8 = Utf8Interop.ToNativeString(inputName);
        int count;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_optimization_profile_get_shape_value_count(profile, inputNameUtf8.Pointer, out count);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_optimization_profile_get_shape_value_count(profile, inputNameUtf8.Pointer, out count);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_optimization_profile_get_shape_value_count(profile, inputNameUtf8.Pointer, out count);
                break;
            default:
                throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.");
        }

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static int[] GetOptimizationProfileShapeValues(TensorRtApiLine line, SafeTensorRtObjectHandle profile, string inputName, TensorRtOptimizationProfileSelector selector)
    {
        int count = GetOptimizationProfileShapeValueCount(line, profile, inputName);
        if (count <= 0)
        {
            return Array.Empty<int>();
        }

        int[] values = new int[count];
        using Utf8Interop.Utf8StringScope inputNameUtf8 = Utf8Interop.ToNativeString(inputName);
        GCHandle pinned = GCHandle.Alloc(values, GCHandleType.Pinned);
        try
        {
            int actualCount;
            BridgeStatusCode status;
            switch (line)
            {
                case TensorRtApiLine.TensorRt8:
                    status = NativeMethodsTensorRt.jyppx_trt8_optimization_profile_get_shape_values(profile, inputNameUtf8.Pointer, (int)selector, pinned.AddrOfPinnedObject(), values.Length, out actualCount);
                    break;
                case TensorRtApiLine.TensorRt10:
                    status = NativeMethodsTensorRt.jyppx_trt10_optimization_profile_get_shape_values(profile, inputNameUtf8.Pointer, (int)selector, pinned.AddrOfPinnedObject(), values.Length, out actualCount);
                    break;
                case TensorRtApiLine.TensorRt11:
                    status = NativeMethodsTensorRt.jyppx_trt11_optimization_profile_get_shape_values(profile, inputNameUtf8.Pointer, (int)selector, pinned.AddrOfPinnedObject(), values.Length, out actualCount);
                    break;
                default:
                    throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.");
            }

            NativeStatus.ThrowIfFailed(status);
            if (actualCount == values.Length)
            {
                return values;
            }

            int[] trimmed = new int[Math.Max(actualCount, 0)];
            Array.Copy(values, trimmed, trimmed.Length);
            return trimmed;
        }
        finally
        {
            pinned.Free();
        }
    }

    public static void SetOptimizationProfileExtraMemoryTarget(TensorRtApiLine line, SafeTensorRtObjectHandle profile, float target)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_optimization_profile_set_extra_memory_target(profile, target),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_optimization_profile_set_extra_memory_target(profile, target),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_optimization_profile_set_extra_memory_target(profile, target),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static float GetOptimizationProfileExtraMemoryTarget(TensorRtApiLine line, SafeTensorRtObjectHandle profile)
    {
        float target;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_optimization_profile_get_extra_memory_target(profile, out target);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_optimization_profile_get_extra_memory_target(profile, out target);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_optimization_profile_get_extra_memory_target(profile, out target);
                break;
            default:
                throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.");
        }

        NativeStatus.ThrowIfFailed(status);
        return target;
    }

    public static bool IsOptimizationProfileValid(TensorRtApiLine line, SafeTensorRtObjectHandle profile)
    {
        int isValid;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_optimization_profile_is_valid(profile, out isValid);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_optimization_profile_is_valid(profile, out isValid);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_optimization_profile_is_valid(profile, out isValid);
                break;
            default:
                throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.");
        }

        NativeStatus.ThrowIfFailed(status);
        return isValid != 0;
    }

    public static int AddOptimizationProfile(TensorRtApiLine line, SafeTensorRtObjectHandle config, SafeTensorRtObjectHandle profile)
    {
        int index;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_builder_config_add_optimization_profile(config, profile, out index);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_builder_config_add_optimization_profile(config, profile, out index);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_builder_config_add_optimization_profile(config, profile, out index);
                break;
            default:
                throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.");
        }

        NativeStatus.ThrowIfFailed(status);
        return index;
    }

    public static void SetBuilderConfigProfileStream(TensorRtApiLine line, SafeTensorRtObjectHandle config, SafeCudaStreamHandle stream)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_profile_stream(config, stream),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_profile_stream(config, stream),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_profile_stream(config, stream),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static bool IsBuilderConfigProfileStreamSet(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int isSet;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_is_profile_stream_set(config, out isSet),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_is_profile_stream_set(config, out isSet),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_is_profile_stream_set(config, out isSet),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return isSet != 0;
    }

    public static int GetBuilderConfigOptimizationProfileCount(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_optimization_profile_count(config, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_optimization_profile_count(config, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_optimization_profile_count(config, out count),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static void SetBuilderConfigCalibrationProfile(TensorRtApiLine line, SafeTensorRtObjectHandle config, SafeTensorRtObjectHandle profile)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_calibration_profile(config, profile),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_calibration_profile(config, profile),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_calibration_profile(config, profile),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static bool HasBuilderConfigCalibrationProfile(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int hasProfile;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_has_calibration_profile(config, out hasProfile),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_has_calibration_profile(config, out hasProfile),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_has_calibration_profile(config, out hasProfile),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return hasProfile != 0;
    }

    public static bool HasBuilderConfigAlgorithmSelectorCompatibility(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int hasSelector;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_has_algorithm_selector(config, out hasSelector),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_has_algorithm_selector(config, out hasSelector),
            TensorRtApiLine.TensorRt11 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "IBuilderConfig::getAlgorithmSelector presence is exposed by this bridge for TensorRT 8 and 10; TensorRT 11 callback ownership remains deferred."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return hasSelector != 0;
    }

    public static bool HasBuilderConfigInt8CalibratorCompatibility(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int hasCalibrator;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_has_int8_calibrator(config, out hasCalibrator),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_has_int8_calibrator(config, out hasCalibrator),
            TensorRtApiLine.TensorRt11 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "IBuilderConfig::getInt8Calibrator presence is exposed by this bridge for TensorRT 8 and 10; TensorRT 11 callback ownership remains deferred."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return hasCalibrator != 0;
    }

    public static void SetBuilderConfigFlag(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtBuilderFlag flag, bool enabled)
    {
        int nativeFlag = TensorRtBuilderFlagMapper.ToNativeFlag(line, flag);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_flag(config, nativeFlag, enabled ? 1 : 0),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_flag(config, nativeFlag, enabled ? 1 : 0),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_flag(config, nativeFlag, enabled ? 1 : 0),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static bool GetBuilderConfigFlag(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtBuilderFlag flag)
    {
        int nativeFlag = TensorRtBuilderFlagMapper.ToNativeFlag(line, flag);
        int enabled;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_flag(config, nativeFlag, out enabled),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_flag(config, nativeFlag, out enabled),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_flag(config, nativeFlag, out enabled),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return enabled != 0;
    }

    public static void SetBuilderConfigEngineCapability(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtEngineCapability capability)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_engine_capability(config, (int)capability),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_engine_capability(config, (int)capability),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_engine_capability(config, (int)capability),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtEngineCapability GetBuilderConfigEngineCapability(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int capability;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_engine_capability(config, out capability),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_engine_capability(config, out capability),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_engine_capability(config, out capability),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtEngineCapability)capability;
    }

    public static void SetBuilderConfigPreviewFeature(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtPreviewFeature feature, bool enabled)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_preview_feature(config, (int)feature, enabled ? 1 : 0),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_preview_feature(config, (int)feature, enabled ? 1 : 0),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_preview_feature(config, (int)feature, enabled ? 1 : 0),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static bool GetBuilderConfigPreviewFeature(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtPreviewFeature feature)
    {
        int enabled;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_preview_feature(config, (int)feature, out enabled),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_preview_feature(config, (int)feature, out enabled),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_preview_feature(config, (int)feature, out enabled),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return enabled != 0;
    }

    public static void SetBuilderConfigHardwareCompatibilityLevel(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtHardwareCompatibilityLevel level)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_hardware_compatibility_level(config, (int)level),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_hardware_compatibility_level(config, (int)level),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_hardware_compatibility_level(config, (int)level),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtHardwareCompatibilityLevel GetBuilderConfigHardwareCompatibilityLevel(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int level;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_hardware_compatibility_level(config, out level),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_hardware_compatibility_level(config, out level),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_hardware_compatibility_level(config, out level),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtHardwareCompatibilityLevel)level;
    }

    public static void SetBuilderConfigRuntimePlatform(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtRuntimePlatform platform)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_runtime_platform(config, (int)platform),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_runtime_platform(config, (int)platform),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_runtime_platform(config, (int)platform),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtRuntimePlatform GetBuilderConfigRuntimePlatform(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int platform;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_runtime_platform(config, out platform),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_runtime_platform(config, out platform),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_runtime_platform(config, out platform),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtRuntimePlatform)platform;
    }

    public static void SetLayerDeviceType(TensorRtApiLine line, SafeTensorRtObjectHandle config, SafeTensorRtObjectHandle layer, TensorRtDeviceType deviceType)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_layer_device_type(config, layer, (int)deviceType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_layer_device_type(config, layer, (int)deviceType),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_layer_device_type(config, layer, (int)deviceType),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDeviceType GetLayerDeviceType(TensorRtApiLine line, SafeTensorRtObjectHandle config, SafeTensorRtObjectHandle layer)
    {
        int deviceType;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_layer_device_type(config, layer, out deviceType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_layer_device_type(config, layer, out deviceType),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_layer_device_type(config, layer, out deviceType),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtDeviceType)deviceType;
    }

    public static bool IsLayerDeviceTypeSet(TensorRtApiLine line, SafeTensorRtObjectHandle config, SafeTensorRtObjectHandle layer)
    {
        int isSet;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_is_layer_device_type_set(config, layer, out isSet),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_is_layer_device_type_set(config, layer, out isSet),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_is_layer_device_type_set(config, layer, out isSet),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return isSet != 0;
    }

    public static void ResetLayerDeviceType(TensorRtApiLine line, SafeTensorRtObjectHandle config, SafeTensorRtObjectHandle layer)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_reset_layer_device_type(config, layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_reset_layer_device_type(config, layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_reset_layer_device_type(config, layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetMemoryPoolLimit(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtMemoryPoolType pool, ulong bytes)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_memory_pool_limit(config, (int)pool, new UIntPtr(bytes)),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_memory_pool_limit(config, (int)pool, new UIntPtr(bytes)),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_memory_pool_limit(config, (int)pool, new UIntPtr(bytes)),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static ulong GetMemoryPoolLimit(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtMemoryPoolType pool)
    {
        UIntPtr bytes;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_memory_pool_limit(config, (int)pool, out bytes),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_memory_pool_limit(config, (int)pool, out bytes),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_memory_pool_limit(config, (int)pool, out bytes),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return bytes.ToUInt64();
    }

    public static void SetBuilderOptimizationLevel(TensorRtApiLine line, SafeTensorRtObjectHandle config, int level)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_optimization_level(config, level),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_optimization_level(config, level),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_optimization_level(config, level),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static int GetBuilderOptimizationLevel(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int level;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_optimization_level(config, out level),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_optimization_level(config, out level),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_optimization_level(config, out level),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return level;
    }

    public static void SetProfilingVerbosity(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtProfilingVerbosity verbosity)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_profiling_verbosity(config, (int)verbosity),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_profiling_verbosity(config, (int)verbosity),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_profiling_verbosity(config, (int)verbosity),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtProfilingVerbosity GetProfilingVerbosity(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int verbosity;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_profiling_verbosity(config, out verbosity),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_profiling_verbosity(config, out verbosity),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_profiling_verbosity(config, out verbosity),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtProfilingVerbosity)verbosity;
    }

    public static void SetMaxAuxStreams(TensorRtApiLine line, SafeTensorRtObjectHandle config, int maxStreams)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_max_aux_streams(config, maxStreams),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_max_aux_streams(config, maxStreams),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_max_aux_streams(config, maxStreams),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static int GetMaxAuxStreams(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int maxStreams;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_max_aux_streams(config, out maxStreams),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_max_aux_streams(config, out maxStreams),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_max_aux_streams(config, out maxStreams),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return maxStreams;
    }

    public static void SetAverageTimingIterations(TensorRtApiLine line, SafeTensorRtObjectHandle config, int iterations)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_average_timing_iterations(config, iterations),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_average_timing_iterations(config, iterations),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_average_timing_iterations(config, iterations),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static int GetAverageTimingIterations(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int iterations;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_average_timing_iterations(config, out iterations),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_average_timing_iterations(config, out iterations),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_average_timing_iterations(config, out iterations),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return iterations;
    }

    public static ulong GetMaxWorkspaceSizeCompatibility(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        if (line != TensorRtApiLine.TensorRt8)
        {
            throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "IBuilderConfig::getMaxWorkspaceSize is a TensorRT 8 legacy compatibility API. Use GetMemoryPoolLimit(Workspace) for portable TensorRT 8/10/11 diagnostics.");
        }

        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt8_builder_config_get_max_workspace_size(config, out UIntPtr workspaceSize);
        NativeStatus.ThrowIfFailed(status);
        return workspaceSize.ToUInt64();
    }

    public static void SetMaxWorkspaceSizeCompatibility(TensorRtApiLine line, SafeTensorRtObjectHandle config, ulong workspaceSize)
    {
        if (line != TensorRtApiLine.TensorRt8)
        {
            throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "IBuilderConfig::setMaxWorkspaceSize is a TensorRT 8 legacy compatibility API. Use SetMemoryPoolLimit(Workspace) for portable TensorRT 8/10/11 configuration.");
        }

        if (UIntPtr.Size == 4 && workspaceSize > uint.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(workspaceSize), "Workspace size exceeds the native size_t range.");
        }

        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt8_builder_config_set_max_workspace_size(config, (UIntPtr)workspaceSize);
        NativeStatus.ThrowIfFailed(status);
    }

    public static int GetMinTimingIterationsCompatibility(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        if (line != TensorRtApiLine.TensorRt8)
        {
            throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "IBuilderConfig::getMinTimingIterations is a TensorRT 8 legacy compatibility API. Use GetAverageTimingIterations for portable TensorRT 8/10/11 diagnostics.");
        }

        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt8_builder_config_get_min_timing_iterations(config, out int iterations);
        NativeStatus.ThrowIfFailed(status);
        return iterations;
    }

    public static void SetMinTimingIterationsCompatibility(TensorRtApiLine line, SafeTensorRtObjectHandle config, int iterations)
    {
        if (line != TensorRtApiLine.TensorRt8)
        {
            throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "IBuilderConfig::setMinTimingIterations is a TensorRT 8 legacy compatibility API. Use SetAverageTimingIterations for portable TensorRT 8/10/11 configuration.");
        }

        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt8_builder_config_set_min_timing_iterations(config, iterations);
        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetTacticSources(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtTacticSources sources)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_tactic_sources(config, (uint)sources),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_tactic_sources(config, (uint)sources),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_tactic_sources(config, (uint)sources),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtTacticSources GetTacticSources(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        uint tacticSources;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_get_tactic_sources(config, out tacticSources),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_get_tactic_sources(config, out tacticSources),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_get_tactic_sources(config, out tacticSources),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtTacticSources)tacticSources;
    }

    public static SafeTensorRtObjectHandle AddNetworkInput(TensorRtApiLine line, SafeTensorRtObjectHandle network, string name, TensorRtDataType dataType, TensorRtDims dims)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Tensor name must not be null or empty.", nameof(name));
        }

        if (dims == null)
        {
            throw new ArgumentNullException(nameof(dims));
        }

        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(name);
        NativeTensorRtDims nativeDims = dims.ToNative();
        SafeTensorRtObjectHandle tensor;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_input(network, nameUtf8.Pointer, (int)dataType, ref nativeDims, out tensor),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_input(network, nameUtf8.Pointer, (int)dataType, ref nativeDims, out tensor),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_input(network, nameUtf8.Pointer, (int)dataType, ref nativeDims, out tensor),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return tensor;
    }

    public static void MarkNetworkOutput(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle tensor)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_mark_output(network, tensor),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_mark_output(network, tensor),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_mark_output(network, tensor),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static void UnmarkNetworkOutput(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle tensor)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_unmark_output(network, tensor),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_unmark_output(network, tensor),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_unmark_output(network, tensor),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static int GetNetworkInputCount(TensorRtApiLine line, SafeTensorRtObjectHandle network)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_get_input_count(network, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_get_input_count(network, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_get_input_count(network, out count),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static int GetNetworkOutputCount(TensorRtApiLine line, SafeTensorRtObjectHandle network)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_get_output_count(network, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_get_output_count(network, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_get_output_count(network, out count),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static int GetNetworkLayerCount(TensorRtApiLine line, SafeTensorRtObjectHandle network)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_get_layer_count(network, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_get_layer_count(network, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_get_layer_count(network, out count),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static SafeTensorRtObjectHandle GetNetworkLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, int index)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_get_layer(network, index, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_get_layer(network, index, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_get_layer(network, index, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static string GetNetworkName(TensorRtApiLine line, SafeTensorRtObjectHandle network)
    {
        BridgeStatusCode status = GetNetworkNameNative(line, network, IntPtr.Zero, UIntPtr.Zero, out UIntPtr requiredSize);
        NativeStatus.ThrowIfFailed(status);

        ulong required = requiredSize.ToUInt64();
        if (required == 0)
        {
            return string.Empty;
        }

        if (required > int.MaxValue)
        {
            throw new BridgeProbeException(BridgeStatusCode.BufferTooSmall, BridgeErrorCategory.TensorRt, "Network name is too large for the managed buffer.");
        }

        IntPtr buffer = Marshal.AllocHGlobal(checked((int)required));
        try
        {
            status = GetNetworkNameNative(line, network, buffer, requiredSize, out _);
            NativeStatus.ThrowIfFailed(status);
            return Utf8Interop.ReadString(buffer);
        }
        finally
        {
            Marshal.FreeHGlobal(buffer);
        }
    }

    public static void SetNetworkName(TensorRtApiLine line, SafeTensorRtObjectHandle network, string name)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Network name must not be null or empty.", nameof(name));
        }

        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(name);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_set_name(network, nameUtf8.Pointer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_set_name(network, nameUtf8.Pointer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_set_name(network, nameUtf8.Pointer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtNetworkDefinitionCreationFlags GetNetworkFlags(TensorRtApiLine line, SafeTensorRtObjectHandle network)
    {
        uint flags;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_get_flags(network, out flags),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_get_flags(network, out flags),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_get_flags(network, out flags),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtNetworkDefinitionCreationFlags)flags;
    }

    public static bool HasImplicitBatchDimension(TensorRtApiLine line, SafeTensorRtObjectHandle network)
    {
        int hasImplicitBatchDimension;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_has_implicit_batch_dimension(network, out hasImplicitBatchDimension),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_has_implicit_batch_dimension(network, out hasImplicitBatchDimension),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_has_implicit_batch_dimension(network, out hasImplicitBatchDimension),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return hasImplicitBatchDimension != 0;
    }

    public static bool GetNetworkFlag(TensorRtApiLine line, SafeTensorRtObjectHandle network, TensorRtNetworkDefinitionCreationFlags flag)
    {
        int flagIndex = GetSingleBitFlagIndex((uint)flag, nameof(flag));
        int enabled;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_get_flag(network, flagIndex, out enabled),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_get_flag(network, flagIndex, out enabled),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_get_flag(network, flagIndex, out enabled),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return enabled != 0;
    }

    public static SafeTensorRtObjectHandle GetNetworkInput(TensorRtApiLine line, SafeTensorRtObjectHandle network, int index)
    {
        SafeTensorRtObjectHandle tensor;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_get_input(network, index, out tensor),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_get_input(network, index, out tensor),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_get_input(network, index, out tensor),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return tensor;
    }

    public static SafeTensorRtObjectHandle GetNetworkOutput(TensorRtApiLine line, SafeTensorRtObjectHandle network, int index)
    {
        SafeTensorRtObjectHandle tensor;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_get_output(network, index, out tensor),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_get_output(network, index, out tensor),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_get_output(network, index, out tensor),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return tensor;
    }

    public static SafeTensorRtObjectHandle AddIdentityLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle input)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_identity(network, input, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_identity(network, input, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_identity(network, input, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddConstantLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, TensorRtDims shape, TensorRtWeights weights)
    {
        if (shape == null)
        {
            throw new ArgumentNullException(nameof(shape));
        }

        if (weights == null)
        {
            throw new ArgumentNullException(nameof(weights));
        }

        NativeTensorRtDims nativeDims = shape.ToNative();
        using TensorRtWeights.PinnedScope pinnedWeights = weights.Pin();
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_constant(network, ref nativeDims, (int)weights.DataType, pinnedWeights.Pointer, (UIntPtr)weights.ElementCount, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_constant(network, ref nativeDims, (int)weights.DataType, pinnedWeights.Pointer, (UIntPtr)weights.ElementCount, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_constant(network, ref nativeDims, (int)weights.DataType, pinnedWeights.Pointer, (UIntPtr)weights.ElementCount, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddConvolutionLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle input,
        int outputMaps,
        TensorRtDims kernelSize,
        TensorRtWeights kernelWeights,
        TensorRtWeights? biasWeights)
    {
        if (kernelSize == null)
        {
            throw new ArgumentNullException(nameof(kernelSize));
        }

        if (kernelWeights == null)
        {
            throw new ArgumentNullException(nameof(kernelWeights));
        }

        if (kernelWeights.IsEmpty)
        {
            throw new ArgumentException("Convolution kernel weights must not be empty.", nameof(kernelWeights));
        }

        if (outputMaps <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(outputMaps), "Convolution output maps must be greater than zero.");
        }

        ValidateOptionalWeightsDataType(kernelWeights.DataType, biasWeights, nameof(biasWeights));
        NativeTensorRtDims nativeKernelSize = kernelSize.ToNative();
        using TensorRtWeights.PinnedScope kernelPinned = kernelWeights.Pin();
        TensorRtWeights.PinnedScope? biasPinned = PinOptionalWeights(biasWeights);
        try
        {
            SafeTensorRtObjectHandle layer;
            BridgeStatusCode status = line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_convolution_nd(network, input, outputMaps, ref nativeKernelSize, (int)kernelWeights.DataType, kernelPinned.Pointer, (UIntPtr)kernelWeights.ElementCount, biasPinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(biasWeights?.ElementCount ?? 0), out layer),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_convolution_nd(network, input, outputMaps, ref nativeKernelSize, (int)kernelWeights.DataType, kernelPinned.Pointer, (UIntPtr)kernelWeights.ElementCount, biasPinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(biasWeights?.ElementCount ?? 0), out layer),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_convolution_nd(network, input, outputMaps, ref nativeKernelSize, (int)kernelWeights.DataType, kernelPinned.Pointer, (UIntPtr)kernelWeights.ElementCount, biasPinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(biasWeights?.ElementCount ?? 0), out layer),
                _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
            };

            NativeStatus.ThrowIfFailed(status);
            return layer;
        }
        finally
        {
            biasPinned?.Dispose();
        }
    }

    public static SafeTensorRtObjectHandle AddDeconvolutionLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle input,
        int outputMaps,
        TensorRtDims kernelSize,
        TensorRtWeights kernelWeights,
        TensorRtWeights? biasWeights)
    {
        if (kernelSize == null)
        {
            throw new ArgumentNullException(nameof(kernelSize));
        }

        if (kernelWeights == null)
        {
            throw new ArgumentNullException(nameof(kernelWeights));
        }

        if (kernelWeights.IsEmpty)
        {
            throw new ArgumentException("Deconvolution kernel weights must not be empty.", nameof(kernelWeights));
        }

        if (outputMaps <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(outputMaps), "Deconvolution output maps must be greater than zero.");
        }

        ValidateOptionalWeightsDataType(kernelWeights.DataType, biasWeights, nameof(biasWeights));
        NativeTensorRtDims nativeKernelSize = kernelSize.ToNative();
        using TensorRtWeights.PinnedScope kernelPinned = kernelWeights.Pin();
        TensorRtWeights.PinnedScope? biasPinned = PinOptionalWeights(biasWeights);
        try
        {
            SafeTensorRtObjectHandle layer;
            BridgeStatusCode status = line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_deconvolution_nd(network, input, outputMaps, ref nativeKernelSize, (int)kernelWeights.DataType, kernelPinned.Pointer, (UIntPtr)kernelWeights.ElementCount, biasPinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(biasWeights?.ElementCount ?? 0), out layer),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_deconvolution_nd(network, input, outputMaps, ref nativeKernelSize, (int)kernelWeights.DataType, kernelPinned.Pointer, (UIntPtr)kernelWeights.ElementCount, biasPinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(biasWeights?.ElementCount ?? 0), out layer),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_deconvolution_nd(network, input, outputMaps, ref nativeKernelSize, (int)kernelWeights.DataType, kernelPinned.Pointer, (UIntPtr)kernelWeights.ElementCount, biasPinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(biasWeights?.ElementCount ?? 0), out layer),
                _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
            };

            NativeStatus.ThrowIfFailed(status);
            return layer;
        }
        finally
        {
            biasPinned?.Dispose();
        }
    }

    public static SafeTensorRtObjectHandle AddScaleLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle input,
        TensorRtScaleMode mode,
        TensorRtWeights? shift,
        TensorRtWeights? scale,
        TensorRtWeights? power,
        int channelAxis)
    {
        TensorRtDataType dataType = GetScaleWeightsDataType(shift, scale, power);
        ValidateOptionalWeightsDataType(dataType, shift, nameof(shift));
        ValidateOptionalWeightsDataType(dataType, scale, nameof(scale));
        ValidateOptionalWeightsDataType(dataType, power, nameof(power));

        TensorRtWeights.PinnedScope? shiftPinned = PinOptionalWeights(shift);
        TensorRtWeights.PinnedScope? scalePinned = PinOptionalWeights(scale);
        TensorRtWeights.PinnedScope? powerPinned = PinOptionalWeights(power);
        try
        {
            SafeTensorRtObjectHandle layer;
            BridgeStatusCode status = line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_scale_nd(network, input, (int)mode, (int)dataType, shiftPinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(shift?.ElementCount ?? 0), scalePinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(scale?.ElementCount ?? 0), powerPinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(power?.ElementCount ?? 0), channelAxis, out layer),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_scale_nd(network, input, (int)mode, (int)dataType, shiftPinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(shift?.ElementCount ?? 0), scalePinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(scale?.ElementCount ?? 0), powerPinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(power?.ElementCount ?? 0), channelAxis, out layer),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_scale_nd(network, input, (int)mode, (int)dataType, shiftPinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(shift?.ElementCount ?? 0), scalePinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(scale?.ElementCount ?? 0), powerPinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(power?.ElementCount ?? 0), channelAxis, out layer),
                _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
            };

            NativeStatus.ThrowIfFailed(status);
            return layer;
        }
        finally
        {
            powerPinned?.Dispose();
            scalePinned?.Dispose();
            shiftPinned?.Dispose();
        }
    }

    public static SafeTensorRtObjectHandle AddPaddingLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle input, TensorRtDims prePadding, TensorRtDims postPadding)
    {
        if (prePadding == null)
        {
            throw new ArgumentNullException(nameof(prePadding));
        }

        if (postPadding == null)
        {
            throw new ArgumentNullException(nameof(postPadding));
        }

        NativeTensorRtDims nativePrePadding = prePadding.ToNative();
        NativeTensorRtDims nativePostPadding = postPadding.ToNative();
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_padding_nd(network, input, ref nativePrePadding, ref nativePostPadding, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_padding_nd(network, input, ref nativePrePadding, ref nativePostPadding, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_padding_nd(network, input, ref nativePrePadding, ref nativePostPadding, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddElementWiseLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle left,
        SafeTensorRtObjectHandle right,
        TensorRtElementWiseOperation operation)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_elementwise(network, left, right, (int)operation, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_elementwise(network, left, right, (int)operation, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_elementwise(network, left, right, (int)operation, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddMatrixMultiplyLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle left,
        TensorRtMatrixOperation leftOperation,
        SafeTensorRtObjectHandle right,
        TensorRtMatrixOperation rightOperation)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_matrix_multiply(network, left, (int)leftOperation, right, (int)rightOperation, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_matrix_multiply(network, left, (int)leftOperation, right, (int)rightOperation, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_matrix_multiply(network, left, (int)leftOperation, right, (int)rightOperation, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static void SetMatrixMultiplyOperation(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int inputIndex, TensorRtMatrixOperation operation)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_matrix_multiply_layer_set_operation(layer, inputIndex, (int)operation),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_matrix_multiply_layer_set_operation(layer, inputIndex, (int)operation),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_matrix_multiply_layer_set_operation(layer, inputIndex, (int)operation),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtMatrixOperation GetMatrixMultiplyOperation(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int inputIndex)
    {
        int operation;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_matrix_multiply_layer_get_operation(layer, inputIndex, out operation),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_matrix_multiply_layer_get_operation(layer, inputIndex, out operation),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_matrix_multiply_layer_get_operation(layer, inputIndex, out operation),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtMatrixOperation)operation;
    }

    public static SafeTensorRtObjectHandle AddShuffleLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle input)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_shuffle(network, input, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_shuffle(network, input, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_shuffle(network, input, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static void SetShuffleReshapeDimensions(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims dims)
    {
        if (dims == null)
        {
            throw new ArgumentNullException(nameof(dims));
        }

        NativeTensorRtDims nativeDims = dims.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_shuffle_layer_set_reshape_dimensions(layer, ref nativeDims),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_shuffle_layer_set_reshape_dimensions(layer, ref nativeDims),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_shuffle_layer_set_reshape_dimensions(layer, ref nativeDims),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDims GetShuffleReshapeDimensions(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        NativeTensorRtDims dims;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_shuffle_layer_get_reshape_dimensions(layer, out dims),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_shuffle_layer_get_reshape_dimensions(layer, out dims),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_shuffle_layer_get_reshape_dimensions(layer, out dims),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(dims);
    }

    public static void SetShuffleFirstTranspose(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims permutation)
    {
        if (permutation == null)
        {
            throw new ArgumentNullException(nameof(permutation));
        }

        NativeTensorRtDims nativePermutation = permutation.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_shuffle_layer_set_first_transpose(layer, ref nativePermutation),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_shuffle_layer_set_first_transpose(layer, ref nativePermutation),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_shuffle_layer_set_first_transpose(layer, ref nativePermutation),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDims GetShuffleFirstTranspose(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        NativeTensorRtDims permutation;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_shuffle_layer_get_first_transpose(layer, out permutation),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_shuffle_layer_get_first_transpose(layer, out permutation),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_shuffle_layer_get_first_transpose(layer, out permutation),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(permutation);
    }

    public static void SetShuffleSecondTranspose(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims permutation)
    {
        if (permutation == null)
        {
            throw new ArgumentNullException(nameof(permutation));
        }

        NativeTensorRtDims nativePermutation = permutation.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_shuffle_layer_set_second_transpose(layer, ref nativePermutation),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_shuffle_layer_set_second_transpose(layer, ref nativePermutation),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_shuffle_layer_set_second_transpose(layer, ref nativePermutation),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDims GetShuffleSecondTranspose(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        NativeTensorRtDims permutation;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_shuffle_layer_get_second_transpose(layer, out permutation),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_shuffle_layer_get_second_transpose(layer, out permutation),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_shuffle_layer_get_second_transpose(layer, out permutation),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(permutation);
    }

    public static void SetShuffleZeroIsPlaceholder(TensorRtApiLine line, SafeTensorRtObjectHandle layer, bool zeroIsPlaceholder)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_shuffle_layer_set_zero_is_placeholder(layer, zeroIsPlaceholder ? 1 : 0),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_shuffle_layer_set_zero_is_placeholder(layer, zeroIsPlaceholder ? 1 : 0),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_shuffle_layer_set_zero_is_placeholder(layer, zeroIsPlaceholder ? 1 : 0),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static bool GetShuffleZeroIsPlaceholder(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int enabled;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_shuffle_layer_get_zero_is_placeholder(layer, out enabled),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_shuffle_layer_get_zero_is_placeholder(layer, out enabled),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_shuffle_layer_get_zero_is_placeholder(layer, out enabled),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return enabled != 0;
    }

    public static SafeTensorRtObjectHandle AddReduceLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle input,
        TensorRtReduceOperation operation,
        uint axes,
        bool keepDimensions)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_reduce(network, input, (int)operation, axes, keepDimensions ? 1 : 0, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_reduce(network, input, (int)operation, axes, keepDimensions ? 1 : 0, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_reduce(network, input, (int)operation, axes, keepDimensions ? 1 : 0, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static TensorRtReduceOperation GetReduceOperation(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int operation;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_reduce_layer_get_operation(layer, out operation),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_reduce_layer_get_operation(layer, out operation),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_reduce_layer_get_operation(layer, out operation),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtReduceOperation)operation;
    }

    public static uint GetReduceAxes(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        uint axes;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_reduce_layer_get_axes(layer, out axes),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_reduce_layer_get_axes(layer, out axes),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_reduce_layer_get_axes(layer, out axes),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return axes;
    }

    public static bool GetReduceKeepDimensions(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int keepDimensions;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_reduce_layer_get_keep_dimensions(layer, out keepDimensions),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_reduce_layer_get_keep_dimensions(layer, out keepDimensions),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_reduce_layer_get_keep_dimensions(layer, out keepDimensions),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return keepDimensions != 0;
    }

    public static SafeTensorRtObjectHandle AddSoftMaxLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle input)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_softmax(network, input, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_softmax(network, input, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_softmax(network, input, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static void SetSoftMaxAxes(TensorRtApiLine line, SafeTensorRtObjectHandle layer, uint axes)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_softmax_layer_set_axes(layer, axes),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_softmax_layer_set_axes(layer, axes),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_softmax_layer_set_axes(layer, axes),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static uint GetSoftMaxAxes(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        uint axes;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_softmax_layer_get_axes(layer, out axes),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_softmax_layer_get_axes(layer, out axes),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_softmax_layer_get_axes(layer, out axes),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return axes;
    }

    public static SafeTensorRtObjectHandle AddUnaryLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle input, TensorRtUnaryOperation operation)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_unary(network, input, (int)operation, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_unary(network, input, (int)operation, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_unary(network, input, (int)operation, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static TensorRtUnaryOperation GetUnaryOperation(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int operation;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_unary_layer_get_operation(layer, out operation),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_unary_layer_get_operation(layer, out operation),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_unary_layer_get_operation(layer, out operation),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtUnaryOperation)operation;
    }

    public static SafeTensorRtObjectHandle AddTopKLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle input,
        TensorRtTopKOperation operation,
        int k,
        uint axes)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_topk(network, input, (int)operation, k, axes, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_topk(network, input, (int)operation, k, axes, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_topk(network, input, (int)operation, k, axes, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static TensorRtTopKOperation GetTopKOperation(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int operation;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_topk_layer_get_operation(layer, out operation),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_topk_layer_get_operation(layer, out operation),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_topk_layer_get_operation(layer, out operation),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtTopKOperation)operation;
    }

    public static int GetTopKValue(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int k;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_topk_layer_get_k(layer, out k),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_topk_layer_get_k(layer, out k),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_topk_layer_get_k(layer, out k),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return k;
    }

    public static uint GetTopKAxes(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        uint axes;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_topk_layer_get_axes(layer, out axes),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_topk_layer_get_axes(layer, out axes),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_topk_layer_get_axes(layer, out axes),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return axes;
    }

    public static SafeTensorRtObjectHandle AddGatherLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle data,
        SafeTensorRtObjectHandle indices,
        int axis)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_gather(network, data, indices, axis, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_gather(network, data, indices, axis, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_gather(network, data, indices, axis, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static int GetGatherAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int axis;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_gather_layer_get_axis(layer, out axis),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_gather_layer_get_axis(layer, out axis),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_gather_layer_get_axis(layer, out axis),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return axis;
    }

    public static SafeTensorRtObjectHandle AddActivationLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle input, TensorRtActivationType activationType)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_activation(network, input, (int)activationType, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_activation(network, input, (int)activationType, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_activation(network, input, (int)activationType, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static TensorRtActivationType GetActivationType(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int activationType;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_activation_layer_get_type(layer, out activationType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_activation_layer_get_type(layer, out activationType),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_activation_layer_get_type(layer, out activationType),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtActivationType)activationType;
    }

    public static SafeTensorRtObjectHandle AddPoolingLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle input, TensorRtPoolingType poolingType, TensorRtDims windowSize)
    {
        if (windowSize == null)
        {
            throw new ArgumentNullException(nameof(windowSize));
        }

        NativeTensorRtDims nativeWindowSize = windowSize.ToNative();
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_pooling_nd(network, input, (int)poolingType, ref nativeWindowSize, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_pooling_nd(network, input, (int)poolingType, ref nativeWindowSize, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_pooling_nd(network, input, (int)poolingType, ref nativeWindowSize, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static TensorRtPoolingType GetPoolingType(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int poolingType;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_pooling_layer_get_type(layer, out poolingType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_pooling_layer_get_type(layer, out poolingType),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_pooling_layer_get_type(layer, out poolingType),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtPoolingType)poolingType;
    }

    public static void SetPoolingWindowSize(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims windowSize)
    {
        if (windowSize == null)
        {
            throw new ArgumentNullException(nameof(windowSize));
        }

        NativeTensorRtDims nativeWindowSize = windowSize.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_pooling_layer_set_window_size_nd(layer, ref nativeWindowSize),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_pooling_layer_set_window_size_nd(layer, ref nativeWindowSize),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_pooling_layer_set_window_size_nd(layer, ref nativeWindowSize),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDims GetPoolingWindowSize(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        NativeTensorRtDims nativeDims;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_pooling_layer_get_window_size_nd(layer, out nativeDims),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_pooling_layer_get_window_size_nd(layer, out nativeDims),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_pooling_layer_get_window_size_nd(layer, out nativeDims),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(nativeDims);
    }

    public static void SetPoolingStride(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims stride)
    {
        if (stride == null)
        {
            throw new ArgumentNullException(nameof(stride));
        }

        NativeTensorRtDims nativeStride = stride.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_pooling_layer_set_stride_nd(layer, ref nativeStride),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_pooling_layer_set_stride_nd(layer, ref nativeStride),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_pooling_layer_set_stride_nd(layer, ref nativeStride),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDims GetPoolingStride(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        NativeTensorRtDims nativeDims;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_pooling_layer_get_stride_nd(layer, out nativeDims),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_pooling_layer_get_stride_nd(layer, out nativeDims),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_pooling_layer_get_stride_nd(layer, out nativeDims),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(nativeDims);
    }

    public static void SetPoolingPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims padding)
    {
        if (padding == null)
        {
            throw new ArgumentNullException(nameof(padding));
        }

        NativeTensorRtDims nativePadding = padding.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_pooling_layer_set_padding_nd(layer, ref nativePadding),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_pooling_layer_set_padding_nd(layer, ref nativePadding),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_pooling_layer_set_padding_nd(layer, ref nativePadding),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDims GetPoolingPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        NativeTensorRtDims nativeDims;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_pooling_layer_get_padding_nd(layer, out nativeDims),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_pooling_layer_get_padding_nd(layer, out nativeDims),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_pooling_layer_get_padding_nd(layer, out nativeDims),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(nativeDims);
    }

    public static SafeTensorRtObjectHandle AddLrnLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle input, int windowSize, float alpha, float beta, float k)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_lrn(network, input, windowSize, alpha, beta, k, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_lrn(network, input, windowSize, alpha, beta, k, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_lrn(network, input, windowSize, alpha, beta, k, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static int GetLrnWindowSize(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int windowSize;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_lrn_layer_get_window_size(layer, out windowSize),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_lrn_layer_get_window_size(layer, out windowSize),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_lrn_layer_get_window_size(layer, out windowSize),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return windowSize;
    }

    public static void SetLrnWindowSize(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int windowSize)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_lrn_layer_set_window_size(layer, windowSize),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_lrn_layer_set_window_size(layer, windowSize),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_lrn_layer_set_window_size(layer, windowSize),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static float GetLrnAlpha(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        float alpha;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_lrn_layer_get_alpha(layer, out alpha),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_lrn_layer_get_alpha(layer, out alpha),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_lrn_layer_get_alpha(layer, out alpha),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return alpha;
    }

    public static void SetLrnAlpha(TensorRtApiLine line, SafeTensorRtObjectHandle layer, float alpha)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_lrn_layer_set_alpha(layer, alpha),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_lrn_layer_set_alpha(layer, alpha),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_lrn_layer_set_alpha(layer, alpha),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static float GetLrnBeta(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        float beta;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_lrn_layer_get_beta(layer, out beta),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_lrn_layer_get_beta(layer, out beta),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_lrn_layer_get_beta(layer, out beta),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return beta;
    }

    public static void SetLrnBeta(TensorRtApiLine line, SafeTensorRtObjectHandle layer, float beta)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_lrn_layer_set_beta(layer, beta),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_lrn_layer_set_beta(layer, beta),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_lrn_layer_set_beta(layer, beta),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static float GetLrnK(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        float k;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_lrn_layer_get_k(layer, out k),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_lrn_layer_get_k(layer, out k),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_lrn_layer_get_k(layer, out k),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return k;
    }

    public static void SetLrnK(TensorRtApiLine line, SafeTensorRtObjectHandle layer, float k)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_lrn_layer_set_k(layer, k),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_lrn_layer_set_k(layer, k),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_lrn_layer_set_k(layer, k),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static SafeTensorRtObjectHandle AddResizeLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle input)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_resize(network, input, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_resize(network, input, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_resize(network, input, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static void SetResizeOutputDimensions(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims dimensions)
    {
        if (dimensions == null)
        {
            throw new ArgumentNullException(nameof(dimensions));
        }

        NativeTensorRtDims nativeDimensions = dimensions.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_resize_layer_set_output_dimensions(layer, ref nativeDimensions),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_resize_layer_set_output_dimensions(layer, ref nativeDimensions),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_resize_layer_set_output_dimensions(layer, ref nativeDimensions),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDims GetResizeOutputDimensions(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        NativeTensorRtDims nativeDims;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_resize_layer_get_output_dimensions(layer, out nativeDims),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_resize_layer_get_output_dimensions(layer, out nativeDims),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_resize_layer_get_output_dimensions(layer, out nativeDims),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(nativeDims);
    }

    public static void SetResizeMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtResizeMode resizeMode)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_resize_layer_set_mode(layer, (int)resizeMode),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_resize_layer_set_mode(layer, (int)resizeMode),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_resize_layer_set_mode(layer, (int)resizeMode),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtResizeMode GetResizeMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int resizeMode;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_resize_layer_get_mode(layer, out resizeMode),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_resize_layer_get_mode(layer, out resizeMode),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_resize_layer_get_mode(layer, out resizeMode),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtResizeMode)resizeMode;
    }

    public static void SetResizeScales(TensorRtApiLine line, SafeTensorRtObjectHandle layer, float[] scales)
    {
        if (scales == null)
        {
            throw new ArgumentNullException(nameof(scales));
        }

        if (scales.Length == 0 || scales.Length > NativeTensorRtDims.MaxDimensionCount)
        {
            throw new ArgumentOutOfRangeException(nameof(scales), $"Resize scales must contain 1 to {NativeTensorRtDims.MaxDimensionCount} values.");
        }

        GCHandle handle = GCHandle.Alloc(scales, GCHandleType.Pinned);
        try
        {
            BridgeStatusCode status = line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_resize_layer_set_scales(layer, handle.AddrOfPinnedObject(), scales.Length),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_resize_layer_set_scales(layer, handle.AddrOfPinnedObject(), scales.Length),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_resize_layer_set_scales(layer, handle.AddrOfPinnedObject(), scales.Length),
                _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
            };

            NativeStatus.ThrowIfFailed(status);
        }
        finally
        {
            handle.Free();
        }
    }

    public static float[] GetResizeScales(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        float[] scales = new float[NativeTensorRtDims.MaxDimensionCount];
        GCHandle handle = GCHandle.Alloc(scales, GCHandleType.Pinned);
        try
        {
            int scaleCount;
            BridgeStatusCode status = line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_resize_layer_get_scales(layer, handle.AddrOfPinnedObject(), scales.Length, out scaleCount),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_resize_layer_get_scales(layer, handle.AddrOfPinnedObject(), scales.Length, out scaleCount),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_resize_layer_get_scales(layer, handle.AddrOfPinnedObject(), scales.Length, out scaleCount),
                _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
            };

            NativeStatus.ThrowIfFailed(status);
            if (scaleCount <= 0)
            {
                return Array.Empty<float>();
            }

            float[] result = new float[scaleCount];
            Array.Copy(scales, result, scaleCount);
            return result;
        }
        finally
        {
            handle.Free();
        }
    }

    public static SafeTensorRtObjectHandle AddConcatenationLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle[] inputs)
    {
        if (inputs == null)
        {
            throw new ArgumentNullException(nameof(inputs));
        }

        if (inputs.Length < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(inputs), "Concatenation requires at least two input tensors.");
        }

        IntPtr[] inputHandles = new IntPtr[inputs.Length];
        for (int index = 0; index < inputs.Length; index++)
        {
            if (inputs[index] == null || inputs[index].IsInvalid)
            {
                throw new ArgumentException("Input tensor handles must not be null or invalid.", nameof(inputs));
            }

            inputHandles[index] = inputs[index].DangerousGetHandle();
        }

        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_concatenation(network, inputHandles, inputHandles.Length, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_concatenation(network, inputHandles, inputHandles.Length, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_concatenation(network, inputHandles, inputHandles.Length, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static void SetConcatenationAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int axis)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_concatenation_layer_set_axis(layer, axis),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_concatenation_layer_set_axis(layer, axis),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_concatenation_layer_set_axis(layer, axis),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static int GetConcatenationAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int axis;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_concatenation_layer_get_axis(layer, out axis),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_concatenation_layer_get_axis(layer, out axis),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_concatenation_layer_get_axis(layer, out axis),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return axis;
    }

    public static SafeTensorRtObjectHandle AddSliceLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle input,
        TensorRtDims start,
        TensorRtDims size,
        TensorRtDims stride)
    {
        if (start == null)
        {
            throw new ArgumentNullException(nameof(start));
        }

        if (size == null)
        {
            throw new ArgumentNullException(nameof(size));
        }

        if (stride == null)
        {
            throw new ArgumentNullException(nameof(stride));
        }

        NativeTensorRtDims nativeStart = start.ToNative();
        NativeTensorRtDims nativeSize = size.ToNative();
        NativeTensorRtDims nativeStride = stride.ToNative();
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_slice(network, input, ref nativeStart, ref nativeSize, ref nativeStride, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_slice(network, input, ref nativeStart, ref nativeSize, ref nativeStride, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_slice(network, input, ref nativeStart, ref nativeSize, ref nativeStride, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static void SetSliceStart(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims start)
    {
        if (start == null)
        {
            throw new ArgumentNullException(nameof(start));
        }

        NativeTensorRtDims nativeStart = start.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_slice_layer_set_start(layer, ref nativeStart),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_slice_layer_set_start(layer, ref nativeStart),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_slice_layer_set_start(layer, ref nativeStart),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDims GetSliceStart(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        NativeTensorRtDims start;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_slice_layer_get_start(layer, out start),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_slice_layer_get_start(layer, out start),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_slice_layer_get_start(layer, out start),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(start);
    }

    public static void SetSliceSize(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims size)
    {
        if (size == null)
        {
            throw new ArgumentNullException(nameof(size));
        }

        NativeTensorRtDims nativeSize = size.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_slice_layer_set_size(layer, ref nativeSize),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_slice_layer_set_size(layer, ref nativeSize),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_slice_layer_set_size(layer, ref nativeSize),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDims GetSliceSize(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        NativeTensorRtDims size;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_slice_layer_get_size(layer, out size),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_slice_layer_get_size(layer, out size),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_slice_layer_get_size(layer, out size),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(size);
    }

    public static void SetSliceStride(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims stride)
    {
        if (stride == null)
        {
            throw new ArgumentNullException(nameof(stride));
        }

        NativeTensorRtDims nativeStride = stride.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_slice_layer_set_stride(layer, ref nativeStride),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_slice_layer_set_stride(layer, ref nativeStride),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_slice_layer_set_stride(layer, ref nativeStride),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDims GetSliceStride(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        NativeTensorRtDims stride;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_slice_layer_get_stride(layer, out stride),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_slice_layer_get_stride(layer, out stride),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_slice_layer_get_stride(layer, out stride),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(stride);
    }

    public static void SetSliceMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtSliceMode mode)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_slice_layer_set_mode(layer, (int)mode),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_slice_layer_set_mode(layer, (int)mode),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_slice_layer_set_mode(layer, (int)mode),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtSliceMode GetSliceMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int mode;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_slice_layer_get_mode(layer, out mode),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_slice_layer_get_mode(layer, out mode),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_slice_layer_get_mode(layer, out mode),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtSliceMode)mode;
    }

    public static SafeTensorRtObjectHandle AddShapeLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle input)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_shape(network, input, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_shape(network, input, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_shape(network, input, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddSelectLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle condition,
        SafeTensorRtObjectHandle thenInput,
        SafeTensorRtObjectHandle elseInput)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_select(network, condition, thenInput, elseInput, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_select(network, condition, thenInput, elseInput, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_select(network, condition, thenInput, elseInput, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddFillLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, TensorRtDims dimensions, TensorRtFillOperation operation)
    {
        if (dimensions == null)
        {
            throw new ArgumentNullException(nameof(dimensions));
        }

        NativeTensorRtDims nativeDimensions = dimensions.ToNative();
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_fill(network, ref nativeDimensions, (int)operation, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_fill(network, ref nativeDimensions, (int)operation, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_fill(network, ref nativeDimensions, (int)operation, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static void SetFillDimensions(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims dimensions)
    {
        if (dimensions == null)
        {
            throw new ArgumentNullException(nameof(dimensions));
        }

        NativeTensorRtDims nativeDimensions = dimensions.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_fill_layer_set_dimensions(layer, ref nativeDimensions),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_fill_layer_set_dimensions(layer, ref nativeDimensions),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_fill_layer_set_dimensions(layer, ref nativeDimensions),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDims GetFillDimensions(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        NativeTensorRtDims dimensions;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_fill_layer_get_dimensions(layer, out dimensions),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_fill_layer_get_dimensions(layer, out dimensions),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_fill_layer_get_dimensions(layer, out dimensions),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(dimensions);
    }

    public static void SetFillOperation(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtFillOperation operation)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_fill_layer_set_operation(layer, (int)operation),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_fill_layer_set_operation(layer, (int)operation),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_fill_layer_set_operation(layer, (int)operation),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtFillOperation GetFillOperation(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int operation;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_fill_layer_get_operation(layer, out operation),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_fill_layer_get_operation(layer, out operation),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_fill_layer_get_operation(layer, out operation),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtFillOperation)operation;
    }

    public static void SetFillAlpha(TensorRtApiLine line, SafeTensorRtObjectHandle layer, double alpha)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_fill_layer_set_alpha(layer, alpha),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_fill_layer_set_alpha(layer, alpha),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_fill_layer_set_alpha(layer, alpha),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static double GetFillAlpha(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        double alpha;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_fill_layer_get_alpha(layer, out alpha),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_fill_layer_get_alpha(layer, out alpha),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_fill_layer_get_alpha(layer, out alpha),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return alpha;
    }

    public static void SetFillBeta(TensorRtApiLine line, SafeTensorRtObjectHandle layer, double beta)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_fill_layer_set_beta(layer, beta),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_fill_layer_set_beta(layer, beta),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_fill_layer_set_beta(layer, beta),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static double GetFillBeta(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        double beta;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_fill_layer_get_beta(layer, out beta),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_fill_layer_get_beta(layer, out beta),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_fill_layer_get_beta(layer, out beta),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return beta;
    }

    public static SafeTensorRtObjectHandle GetLayerOutput(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        SafeTensorRtObjectHandle tensor;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_get_output(layer, index, out tensor),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_get_output(layer, index, out tensor),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_get_output(layer, index, out tensor),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return tensor;
    }

    public static SafeTensorRtObjectHandle GetLayerInput(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        SafeTensorRtObjectHandle tensor;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_get_input(layer, index, out tensor),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_get_input(layer, index, out tensor),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_get_input(layer, index, out tensor),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return tensor;
    }

    public static int GetLayerInputCount(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_get_input_count(layer, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_get_input_count(layer, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_get_input_count(layer, out count),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static int GetLayerOutputCount(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_get_output_count(layer, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_get_output_count(layer, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_get_output_count(layer, out count),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static TensorRtLayerType GetLayerType(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int type;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_get_type(layer, out type),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_get_type(layer, out type),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_get_type(layer, out type),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return MapLayerType(line, type);
    }

    private static TensorRtLayerType MapLayerType(TensorRtApiLine line, int nativeType)
    {
        if (line == TensorRtApiLine.TensorRt8)
        {
            switch (nativeType)
            {
                case 21:
                    return TensorRtLayerType.IdentityTrt8;
                case 23:
                    return TensorRtLayerType.SliceTrt8;
                case 24:
                    return TensorRtLayerType.ShapeTrt8;
                case 26:
                    return TensorRtLayerType.ResizeTrt8;
                case 31:
                    return TensorRtLayerType.SelectTrt8;
                case 32:
                    return TensorRtLayerType.FillTrt8;
                case 33:
                    return TensorRtLayerType.QuantizeTrt8;
                case 34:
                    return TensorRtLayerType.DequantizeTrt8;
                case 39:
                    return TensorRtLayerType.Einsum;
                case 40:
                    return TensorRtLayerType.Assertion;
                case 41:
                    return TensorRtLayerType.OneHot;
                case 43:
                    return TensorRtLayerType.GridSample;
                case 44:
                    return TensorRtLayerType.Nms;
                case 45:
                    return TensorRtLayerType.ReverseSequence;
                case 46:
                    return TensorRtLayerType.NormalizationTrt8;
                case 47:
                    return TensorRtLayerType.Cast;
            }
        }

        if (line == TensorRtApiLine.TensorRt10)
        {
            switch (nativeType)
            {
                case 20:
                    return TensorRtLayerType.IdentityTrt10;
                case 22:
                    return TensorRtLayerType.SliceTrt10;
                case 23:
                    return TensorRtLayerType.ShapeTrt10;
                case 25:
                    return TensorRtLayerType.ResizeTrt10;
                case 30:
                    return TensorRtLayerType.SelectTrt10;
                case 31:
                    return TensorRtLayerType.FillTrt10;
                case 32:
                    return TensorRtLayerType.QuantizeTrt10;
                case 33:
                    return TensorRtLayerType.DequantizeTrt10;
                case 45:
                    return TensorRtLayerType.NormalizationTrt10;
            }
        }

        return Enum.IsDefined(typeof(TensorRtLayerType), nativeType)
            ? (TensorRtLayerType)nativeType
            : TensorRtLayerType.Unknown;
    }

    public static string GetLayerName(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        BridgeStatusCode status = GetLayerNameNative(line, layer, IntPtr.Zero, UIntPtr.Zero, out UIntPtr requiredSize);
        NativeStatus.ThrowIfFailed(status);

        ulong required = requiredSize.ToUInt64();
        if (required == 0)
        {
            return string.Empty;
        }

        if (required > int.MaxValue)
        {
            throw new BridgeProbeException(BridgeStatusCode.BufferTooSmall, BridgeErrorCategory.TensorRt, "Layer name is too large for the managed buffer.");
        }

        IntPtr buffer = Marshal.AllocHGlobal(checked((int)required));
        try
        {
            status = GetLayerNameNative(line, layer, buffer, requiredSize, out _);
            NativeStatus.ThrowIfFailed(status);
            return Utf8Interop.ReadString(buffer);
        }
        finally
        {
            Marshal.FreeHGlobal(buffer);
        }
    }

    public static void SetLayerName(TensorRtApiLine line, SafeTensorRtObjectHandle layer, string name)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Layer name must not be null or empty.", nameof(name));
        }

        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(name);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_set_name(layer, nameUtf8.Pointer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_set_name(layer, nameUtf8.Pointer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_set_name(layer, nameUtf8.Pointer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetLayerPrecision(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDataType dataType)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_set_precision(layer, (int)dataType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_set_precision(layer, (int)dataType),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_set_precision(layer, (int)dataType),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDataType GetLayerPrecision(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int dataType;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_get_precision(layer, out dataType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_get_precision(layer, out dataType),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_get_precision(layer, out dataType),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtDataType)dataType;
    }

    public static bool IsLayerPrecisionSet(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int isSet;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_precision_is_set(layer, out isSet),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_precision_is_set(layer, out isSet),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_precision_is_set(layer, out isSet),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return isSet != 0;
    }

    public static void ResetLayerPrecision(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_reset_precision(layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_reset_precision(layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_reset_precision(layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetLayerOutputType(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int outputIndex, TensorRtDataType dataType)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_set_output_type(layer, outputIndex, (int)dataType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_set_output_type(layer, outputIndex, (int)dataType),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_set_output_type(layer, outputIndex, (int)dataType),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDataType GetLayerOutputType(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int outputIndex)
    {
        int dataType;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_get_output_type(layer, outputIndex, out dataType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_get_output_type(layer, outputIndex, out dataType),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_get_output_type(layer, outputIndex, out dataType),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtDataType)dataType;
    }

    public static bool IsLayerOutputTypeSet(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int outputIndex)
    {
        int isSet;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_output_type_is_set(layer, outputIndex, out isSet),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_output_type_is_set(layer, outputIndex, out isSet),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_output_type_is_set(layer, outputIndex, out isSet),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return isSet != 0;
    }

    public static void ResetLayerOutputType(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int outputIndex)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_reset_output_type(layer, outputIndex),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_reset_output_type(layer, outputIndex),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_reset_output_type(layer, outputIndex),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static string GetTensorName(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        BridgeStatusCode status = GetTensorNameNative(line, tensor, IntPtr.Zero, UIntPtr.Zero, out UIntPtr requiredSize);
        NativeStatus.ThrowIfFailed(status);

        ulong required = requiredSize.ToUInt64();
        if (required == 0)
        {
            return string.Empty;
        }

        if (required > int.MaxValue)
        {
            throw new BridgeProbeException(BridgeStatusCode.BufferTooSmall, BridgeErrorCategory.TensorRt, "Tensor name is too large for the managed buffer.");
        }

        IntPtr buffer = Marshal.AllocHGlobal(checked((int)required));
        try
        {
            status = GetTensorNameNative(line, tensor, buffer, requiredSize, out _);
            NativeStatus.ThrowIfFailed(status);
            return Utf8Interop.ReadString(buffer);
        }
        finally
        {
            Marshal.FreeHGlobal(buffer);
        }
    }

    public static void SetTensorName(TensorRtApiLine line, SafeTensorRtObjectHandle tensor, string name)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Tensor name must not be null or empty.", nameof(name));
        }

        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(name);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_set_name(tensor, nameUtf8.Pointer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_set_name(tensor, nameUtf8.Pointer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_set_name(tensor, nameUtf8.Pointer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDataType GetTensorDataType(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        int dataType;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_get_data_type(tensor, out dataType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_get_data_type(tensor, out dataType),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_get_data_type(tensor, out dataType),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtDataType)dataType;
    }

    public static TensorRtDims GetTensorShape(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        NativeTensorRtDims nativeDims;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_get_shape(tensor, out nativeDims),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_get_shape(tensor, out nativeDims),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_get_shape(tensor, out nativeDims),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(nativeDims);
    }

    public static void SetTensorShape(TensorRtApiLine line, SafeTensorRtObjectHandle tensor, TensorRtDims dims)
    {
        if (dims == null)
        {
            throw new ArgumentNullException(nameof(dims));
        }

        NativeTensorRtDims nativeDims = dims.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_set_shape(tensor, ref nativeDims),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_set_shape(tensor, ref nativeDims),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetTensorDataType(TensorRtApiLine line, SafeTensorRtObjectHandle tensor, TensorRtDataType dataType)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_set_data_type(tensor, (int)dataType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_set_data_type(tensor, (int)dataType),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtTensorLocation GetTensorLocation(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        int location;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_get_location(tensor, out location),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_get_location(tensor, out location),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_get_location(tensor, out location),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtTensorLocation)location;
    }

    public static void SetTensorLocation(TensorRtApiLine line, SafeTensorRtObjectHandle tensor, TensorRtTensorLocation location)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_set_location(tensor, (int)location),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_set_location(tensor, (int)location),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_set_location(tensor, (int)location),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtTensorFormats GetTensorAllowedFormats(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        uint formats;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_get_allowed_formats(tensor, out formats),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_get_allowed_formats(tensor, out formats),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_get_allowed_formats(tensor, out formats),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtTensorFormats)formats;
    }

    public static void SetTensorAllowedFormats(TensorRtApiLine line, SafeTensorRtObjectHandle tensor, TensorRtTensorFormats formats)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_set_allowed_formats(tensor, (uint)formats),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_set_allowed_formats(tensor, (uint)formats),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_set_allowed_formats(tensor, (uint)formats),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static bool IsTensorShapeTensor(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        int isShapeTensor = 0;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_is_shape_tensor(tensor, out isShapeTensor),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_is_shape_tensor(tensor, out isShapeTensor),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_is_shape_tensor(tensor, out isShapeTensor),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return isShapeTensor != 0;
    }

    public static bool IsTensorExecutionTensor(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        int isExecutionTensor = 0;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_is_execution_tensor(tensor, out isExecutionTensor),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_is_execution_tensor(tensor, out isExecutionTensor),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_is_execution_tensor(tensor, out isExecutionTensor),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return isExecutionTensor != 0;
    }

    public static bool GetTensorBroadcastAcrossBatch(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        int value = 0;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_get_broadcast_across_batch(tensor, out value),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_get_broadcast_across_batch(tensor, out value),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_get_broadcast_across_batch(tensor, out value),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return value != 0;
    }

    public static void SetTensorBroadcastAcrossBatch(TensorRtApiLine line, SafeTensorRtObjectHandle tensor, bool value)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_set_broadcast_across_batch(tensor, value ? 1 : 0),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_set_broadcast_across_batch(tensor, value ? 1 : 0),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_set_broadcast_across_batch(tensor, value ? 1 : 0),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static string GetTensorDimensionName(TensorRtApiLine line, SafeTensorRtObjectHandle tensor, int dimensionIndex)
    {
        if (dimensionIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dimensionIndex), "Dimension index must be greater than or equal to zero.");
        }

        return line switch
        {
            TensorRtApiLine.TensorRt8 => ReadUtf8Buffer((byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt8_tensor_get_dimension_name(tensor, dimensionIndex, buffer, size, out required), "Tensor dimension name is too large for the managed buffer."),
            TensorRtApiLine.TensorRt10 => ReadUtf8Buffer((byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt10_tensor_get_dimension_name(tensor, dimensionIndex, buffer, size, out required), "Tensor dimension name is too large for the managed buffer."),
            TensorRtApiLine.TensorRt11 => ReadUtf8Buffer((byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt11_tensor_get_dimension_name(tensor, dimensionIndex, buffer, size, out required), "Tensor dimension name is too large for the managed buffer."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
    }

    public static void SetTensorDimensionName(TensorRtApiLine line, SafeTensorRtObjectHandle tensor, int dimensionIndex, string name)
    {
        if (dimensionIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dimensionIndex), "Dimension index must be greater than or equal to zero.");
        }

        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Dimension name must not be null or empty. Use ClearTensorDimensionName to remove a name.", nameof(name));
        }

        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(name);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_set_dimension_name(tensor, dimensionIndex, nameUtf8.Pointer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_set_dimension_name(tensor, dimensionIndex, nameUtf8.Pointer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_set_dimension_name(tensor, dimensionIndex, nameUtf8.Pointer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static void ClearTensorDimensionName(TensorRtApiLine line, SafeTensorRtObjectHandle tensor, int dimensionIndex)
    {
        if (dimensionIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dimensionIndex), "Dimension index must be greater than or equal to zero.");
        }

        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_clear_dimension_name(tensor, dimensionIndex),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_clear_dimension_name(tensor, dimensionIndex),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_clear_dimension_name(tensor, dimensionIndex),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetTensorDynamicRange(TensorRtApiLine line, SafeTensorRtObjectHandle tensor, float minimum, float maximum)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_set_dynamic_range(tensor, minimum, maximum),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_set_dynamic_range(tensor, minimum, maximum),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_set_dynamic_range(tensor, minimum, maximum),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static bool IsTensorDynamicRangeSet(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        int value = 0;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_dynamic_range_is_set(tensor, out value),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_dynamic_range_is_set(tensor, out value),
            TensorRtApiLine.TensorRt11 => BridgeStatusCode.NotSupported,
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return value != 0;
    }

    public static float GetTensorDynamicRangeMin(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        float value;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_get_dynamic_range_min(tensor, out value),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_get_dynamic_range_min(tensor, out value),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_get_dynamic_range_min(tensor, out value),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    public static float GetTensorDynamicRangeMax(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        float value;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_get_dynamic_range_max(tensor, out value),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_get_dynamic_range_max(tensor, out value),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_get_dynamic_range_max(tensor, out value),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    public static void ResetTensorDynamicRange(TensorRtApiLine line, SafeTensorRtObjectHandle tensor)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_reset_dynamic_range(tensor),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_reset_dynamic_range(tensor),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_reset_dynamic_range(tensor),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    private static int GetSingleBitFlagIndex(uint value, string parameterName)
    {
        if (value == 0 || (value & (value - 1)) != 0)
        {
            throw new ArgumentOutOfRangeException(parameterName, "Flag query requires exactly one bit flag.");
        }

        int index = 0;
        while ((value >>= 1) != 0)
        {
            index++;
        }

        return index;
    }

    private static BridgeStatusCode GetNetworkNameNative(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        IntPtr outputBuffer,
        UIntPtr outputBufferSize,
        out UIntPtr requiredSize)
    {
        return line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_get_name(network, outputBuffer, outputBufferSize, out requiredSize),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_get_name(network, outputBuffer, outputBufferSize, out requiredSize),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_get_name(network, outputBuffer, outputBufferSize, out requiredSize),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
    }

    private static BridgeStatusCode GetTensorNameNative(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle tensor,
        IntPtr outputBuffer,
        UIntPtr outputBufferSize,
        out UIntPtr requiredSize)
    {
        return line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_tensor_get_name(tensor, outputBuffer, outputBufferSize, out requiredSize),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_tensor_get_name(tensor, outputBuffer, outputBufferSize, out requiredSize),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_tensor_get_name(tensor, outputBuffer, outputBufferSize, out requiredSize),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
    }

    private static BridgeStatusCode GetLayerNameNative(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer,
        IntPtr outputBuffer,
        UIntPtr outputBufferSize,
        out UIntPtr requiredSize)
    {
        return line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_layer_get_name(layer, outputBuffer, outputBufferSize, out requiredSize),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_layer_get_name(layer, outputBuffer, outputBufferSize, out requiredSize),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_layer_get_name(layer, outputBuffer, outputBufferSize, out requiredSize),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
    }

    public static bool TryCreateRuntime(TensorRtApiLine line, SafeTensorRtObjectHandle logger, out string message)
    {
        return TryCreateRuntimeCore(GetBindings(line), logger, out message);
    }

    public static bool TryCreateBuilder(TensorRtApiLine line, out string message)
    {
        return TryCreateBuilderCore(GetBindings(line), out message);
    }

    public static bool TryRunTrt10MinimalBuildChain(out string message)
    {
        return TryRunTrtMinimalBuildChain(Trt10Bindings, out message);
    }

    public static bool TryBuildTrt10SerializedNetworkOnly(out string message)
    {
        return TryBuildSerializedNetworkOnly(Trt10Bindings, out message);
    }

    public static bool TryRunTrt8MinimalBuildChain(out string message)
    {
        return TryRunTrtMinimalBuildChain(Trt8Bindings, out message);
    }

    public static bool TryBuildTrt8SerializedNetworkOnly(out string message)
    {
        return TryBuildSerializedNetworkOnly(Trt8Bindings, out message);
    }

    private delegate BridgeStatusCode RuntimeCreateDelegate(SafeTensorRtObjectHandle logger, out SafeTensorRtObjectHandle runtime);
    private delegate BridgeStatusCode BuilderCreateDelegate(SafeTensorRtObjectHandle logger, out SafeTensorRtObjectHandle builder);
    private delegate BridgeStatusCode ConfigCreateDelegate(SafeTensorRtObjectHandle builder, out SafeTensorRtObjectHandle config);
    private delegate BridgeStatusCode NetworkCreateDelegate(SafeTensorRtObjectHandle builder, uint flags, out SafeTensorRtObjectHandle network);
    private delegate BridgeStatusCode SerializedBuildDelegate(SafeTensorRtObjectHandle builder, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle config, out SafeTensorRtObjectHandle hostMemory);
    private delegate BridgeStatusCode DeserializeHostMemoryDelegate(SafeTensorRtObjectHandle runtime, SafeTensorRtObjectHandle hostMemory, out SafeTensorRtObjectHandle engine);
    private delegate BridgeStatusCode ExecutionContextCreateDelegate(SafeTensorRtObjectHandle engine, out SafeTensorRtObjectHandle context);

    private sealed class TensorRtLineBindings
    {
        public TensorRtLineBindings(
            TensorRtApiLine line,
            string lineName,
            QueryAdapterInfoDelegate queryAdapterInfo,
            LoggerCreateDelegate createLogger,
            RuntimeCreateDelegate runtimeCreate,
            BuilderCreateDelegate builderCreate,
            ConfigCreateDelegate configCreate,
            NetworkCreateDelegate networkCreate,
            SerializedBuildDelegate serializedBuild,
            DeserializeHostMemoryDelegate deserializeHostMemory,
            ExecutionContextCreateDelegate executionContextCreate)
        {
            Line = line;
            LineName = lineName;
            QueryAdapterInfo = queryAdapterInfo;
            CreateLogger = createLogger;
            RuntimeCreate = runtimeCreate;
            BuilderCreate = builderCreate;
            ConfigCreate = configCreate;
            NetworkCreate = networkCreate;
            SerializedBuild = serializedBuild;
            DeserializeHostMemory = deserializeHostMemory;
            ExecutionContextCreate = executionContextCreate;
        }

        public TensorRtApiLine Line { get; }
        public string LineName { get; }
        public QueryAdapterInfoDelegate QueryAdapterInfo { get; }
        public LoggerCreateDelegate CreateLogger { get; }
        public RuntimeCreateDelegate RuntimeCreate { get; }
        public BuilderCreateDelegate BuilderCreate { get; }
        public ConfigCreateDelegate ConfigCreate { get; }
        public NetworkCreateDelegate NetworkCreate { get; }
        public SerializedBuildDelegate SerializedBuild { get; }
        public DeserializeHostMemoryDelegate DeserializeHostMemory { get; }
        public ExecutionContextCreateDelegate ExecutionContextCreate { get; }
    }

    private static TensorRtLineBindings GetBindings(TensorRtApiLine line)
    {
        return line switch
        {
            TensorRtApiLine.TensorRt8 when !IsBridgeBuiltForTensorRt11() => Trt8Bindings,
            TensorRtApiLine.TensorRt10 when !IsBridgeBuiltForTensorRt11() => Trt10Bindings,
            TensorRtApiLine.TensorRt8 or TensorRtApiLine.TensorRt10 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "TensorRT 8/10 adapter exports are not available when the bridge is built against TensorRT 11."),
            TensorRtApiLine.TensorRt11 when IsBridgeBuiltForTensorRt11() => Trt11Bindings,
            TensorRtApiLine.TensorRt11 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "TensorRT 11 adapter exports are only available when the bridge is built against TensorRT 11."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
    }

    private static bool IsBridgeBuiltForTensorRt11()
    {
        NativeBuildInfo buildInfo = GetBuildInfo();
        string version = Utf8Interop.ReadString(buildInfo.TensorRtVersion);
        return version.StartsWith("11.", StringComparison.Ordinal);
    }

    private static bool TryRunTrtMinimalBuildChain(
        TensorRtLineBindings bindings,
        out string message)
    {
        using SafeTensorRtObjectHandle logger = CreateLogger(bindings.Line);
        BridgeStatusCode status = bindings.RuntimeCreate(logger, out SafeTensorRtObjectHandle runtime);
        if (status != BridgeStatusCode.Ok)
        {
            message = GetLastErrorMessageOrFallback($"{bindings.LineName} runtime creation failed with status '{status}'.");
            return false;
        }

        using (runtime)
        {
            status = bindings.BuilderCreate(logger, out SafeTensorRtObjectHandle builder);
            if (status != BridgeStatusCode.Ok)
            {
                message = GetLastErrorMessageOrFallback($"{bindings.LineName} builder creation failed with status '{status}'.");
                return false;
            }

            using (builder)
            {
                status = bindings.ConfigCreate(builder, out SafeTensorRtObjectHandle config);
                if (status != BridgeStatusCode.Ok)
                {
                    message = GetLastErrorMessageOrFallback($"{bindings.LineName} builder config creation failed with status '{status}'.");
                    return false;
                }

                using (config)
                {
                    status = bindings.NetworkCreate(builder, 0, out SafeTensorRtObjectHandle network);
                    if (status != BridgeStatusCode.Ok)
                    {
                        message = GetLastErrorMessageOrFallback($"{bindings.LineName} network creation failed with status '{status}'.");
                        return false;
                    }

                    using (network)
                    {
                        status = bindings.SerializedBuild(builder, network, config, out SafeTensorRtObjectHandle hostMemory);
                        if (status != BridgeStatusCode.Ok)
                        {
                            message = GetLastErrorMessageOrFallback($"{bindings.LineName} serialized network build failed with status '{status}'.");
                            return false;
                        }

                        using (hostMemory)
                        {
                            status = bindings.DeserializeHostMemory(runtime, hostMemory, out SafeTensorRtObjectHandle engine);
                            if (status != BridgeStatusCode.Ok)
                            {
                                message = GetLastErrorMessageOrFallback($"{bindings.LineName} engine deserialization failed with status '{status}'.");
                                return false;
                            }

                            using (engine)
                            {
                                status = bindings.ExecutionContextCreate(engine, out SafeTensorRtObjectHandle context);
                                if (status != BridgeStatusCode.Ok)
                                {
                                    message = GetLastErrorMessageOrFallback($"{bindings.LineName} execution context creation failed with status '{status}'.");
                                    return false;
                                }

                                using (context)
                                {
                                    message = $"{bindings.LineName} minimal build chain completed successfully.";
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private static bool TryBuildSerializedNetworkOnly(
        TensorRtLineBindings bindings,
        out string message)
    {
        using SafeTensorRtObjectHandle logger = CreateLogger(bindings.Line);
        BridgeStatusCode status = bindings.BuilderCreate(logger, out SafeTensorRtObjectHandle builder);
        if (status != BridgeStatusCode.Ok)
        {
            message = GetLastErrorMessageOrFallback($"{bindings.LineName} builder creation failed with status '{status}'.");
            return false;
        }

        using (builder)
        {
            status = bindings.NetworkCreate(builder, 0, out SafeTensorRtObjectHandle network);
            if (status != BridgeStatusCode.Ok)
            {
                message = GetLastErrorMessageOrFallback($"{bindings.LineName} network creation failed with status '{status}'.");
                return false;
            }

            using (network)
            {
                status = bindings.ConfigCreate(builder, out SafeTensorRtObjectHandle config);
                if (status != BridgeStatusCode.Ok)
                {
                    message = GetLastErrorMessageOrFallback($"{bindings.LineName} builder config creation failed with status '{status}'.");
                    return false;
                }

                using (config)
                {
                    status = bindings.SerializedBuild(builder, network, config, out SafeTensorRtObjectHandle hostMemory);
                    if (status != BridgeStatusCode.Ok)
                    {
                        message = GetLastErrorMessageOrFallback($"{bindings.LineName} serialized network build failed with status '{status}'.");
                        return false;
                    }

                    using (hostMemory)
                    {
                        message = $"{bindings.LineName} serialized network build completed successfully.";
                        return true;
                    }
                }
            }
        }
    }

    private static TensorRtWeights.PinnedScope? PinOptionalWeights(TensorRtWeights? weights)
    {
        if (weights == null || weights.IsEmpty)
        {
            return null;
        }

        return weights.Pin();
    }

    private static TensorRtDataType GetScaleWeightsDataType(TensorRtWeights? shift, TensorRtWeights? scale, TensorRtWeights? power)
    {
        if (shift != null && !shift.IsEmpty)
        {
            return shift.DataType;
        }

        if (scale != null && !scale.IsEmpty)
        {
            return scale.DataType;
        }

        if (power != null && !power.IsEmpty)
        {
            return power.DataType;
        }

        return TensorRtDataType.Float;
    }

    private static void ValidateOptionalWeightsDataType(TensorRtDataType expected, TensorRtWeights? weights, string argumentName)
    {
        if (weights == null || weights.IsEmpty)
        {
            return;
        }

        if (weights.DataType != expected)
        {
            throw new ArgumentException("All TensorRT weights passed to a single layer must use the same data type.", argumentName);
        }
    }
}
