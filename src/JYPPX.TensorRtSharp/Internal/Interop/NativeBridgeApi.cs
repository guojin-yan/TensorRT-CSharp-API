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

}
